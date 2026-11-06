"""AsyncLLMManager: prueba los proveedores en orden (Claude primero, Gemini como respaldo)."""

import logging
import os
from collections.abc import AsyncIterator

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .gemini_client import GeminiClient
from .schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, Provider, StreamChunk

logger = logging.getLogger(__name__)

type Prompt = str | list[ChatMessage]

CLIENTS: dict[Provider, type[BaseLLMClient]] = {
    Provider.ANTHROPIC: AnthropicClient,
    Provider.GEMINI: GeminiClient,
}
API_KEY_VARS = {Provider.ANTHROPIC: "ANTHROPIC_API_KEY", Provider.GEMINI: "GEMINI_API_KEY"}
DEFAULT_MODELS = {Provider.ANTHROPIC: "claude-haiku-5-5", Provider.GEMINI: "gemini-3.1-flash-lite"}
DEFAULT_ORDER = "anthropic,gemini"


class MissingAPIKeyError(ValueError):
    pass


class AsyncLLMManager:
    def __init__(self, configs: list[ModelConfig]) -> None:
        self.clients: list[BaseLLMClient] = []
        for config in configs:
            key_var = API_KEY_VARS[config.provider]
            if not os.getenv(key_var):
                logger.warning("Se omite %s: falta la variable %s.", config.provider, key_var)
                continue
            self.clients.append(CLIENTS[config.provider](config))
        if not self.clients:
            raise MissingAPIKeyError("Ningún proveedor tiene API key configurada en el .env.")

    @classmethod
    def from_env(cls, providers: list[str] | None = None) -> "AsyncLLMManager":
        """Lee el .env. El orden de LLM_PROVIDERS define la prioridad del fallback."""
        names = providers or os.getenv("LLM_PROVIDERS", DEFAULT_ORDER).split(",")
        configs = [cls._config_from_env(Provider(n.strip().lower())) for n in names if n.strip()]
        return cls(configs)

    @staticmethod
    def _config_from_env(provider: Provider) -> ModelConfig:
        raw = {
            "provider": provider,
            "model": os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider],
            "temperature": os.getenv("LLM_TEMPERATURE"),
            "max_tokens": os.getenv("LLM_MAX_TOKENS"),
            "timeout": os.getenv("LLM_TIMEOUT"),
            "max_retries": os.getenv("LLM_MAX_RETRIES"),
        }
        # Pydantic convierte y valida los strings del entorno ("0.7" -> 0.7).
        return ModelConfig(**{k: v for k, v in raw.items() if v not in (None, "")})

    @property
    def providers(self) -> list[Provider]:
        return [c.provider for c in self.clients]

    @staticmethod
    def _to_messages(prompt: Prompt, system: str | None) -> list[ChatMessage]:
        if isinstance(prompt, str):
            messages = [ChatMessage(role="user", content=prompt)]
        else:
            messages = list(prompt)
        if not messages:
            raise ValueError("Se necesita al menos un mensaje.")
        if system:
            messages.insert(0, ChatMessage(role="system", content=system))
        return messages

    async def generate(self, prompt: Prompt, system: str | None = None) -> ModelResponse:
        """Prueba cada proveedor en orden y devuelve la primera respuesta correcta."""
        messages = self._to_messages(prompt, system)
        failures: list[LLMError] = []
        response: ModelResponse | None = None
        for index, client in enumerate(self.clients):
            response = await client.generate(messages)
            if response.ok:
                response.previous_errors = failures
                return response
            failures.append(response.error)
            if index < len(self.clients) - 1:
                logger.warning(
                    "%s falló (%s). Probando el siguiente...", client.provider, response.error.type
                )
        response.previous_errors = failures[:-1]
        return response  # todos fallaron: se devuelve el último error

    async def stream(self, prompt: Prompt, system: str | None = None) -> AsyncIterator[StreamChunk]:
        """Streaming con fallback: si un proveedor falla antes de emitir texto, pasa al siguiente."""
        messages = self._to_messages(prompt, system)
        for index, client in enumerate(self.clients):
            is_last = index == len(self.clients) - 1
            emitted = False
            async for chunk in client.stream(messages):
                if chunk.error:
                    if not emitted and not is_last:
                        logger.warning("%s falló (%s). Probando el siguiente...", client.provider, chunk.error.type)
                        break  # pasa al siguiente proveedor
                    yield chunk  # error a mitad de respuesta o último proveedor
                    return
                emitted = True
                yield chunk
            else:
                return  # el stream terminó bien

    async def close(self) -> None:
        for client in self.clients:
            await client.close()

    async def __aenter__(self) -> "AsyncLLMManager":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
