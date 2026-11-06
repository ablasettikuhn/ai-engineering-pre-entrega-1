from collections.abc import AsyncIterator

from google import genai
from google.genai import errors, types

from .base import BaseLLMClient
from .schemas import ChatMessage, ErrorType, LLMError, ModelConfig, Provider, TokenUsage


class GeminiClient(BaseLLMClient):
    provider = Provider.GEMINI

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        # Si api_key es None, el SDK lee GEMINI_API_KEY del entorno.
        # El timeout de google-genai va en milisegundos.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(config.timeout * 1000)),
        )

    def _params(self, messages: list[ChatMessage]) -> dict:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        # Gemini usa role="model" en lugar de "assistant".
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=self.config.max_tokens,
            temperature=self.config.temperature,  # None = no se envía
        )
        return {"model": self.config.model, "contents": contents, "config": config}

    async def _generate(self, messages: list[ChatMessage]) -> tuple[str, TokenUsage | None]:
        # client.aio = versión asíncrona del SDK
        response = await self._client.aio.models.generate_content(**self._params(messages))
        usage = None
        if meta := response.usage_metadata:
            usage = TokenUsage(
                input_tokens=meta.prompt_token_count or 0,
                output_tokens=meta.candidates_token_count or 0,
            )
        return response.text or "", usage

    async def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.aio.models.generate_content_stream(**self._params(messages))
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    def _classify_error(self, exc: Exception) -> LLMError:
        if isinstance(exc, errors.APIError):
            code = exc.code or 0
            message = exc.message or str(exc)
            # Gemini responde 400 (no 401) cuando la API key es inválida.
            if code in (401, 403) or "api key" in message.lower():
                kind, retryable = ErrorType.AUTHENTICATION, False
            elif code == 429:  # RESOURCE_EXHAUSTED: límite del plan gratuito
                kind, retryable = ErrorType.RATE_LIMIT, True
            elif code >= 500:  # 503 UNAVAILABLE es frecuente en el plan gratuito
                kind, retryable = ErrorType.SERVER, True
            elif code in (400, 404):
                kind, retryable = ErrorType.BAD_REQUEST, False
            else:
                kind, retryable = ErrorType.UNKNOWN, False
            return LLMError(provider=self.provider, type=kind, message=message,
                            retryable=retryable, status_code=code or None)

        return self._network_error(exc) or LLMError(
            provider=self.provider, type=ErrorType.UNKNOWN, message=str(exc), retryable=False
        )

    async def close(self) -> None:
        await self._client.aio.aclose()
