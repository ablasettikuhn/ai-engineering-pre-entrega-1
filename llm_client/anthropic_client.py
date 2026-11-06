from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from .base import BaseLLMClient
from .schemas import ChatMessage, ErrorType, LLMError, ModelConfig, Provider, TokenUsage


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        # max_retries=0: los reintentos los hace BaseLLMClient (evita reintentos dobles).
        # Si api_key es None, el SDK lee ANTHROPIC_API_KEY del entorno.
        self._client = AsyncAnthropic(api_key=api_key, timeout=config.timeout, max_retries=0)

    def _params(self, messages: list[ChatMessage]) -> dict:
        # Claude recibe el mensaje "system" en un parámetro aparte, no dentro de messages.
        # No se envía temperature: los modelos actuales de Claude ya no la admiten.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        params: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [m.model_dump() for m in messages if m.role != "system"],
        }
        if system:
            params["system"] = system
        return params

    async def _generate(self, messages: list[ChatMessage]) -> tuple[str, TokenUsage | None]:
        response = await self._client.messages.create(**self._params(messages))
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return text, usage

    async def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async with self._client.messages.stream(**self._params(messages)) as stream:
            async for text in stream.text_stream:
                yield text

    def _classify_error(self, exc: Exception) -> LLMError:
        status = getattr(exc, "status_code", None)
        message = _api_message(exc)
        # El orden importa: APITimeoutError hereda de APIConnectionError.
        match exc:
            case anthropic.AuthenticationError() | anthropic.PermissionDeniedError():
                kind, retryable = ErrorType.AUTHENTICATION, False
            case anthropic.RateLimitError():
                kind, retryable = ErrorType.RATE_LIMIT, True
            case anthropic.APITimeoutError():
                kind, retryable = ErrorType.TIMEOUT, True
            case anthropic.APIConnectionError():
                kind, retryable = ErrorType.CONNECTION, True
            case anthropic.BadRequestError() if "credit balance" in message.lower():
                kind, retryable = ErrorType.QUOTA, False  # cuenta sin crédito
            case anthropic.BadRequestError() | anthropic.NotFoundError():
                kind, retryable = ErrorType.BAD_REQUEST, False
            case anthropic.APIStatusError() if status is not None and status >= 500:
                kind, retryable = ErrorType.SERVER, True
            case _:
                kind, retryable = ErrorType.UNKNOWN, False
        return LLMError(provider=self.provider, type=kind, message=message,
                        retryable=retryable, status_code=status)

    async def close(self) -> None:
        await self._client.close()


def _api_message(exc: Exception) -> str:
    """Extrae body["error"]["message"] si existe; si no, str(exc)."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return str(exc)
