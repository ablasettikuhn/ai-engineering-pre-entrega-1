"""Clase base abstracta con la lógica común: reintentos, backoff y errores controlados."""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .schemas import (
    ChatMessage,
    ErrorType,
    LLMError,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamChunk,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Interfaz común. Cada proveedor solo implementa la llamada concreta a su SDK."""

    provider: Provider

    def __init__(self, config: ModelConfig) -> None:
        if config.provider is not self.provider:
            raise ValueError(f"{type(self).__name__} requiere provider={self.provider}.")
        self.config = config

    # ---- Lo que implementa cada proveedor --------------------------------------

    @abstractmethod
    async def _generate(self, messages: list[ChatMessage]) -> tuple[str, TokenUsage | None]:
        """Llamada sin streaming. Devuelve (texto, uso de tokens)."""

    @abstractmethod
    def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Generador asíncrono que hace `yield` de cada fragmento de texto."""

    @abstractmethod
    def _classify_error(self, exc: Exception) -> LLMError:
        """Traduce una excepción del SDK a un LLMError."""

    @abstractmethod
    async def close(self) -> None:
        """Cierra el cliente HTTP del SDK."""

    # ---- API pública -----------------------------------------------------------

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Respuesta completa. Nunca lanza errores de API: los devuelve en `error`."""
        start = time.perf_counter()
        for attempt in range(self.config.max_retries + 1):
            try:
                content, usage = await self._generate(messages)
                return ModelResponse(
                    provider=self.provider,
                    model=self.config.model,
                    content=content,
                    usage=usage,
                    latency_ms=round((time.perf_counter() - start) * 1000, 1),
                )
            except asyncio.CancelledError:
                raise  # la cancelación nunca se silencia
            except Exception as exc:
                error = self._classify_error(exc)
                if error.retryable and attempt < self.config.max_retries:
                    await self._wait_before_retry(attempt, error)
                    continue
                return ModelResponse(provider=self.provider, model=self.config.model, error=error)
        raise AssertionError("inalcanzable")

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Generador asíncrono. Ante un error emite un último StreamChunk con `error`."""
        for attempt in range(self.config.max_retries + 1):
            emitted = False
            try:
                async for text in self._stream(messages):
                    emitted = True
                    yield StreamChunk(provider=self.provider, text=text)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = self._classify_error(exc)
                # Solo se reintenta si aún no salió texto (para no duplicar la salida).
                if error.retryable and not emitted and attempt < self.config.max_retries:
                    await self._wait_before_retry(attempt, error)
                    continue
                yield StreamChunk(provider=self.provider, error=error)
                return

    # ---- Utilidades ------------------------------------------------------------

    async def _wait_before_retry(self, attempt: int, error: LLMError) -> None:
        delay = min(2**attempt, 20) + random.uniform(0, 0.5)  # 1s, 2s, 4s... + jitter
        logger.warning(
            "[%s] %s (intento %d/%d). Reintentando en %.1fs...",
            self.provider, error.type, attempt + 1, self.config.max_retries + 1, delay,
        )
        await asyncio.sleep(delay)  # asyncio.sleep, NO time.sleep: no bloquea el event loop

    def _network_error(self, exc: Exception) -> LLMError | None:
        """Detecta timeouts y fallos de red genéricos (httpx/httpx2/asyncio)."""
        names = {cls.__name__ for cls in type(exc).__mro__}
        if names & {"TimeoutError", "TimeoutException"}:
            return LLMError(provider=self.provider, type=ErrorType.TIMEOUT, message=str(exc) or "Timeout", retryable=True)
        if names & {"TransportError", "ConnectError", "NetworkError", "ConnectionError"}:
            return LLMError(provider=self.provider, type=ErrorType.CONNECTION, message=str(exc) or "Error de red", retryable=True)
        return None
