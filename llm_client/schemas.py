"""Esquemas Pydantic: validan todo lo que entra y sale de los clientes LLM."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


type Role = Literal["system", "user", "assistant"]  # alias de tipo (Python 3.12)


class ChatMessage(BaseModel):
    """Un mensaje de la conversación."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("El contenido del mensaje no puede estar vacío.")
        return value


class ModelConfig(BaseModel):
    """Configuración del modelo y parámetros de generación."""

    provider: Provider
    model: str = Field(min_length=1)
    # Opcional: si es None no se envía. Solo Gemini la usa (Claude ya no la admite).
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=32_000)
    timeout: float = Field(default=30.0, gt=0, description="Segundos por petición")
    max_retries: int = Field(default=2, ge=0, le=10)


class ErrorType(StrEnum):
    AUTHENTICATION = "authentication"  # API key inválida o sin permisos
    QUOTA = "quota"                    # sin crédito / saldo insuficiente
    RATE_LIMIT = "rate_limit"          # demasiadas peticiones (429)
    CONNECTION = "connection"          # fallo de red
    TIMEOUT = "timeout"
    BAD_REQUEST = "bad_request"        # parámetros o modelo inválidos
    SERVER = "server"                  # errores 5xx / sobrecarga
    UNKNOWN = "unknown"


class LLMError(BaseModel):
    """Error controlado: se devuelve en lugar de dejar que la excepción rompa el programa."""

    provider: Provider
    type: ErrorType
    message: str
    retryable: bool
    status_code: int | None = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ModelResponse(BaseModel):
    """Respuesta normalizada, igual para cualquier proveedor."""

    provider: Provider
    model: str
    content: str = ""
    usage: TokenUsage | None = None
    latency_ms: float | None = None
    error: LLMError | None = None
    # Errores de los proveedores que fallaron antes (cuando hubo fallback).
    previous_errors: list[LLMError] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


class StreamChunk(BaseModel):
    """Fragmento emitido durante el streaming: trae texto o un error controlado."""

    provider: Provider
    text: str = ""
    error: LLMError | None = None
