# Pre-entrega 1: Cliente de LLM robusto y asíncrono

**Alumna:** Alexandra Blasetti Kuhn  
**Curso:** AI Engineering, Comisión #102135  
**Fecha de entrega:** 6 de noviembre de 2026

Implementación de un **Unified Async LLM Client** en Python (3.12 o superior; desarrollado y probado con Python 3.14) para **Claude (Anthropic)** y **Gemini (Google)**. Ambos proveedores comparten una interfaz común (`generate` y `stream`), la entrada y la configuración se validan con **Pydantic**, y los errores de red, autenticación y límite de tasa se devuelven como errores controlados en lugar de detener el programa.

El cliente intenta primero con **Claude**. Si Claude falla (sin crédito, API key inválida o ausente, límite de tasa, error de red), pasa automáticamente a **Gemini**.

## Cumplimiento de la consigna

| Requisito | Dónde se cumple |
|---|---|
| Intercambiabilidad: interfaz común para varios proveedores | `BaseLLMClient` en `llm_client/base.py`, implementada por `AnthropicClient` y `GeminiClient` |
| Asincronía: llamadas no bloqueantes con `async/await` | SDKs asíncronos: `AsyncAnthropic` y `client.aio` de `google-genai` |
| Streaming con generador asíncrono y `yield` | Método `stream()` en `base.py` y `_stream()` en cada cliente |
| Validación con Pydantic (mensajes y configuración del modelo) | `llm_client/schemas.py`: `ChatMessage`, `ModelConfig` (temperatura de 0 a 2, `max_tokens`, etc.) |
| Clase `AsyncLLMManager` que carga el proveedor según la configuración | `llm_client/manager.py`, a partir de la variable `LLM_PROVIDERS` |
| Captura de errores de red y de rate limiting sin crash | `LLMError` + reintentos con backoff exponencial en `base.py` |
| Script de prueba en modo normal y streaming | `main.py` |
| Variables de entorno de ejemplo | `.env.example` |

### Nota sobre la elección de proveedores

La consigna propone OpenAI y Anthropic. En esta entrega se usa **Gemini en lugar de OpenAI**, porque Gemini ofrece un plan gratuito que permite probar el proyecto sin costo. La arquitectura no cambia: agregar OpenAI solo requiere una nueva subclase de `BaseLLMClient` y una entrada en el registro de `manager.py`.

## Estructura del proyecto

```
.
├── llm_client/
│   ├── __init__.py          # Exporta las clases públicas del paquete
│   ├── schemas.py           # Pydantic: ChatMessage, ModelConfig, ModelResponse, StreamChunk, LLMError
│   ├── base.py              # BaseLLMClient (ABC): reintentos con backoff y errores controlados
│   ├── anthropic_client.py  # AnthropicClient (AsyncAnthropic)
│   ├── gemini_client.py     # GeminiClient (google-genai, cliente asíncrono .aio)
│   └── manager.py           # AsyncLLMManager: orden de prioridad y fallback entre proveedores
├── main.py                  # Script de validación (modo normal + streaming)
├── .env.example             # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt
└── README.md
```

## Requisitos previos

- Python 3.12 o superior. El código usa sintaxis introducida en Python 3.12 (alias de tipo con `type`) y fue desarrollado y probado con Python 3.14.
- Una API key de Gemini (gratuita): se crea en Google AI Studio (aistudio.google.com → "Get API key").
- Opcional: una API key de Claude, que se crea en la consola de Anthropic. Si no tiene crédito, el programa usará Gemini automáticamente.

## Instalación

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/ablasettikuhn/ai-engineering-pre-entrega-1.git
   cd ai-engineering-pre-entrega-1
   ```

2. Crear y activar el entorno virtual:

   ```bash
   # Windows (Command Prompt)
   py -m venv .venv
   .venv\Scripts\activate

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instalar las dependencias (`anthropic`, `google-genai`, `pydantic`, `python-dotenv`):

   ```bash
   pip install -r requirements.txt
   ```

4. Crear el archivo `.env` a partir de la plantilla y completar las claves:

   ```bash
   copy .env.example .env      # Windows
   cp .env.example .env        # macOS / Linux
   ```

El archivo `.env` está incluido en `.gitignore`, así que las claves nunca se suben al repositorio.

## Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `ANTHROPIC_API_KEY` | No* | Clave de Claude |
| `GEMINI_API_KEY` | No* | Clave de Gemini |
| `LLM_PROVIDERS` | No (default `anthropic,gemini`) | Orden de prioridad para el fallback |
| `ANTHROPIC_MODEL` | No (default `claude-haiku-5-5`) | Modelo de Claude |
| `GEMINI_MODEL` | No (default `gemini-3.1-flash-lite`) | Modelo de Gemini (Flash-Lite, incluido en el plan gratuito) |
| `LLM_MAX_TOKENS` | No (default `1024`) | Máximo de tokens de salida |
| `LLM_TIMEOUT` | No (default `30`) | Timeout por petición, en segundos |
| `LLM_MAX_RETRIES` | No (default `2`) | Reintentos ante errores transitorios, por proveedor |
| `LLM_TEMPERATURE` | No | Entre 0 y 2. Solo se envía a Gemini |

\* Se necesita al menos una de las dos claves. Un proveedor sin clave se omite con un aviso.

Pydantic valida todos los valores antes de llamar a la API. Por ejemplo, `LLM_TEMPERATURE=2.5`, `LLM_MAX_TOKENS=0` o un proveedor desconocido se rechazan con un mensaje claro.

## Ejecutar el script de prueba

Con el entorno virtual activado:

```bash
python main.py
```

El script hace la pregunta "¿Qué es la entropía?" dos veces:

1. **Modo normal:** muestra la respuesta completa, qué proveedor respondió, la latencia y los tokens usados.
2. **Modo streaming:** el texto aparece a medida que llega de la API.

Ejemplo de salida cuando Claude no tiene crédito y responde Gemini:

```
Orden de proveedores: anthropic → gemini

=== Modo normal ===
  ⚠️  anthropic falló (quota). Probando el siguiente...
[respondió: gemini / gemini-3.1-flash-lite]

La entropía es una medida del desorden de un sistema...

(1834.2 ms | 21 tokens entrada | 68 tokens salida)

=== Modo streaming ===
  ⚠️  anthropic falló (quota). Probando el siguiente...
[respondiendo: gemini]

La entropía es una medida del desorden de un sistema...
```

Para probar un solo proveedor, cambia temporalmente `LLM_PROVIDERS` en el `.env` (por ejemplo, `LLM_PROVIDERS=gemini`).

## Uso en código

```python
import asyncio
from dotenv import load_dotenv
from llm_client import AsyncLLMManager

async def main():
    load_dotenv()
    async with AsyncLLMManager.from_env() as llm:
        resp = await llm.generate("¿Qué es la entropía?")
        print(resp.provider, resp.content if resp.ok else resp.error.message)

        async for chunk in llm.stream("Explícalo en una frase"):
            if chunk.error:
                print(chunk.error.message)
                break
            print(chunk.text, end="", flush=True)

asyncio.run(main())
```

## Decisiones de diseño

**Asincronía real.** Se usan `AsyncAnthropic` y el cliente asíncrono de Gemini (`client.aio`) con `await`. Las esperas entre reintentos usan `asyncio.sleep` y nunca `time.sleep`, para no bloquear el event loop.

**Streaming.** `stream()` es un generador asíncrono que recorre el stream del SDK con `async for` y hace `yield` de cada fragmento como un `StreamChunk`.

**Errores controlados.** Las excepciones de cada SDK se traducen a un `LLMError` con un tipo (`authentication`, `quota`, `rate_limit`, `connection`, `timeout`, `bad_request`, `server`). `generate()` lo devuelve en `ModelResponse.error` y `stream()` emite un último `StreamChunk` con el error, así que ninguna falla de la API detiene el programa.

**Reintentos y fallback.** Los errores transitorios (429, red, timeout, 5xx) se reintentan con backoff exponencial dentro de cada proveedor. Si el proveedor sigue fallando, o el error no es transitorio (por ejemplo, falta de crédito), el manager pasa al siguiente. En streaming solo se cambia de proveedor si todavía no se mostró texto, para no mezclar dos respuestas.

**`temperature`.** Los modelos actuales de Claude ya no aceptan `temperature` (la API responde con error 400), así que no se envía a Claude. El parámetro es opcional, se valida entre 0 y 2 y solo se aplica a Gemini.
