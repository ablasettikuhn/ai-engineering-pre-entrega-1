"""Script de validación: pregunta "¿Qué es la entropía?" en modo normal y en streaming.

Claude se intenta primero; si falla (sin crédito, sin API key, límite...), responde Gemini.
"""

import asyncio
import logging

from dotenv import load_dotenv

from llm_client import AsyncLLMManager

QUESTION = "¿Qué es la entropía?"
SYSTEM = "Responde en español, de forma clara y en no más de 3 oraciones."


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="  ⚠️  %(message)s")

    try:
        manager = AsyncLLMManager.from_env()
    except ValueError as exc:  # falta de API keys o configuración inválida (Pydantic)
        print(f"❌ Configuración inválida: {exc}")
        return

    async with manager:
        print(f"Orden de proveedores: {' → '.join(manager.providers)}")

        # 1) Modo normal
        print("\n=== Modo normal ===")
        response = await manager.generate(QUESTION, system=SYSTEM)
        if response.ok:
            print(f"[respondió: {response.provider} / {response.model}]\n")
            print(response.content)
            if response.usage:
                print(f"\n({response.latency_ms} ms | {response.usage.input_tokens} tokens entrada"
                      f" | {response.usage.output_tokens} tokens salida)")
        else:
            print(f"❌ Ningún proveedor respondió. Último error ({response.error.type}): {response.error.message}")

        # 2) Modo streaming
        print("\n=== Modo streaming ===")
        current = None
        async for chunk in manager.stream(QUESTION, system=SYSTEM):
            if chunk.error:
                print(f"\n❌ Error controlado ({chunk.error.provider}, {chunk.error.type}): {chunk.error.message}")
                break
            if chunk.provider != current:
                current = chunk.provider
                print(f"[respondiendo: {current}]\n")
            print(chunk.text, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
