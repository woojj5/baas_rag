"""Ollama HTTP client wrapper."""
import httpx
from app.config import Config


async def generate(prompt: str, model: str | None = None) -> str:
    """Generate text using Ollama API."""
    url = f"{Config.OLLAMA_BASE_URL}/api/generate"
    model_name = model or Config.OLLAMA_MODEL
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama API HTTP error: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}")

