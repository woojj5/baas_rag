"""Ollama LLM client implementation."""
import httpx
import numpy as np
from typing import List, Dict, Optional
from app.config import Config
from app.llm.base import LLMClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaLLMClient(LLMClient):
    """Ollama chat API client."""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.model = model or Config.OLLAMA_MODEL
    
    async def chat(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
        inputs_embeds: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate response using Ollama chat API.
        
        Args:
            system_prompt: System prompt
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            inputs_embeds: Optional pre-computed embeddings (Ollama doesn't support this, will fallback to text)
            
        Returns:
            Generated text response
        """
        # Note: Ollama API doesn't support inputs_embeds directly
        # If provided, we log a warning and fallback to text-based input
        if inputs_embeds is not None:
            logger.warning(
                "inputs_embeds provided but Ollama API doesn't support it. "
                "Falling back to text-based input. "
                "Consider using a different LLM backend that supports inputs_embeds."
            )
        
        url = f"{self.base_url}/api/chat"
        
        # Build messages list with system prompt
        chat_messages = []
        if system_prompt:
            chat_messages.append({
                "role": "system",
                "content": system_prompt
            })
        chat_messages.extend(messages)
        
        # Optimize Ollama options for faster generation
        # num_ctx: Limit context window to reduce processing time (smaller = faster)
        # For 27B model, smaller context window = faster processing
        estimated_context_size = len(system_prompt) + sum(len(m.get("content", "")) for m in chat_messages)
        # Set num_ctx to actual size + small buffer (don't over-allocate)
        # Smaller context = faster KV cache processing
        num_ctx = min(int(estimated_context_size * 1.2), 2048)  # Cap at 2048 for speed (was 4096)
        
        payload = {
            "model": self.model,
            "messages": chat_messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_ctx": num_ctx,  # Limit context window for faster processing
                "repeat_penalty": 1.1,  # Prevent repetition (lower = faster)
                "top_p": 0.9,  # Nucleus sampling (slightly lower for speed)
                "top_k": 40,  # Limit vocabulary (lower = faster)
                "num_thread": 4  # Optimize thread count for 27B model
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Extract message content from response
                message = data.get("message", {})
                return message.get("content", "").strip()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama API HTTP error: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

