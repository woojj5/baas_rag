"""Abstract LLM client interface."""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def chat(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
        inputs_embeds: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate response using chat API.
        
        Args:
            system_prompt: System prompt/instruction
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            inputs_embeds: Optional pre-computed embeddings (for TokenProjector)
            
        Returns:
            Generated text response
        """
        pass

