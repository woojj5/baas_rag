"""Abstract embedding client interface."""
from abc import ABC, abstractmethod
import numpy as np
from typing import List


class EmbeddingClient(ABC):
    """Abstract base class for embedding clients."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple documents.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query.
        
        Args:
            text: Query text
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        pass

