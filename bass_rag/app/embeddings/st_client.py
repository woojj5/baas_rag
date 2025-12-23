"""SentenceTransformers embedding client."""
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from app.config import Config
from app.embeddings.base import EmbeddingClient
from app.utils.cache import embedding_cache


class SentenceTransformersEmbeddingClient(EmbeddingClient):
    """SentenceTransformers-based embedding client with caching."""
    
    def __init__(self, model_name: str = None):
        model_name = model_name or Config.EMBED_MODEL_NAME
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed multiple documents with caching."""
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = embedding_cache.get(text)
            if cached is not None:
                embeddings.append((i, cached))
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            new_embeddings = self.model.encode(
                texts_to_embed,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            # Store in cache and add to results
            for idx, text, embedding in zip(indices_to_embed, texts_to_embed, new_embeddings):
                embedding_cache.set(text, embedding)
                embeddings.append((idx, embedding))
        
        # Sort by original index and return as array
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query with caching."""
        # Check cache first
        cached = embedding_cache.get(text)
        if cached is not None:
            return cached
        
        # Embed and cache
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embedding_cache.set(text, embedding)
        return embedding

