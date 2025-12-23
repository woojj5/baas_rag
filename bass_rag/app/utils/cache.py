"""Caching utilities for embeddings and query expansions."""
from functools import lru_cache
from typing import List, Tuple, Optional
import hashlib
import numpy as np
from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """LRU cache for embeddings to avoid re-embedding the same text."""
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or Config.EMBEDDING_CACHE_SIZE
        self._cache: dict = {}
        self._access_order: list = []  # For LRU eviction
    
    def _get_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        if not Config.ENABLE_EMBEDDING_CACHE:
            return None
        
        key = self._get_key(text)
        if key in self._cache:
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key].copy()
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""
        if not Config.ENABLE_EMBEDDING_CACHE:
            return
        
        key = self._get_key(text)
        
        # Evict least recently used if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            if self._access_order:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
        
        self._cache[key] = embedding.copy()
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class QueryExpansionCache:
    """LRU cache for query expansion results."""
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or Config.QUERY_EXPANSION_CACHE_SIZE
        self._cache: dict = {}
        self._access_order: list = []
    
    def _get_key(self, query: str, max_expansions: int) -> str:
        """Generate cache key from query and parameters."""
        key_str = f"{query}:{max_expansions}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def get(self, query: str, max_expansions: int) -> Optional[List[str]]:
        """Get query expansion from cache."""
        if not Config.ENABLE_QUERY_EXPANSION_CACHE:
            return None
        
        key = self._get_key(query, max_expansions)
        if key in self._cache:
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key].copy()
        return None
    
    def set(self, query: str, max_expansions: int, expansions: List[str]):
        """Store query expansion in cache."""
        if not Config.ENABLE_QUERY_EXPANSION_CACHE:
            return
        
        key = self._get_key(query, max_expansions)
        
        # Evict least recently used if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            if self._access_order:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
        
        self._cache[key] = expansions.copy()
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class ChunkEmbeddingCache:
    """LRU cache for chunk embeddings by chunk ID."""
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or Config.EMBEDDING_CACHE_SIZE * 2  # Larger cache for chunks
        self._cache: dict = {}  # chunk_id -> embedding
        self._access_order: list = []  # For LRU eviction
    
    def get(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get chunk embedding from cache by chunk ID."""
        if not Config.ENABLE_EMBEDDING_CACHE:
            return None
        
        if chunk_id in self._cache:
            # Move to end (most recently used)
            if chunk_id in self._access_order:
                self._access_order.remove(chunk_id)
            self._access_order.append(chunk_id)
            return self._cache[chunk_id].copy()
        return None
    
    def set(self, chunk_id: str, embedding: np.ndarray):
        """Store chunk embedding in cache."""
        if not Config.ENABLE_EMBEDDING_CACHE:
            return
        
        # Evict least recently used if cache is full
        if len(self._cache) >= self.max_size and chunk_id not in self._cache:
            if self._access_order:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
        
        self._cache[chunk_id] = embedding.copy()
        if chunk_id in self._access_order:
            self._access_order.remove(chunk_id)
        self._access_order.append(chunk_id)
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def invalidate(self, chunk_id: str):
        """Invalidate a specific chunk from cache."""
        if chunk_id in self._cache:
            del self._cache[chunk_id]
            if chunk_id in self._access_order:
                self._access_order.remove(chunk_id)


# Global cache instances
embedding_cache = EmbeddingCache()
query_expansion_cache = QueryExpansionCache()
chunk_embedding_cache = ChunkEmbeddingCache()

