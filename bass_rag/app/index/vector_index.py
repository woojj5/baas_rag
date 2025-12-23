"""FAISS-based vector index."""
import faiss
import numpy as np
from typing import List, Tuple
from app.models.schemas import Chunk


class VectorIndex:
    """FAISS-based vector index with chunk metadata."""
    
    def __init__(self, dimension: int):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
        """
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []  # Store chunks with metadata
    
    def add(self, embeddings: np.ndarray, chunks: List[Chunk]):
        """
        Add embeddings and chunks to index.
        
        Args:
            embeddings: numpy array of shape (n_chunks, dimension)
            chunks: List of Chunk objects
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector (1D array)
            top_k: Number of results to return
            
        Returns:
            List of (Chunk, score) tuples
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunks):
                score = float(distances[0][i])
                chunk = self.chunks[idx]
                results.append((chunk, score))
        
        return results
    
    def __len__(self) -> int:
        """Return number of chunks in index."""
        return len(self.chunks)

