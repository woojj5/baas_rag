"""Adapter to convert existing index format to REFRAG format."""
import json
from typing import List, Tuple
from pathlib import Path
import faiss
import numpy as np
from app.models.schemas import Chunk, Document
from app.index.vector_index import VectorIndex
from app.embeddings.st_client import SentenceTransformersEmbeddingClient
from app.utils.tokenizer import count_tokens
from app.config import Config


def convert_existing_index_to_refrag(
    faiss_index: faiss.Index,
    passages: List[str],
    embedding_client: SentenceTransformersEmbeddingClient,
    saved_embeddings: np.ndarray = None
) -> VectorIndex:
    """
    Convert existing FAISS index + passages to REFRAG VectorIndex format.
    
    Optimized: Uses saved embeddings if available to avoid re-embedding.
    
    Args:
        faiss_index: Existing FAISS index
        passages: List of passage strings
        embedding_client: Embedding client (for dimension)
        saved_embeddings: Optional pre-computed embeddings (for optimization)
        
    Returns:
        VectorIndex with Chunk objects
    """
    # Get dimension from FAISS index
    dimension = faiss_index.d
    vector_index = VectorIndex(dimension)
    
    # Convert passages to Chunk objects
    chunks = []
    for i, passage in enumerate(passages):
        # Extract metadata if present
        metadata = {}
        text = passage
        
        if "[메타데이터:" in passage:
            # Parse metadata
            metadata_start = passage.find("[메타데이터:")
            metadata_end = passage.find("]", metadata_start)
            if metadata_end != -1:
                metadata_str = passage[metadata_start + len("[메타데이터:"):metadata_end]
                # Parse metadata parts
                for part in metadata_str.split("|"):
                    part = part.strip()
                    if ":" in part:
                        key, value = part.split(":", 1)
                        metadata[key.strip()] = value.strip()
                
                # Remove metadata from text
                text = passage[metadata_end + 1:].strip()
        
        # Create chunk
        chunk = Chunk(
            id=f"chunk_{i}",
            document_id=metadata.get("문서ID", f"doc_{i}"),
            text=text,
            token_count=count_tokens(text),
            start_offset=0,
            end_offset=len(text),
            metadata=metadata if metadata else None
        )
        chunks.append(chunk)
    
    # Optimization: Use saved embeddings if available
    if saved_embeddings is not None and len(saved_embeddings) == len(chunks):
        # Verify dimension matches
        if saved_embeddings.shape[1] == dimension:
            embeddings = saved_embeddings.copy()
        else:
            # Dimension mismatch - need to re-embed
            texts = [chunk.text for chunk in chunks]
            embeddings = embedding_client.embed_documents(texts)
    else:
        # No saved embeddings or mismatch - re-embed (fallback)
        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_client.embed_documents(texts)
    
    # Add to vector index
    vector_index.add(embeddings, chunks)
    
    return vector_index

