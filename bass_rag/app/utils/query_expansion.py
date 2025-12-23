"""Query expansion utilities for semantic search."""
from typing import List, Optional
import numpy as np
from app.utils.cache import query_expansion_cache


def expand_query_semantically(question: str, max_expansions: int = 3) -> List[str]:
    """Expand query with semantically similar variations for better retrieval (optimized)."""
    # Check cache first
    cached = query_expansion_cache.get(question, max_expansions)
    if cached is not None:
        return cached
    
    expanded = [question]  # Original question
    
    question_lower = question.lower()
    
    # Limit expansions for performance
    variations = []
    
    # Expand "사용되지 않는" related queries (limit to 3 most relevant)
    if any(term in question_lower for term in ['사용되지', '사용하지', '미사용', '사용 안']):
        variations.extend([
            question.replace('사용되지 않는', '실제 사용하지 않는'),
            question + ' 비고 필드에 실제 사용하지 않음이 있는',
            question.replace('사용되지 않는', '미사용인'),
        ])
    
    # Expand "속하지 않는" related queries (limit to 2 most relevant)
    elif any(term in question_lower for term in ['속하지 않는', '포함되지 않는', '해당하지 않는']):
        variations.extend([
            question.replace('속하지 않는', '포함되지 않는'),
            question + ' 테이블구분이 BMS도 GPS도 아닌',
        ])
    
    # Expand queries about BMS and GPS together (limit to 2)
    elif 'bms' in question_lower and 'gps' in question_lower:
        variations.extend([
            question + ' 테이블구분 필드를 확인하여',
            '테이블구분이 BMS도 아니고 GPS도 아닌 변수는',
        ])
    
    # Limit total expansions
    expanded.extend([v for v in variations if v != question][:max_expansions])
    
    # Cache the result
    query_expansion_cache.set(question, max_expansions, expanded)
    
    return expanded


def expand_query_with_embeddings(
    question: str,
    embedding_model,
    passages: List[str],
    top_k: int = 5,
    max_expansions: int = 2
) -> List[str]:
    """
    Expand query using embedding-based similarity to find similar passages.
    This generates query variations based on semantically similar content.
    
    Args:
        question: Original query
        embedding_model: SentenceTransformer model for embeddings
        passages: List of passages to find similar content from
        top_k: Number of similar passages to consider
        max_expansions: Maximum number of query expansions to return
    
    Returns:
        List of expanded queries (including original)
    """
    if not passages or embedding_model is None:
        return [question]
    
    try:
        # Get query embedding
        query_embedding = embedding_model.encode(question, convert_to_numpy=True, show_progress_bar=False)
        
        # Get passage embeddings (batch for efficiency)
        passage_embeddings = embedding_model.encode(
            passages[:min(100, len(passages))],  # Limit to first 100 for performance
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        
        # Normalize embeddings
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        passage_embeddings = passage_embeddings / np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(passage_embeddings, query_embedding)
        
        # Get top-k similar passages
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Extract key phrases from similar passages (simple approach: first sentence or key terms)
        expanded_queries = [question]
        for idx in top_indices:
            if similarities[idx] > 0.5:  # Only use passages with reasonable similarity
                passage = passages[idx]
                # Extract first sentence or key phrase (simple heuristic)
                first_sentence = passage.split('。')[0].split('.')[0].strip()
                if first_sentence and len(first_sentence) > 10 and first_sentence not in expanded_queries:
                    # Create variation: combine original query with key phrase
                    if len(expanded_queries) < max_expansions + 1:
                        expanded_queries.append(f"{question} {first_sentence[:50]}")
        
        return expanded_queries[:max_expansions + 1]
    except Exception:
        # Fallback to original query if expansion fails
        return [question]

