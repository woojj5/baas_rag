"""Token counting utilities."""
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Fallback: simple character-based estimation
def _estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token average)."""
    return len(text) // 4


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text.
    
    Args:
        text: Input text
        model: Model name for tiktoken (default: gpt-3.5-turbo)
        
    Returns:
        Estimated token count
    """
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            return _estimate_tokens(text)
    else:
        return _estimate_tokens(text)


def estimate_prompt_tokens(query: str, retrieved_chunks: list, system_prompt_base: str = "") -> int:
    """
    Estimate prompt token count before building full prompt.
    
    Args:
        query: User query
        retrieved_chunks: List of RetrievedChunk objects
        system_prompt_base: Base system prompt (optional)
        
    Returns:
        Estimated token count
    """
    # Estimate query tokens
    query_tokens = count_tokens(query)
    
    # Estimate chunk tokens (rough: average chunk size * number of chunks)
    # Assume expanded chunks are ~200 tokens, compressed chunks are ~50 tokens
    chunk_tokens = 0
    if retrieved_chunks:
        avg_chunk_size = sum(len(rc.chunk.text) for rc in retrieved_chunks) / len(retrieved_chunks)
        # Rough estimate: 1 token per 4 chars
        avg_tokens_per_chunk = avg_chunk_size // 4
        # Assume 50% expanded, 50% compressed (conservative estimate)
        chunk_tokens = int(len(retrieved_chunks) * avg_tokens_per_chunk * 0.6)  # 60% of full size
    
    # Estimate system prompt tokens
    system_tokens = count_tokens(system_prompt_base) if system_prompt_base else 200
    
    # Add overhead for formatting, headers, etc.
    overhead = 100
    
    return query_tokens + chunk_tokens + system_tokens + overhead

