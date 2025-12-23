"""REFRAG-style prompt builder."""
from typing import List, Dict
from app.models.schemas import RetrievedChunk, Chunk
from app.compression.compressor import compress_chunk, expand_chunk
from app.compression.policy import CompressionPolicy


def build_refrag_prompt(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    compression_decisions: Dict[str, str]
) -> tuple[str, List[Dict[str, str]]]:
    """
    Build REFRAG-style prompt with compressed/expanded chunks.
    
    Args:
        query: User query
        retrieved_chunks: List of retrieved chunks
        compression_decisions: Dict mapping chunk_id to "COMPRESS" or "EXPAND"
        
    Returns:
        Tuple of (system_prompt, messages)
    """
    # System prompt - 최적화: 더 간결하게 (토큰 수 감소)
    system_prompt = """답변: 질문에 직접 답변만, 서론/사족 금지. 변수 목록은 변수명만 나열.
[HIGH]: 전체 내용 (우선 참고)
[LOW]: 요약 (필요시 참고)
없으면 "문서에서 해당 내용을 찾을 수 없습니다"."""

    # Separate expanded and compressed chunks
    expanded_chunks = []
    compressed_chunks = []
    
    for retrieved_chunk in retrieved_chunks:
        chunk = retrieved_chunk.chunk
        chunk_id = chunk.id
        decision = compression_decisions.get(chunk_id, "COMPRESS")
        
        if decision == "EXPAND":
            expanded_text = expand_chunk(chunk, retrieved_chunk.score)
            expanded_chunks.append(expanded_text)
        else:
            compressed_text = compress_chunk(chunk, mode="head")
            compressed_chunks.append(compressed_text)
    
    # Build context block
    context_parts = []
    
    if expanded_chunks:
        context_parts.append("[HIGH]\n")  # 간소화된 헤더
        context_parts.extend(expanded_chunks)
        context_parts.append("")
    
    if compressed_chunks:
        context_parts.append("[LOW]\n")  # 간소화된 헤더
        context_parts.extend(compressed_chunks)
    
    context_text = "\n".join(context_parts)
    
    # Build user message - 답변 프롬프트 (간소화)
    user_message = f"""{context_text}

[질문]
{query}

[답변]
서론 없이 직접 답변."""
    
    messages = [
        {"role": "user", "content": user_message}
    ]
    
    return system_prompt, messages

