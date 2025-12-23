"""REFRAG RAG pipeline."""
from typing import List

from app.models.schemas import QueryResponse, RetrievedChunk
from app.llm.base import LLMClient
from app.embeddings.base import EmbeddingClient
from app.index.vector_index import VectorIndex
from app.compression.policy import CompressionPolicy, HeuristicCompressionPolicy
from app.rag.prompt_builder import build_refrag_prompt
from app.utils.tokenizer import count_tokens
from app.utils.latency import measure_llm_latency
from app.config import Config


async def rag_answer(
    query: str,
    top_k: int,
    llm: LLMClient,
    embedder: EmbeddingClient,
    index: VectorIndex,
    compression_policy: CompressionPolicy = None
) -> QueryResponse:
    """
    REFRAG-style RAG pipeline: Compress - Sense - Expand.
    
    Args:
        query: User query
        top_k: Number of chunks to retrieve
        llm: LLM client
        embedder: Embedding client
        index: Vector index
        compression_policy: Compression policy (default: HeuristicCompressionPolicy)
        
    Returns:
        QueryResponse with answer and metadata
    """
    if compression_policy is None:
        compression_policy = HeuristicCompressionPolicy(
            max_expanded_chunks=Config.MAX_EXPANDED_CHUNKS if hasattr(Config, 'MAX_EXPANDED_CHUNKS') else 5
        )
    
    # Step 1: Retrieve
    # 기본 검색 (임베딩 기반)
    query_embedding = embedder.embed_query(query)
    search_results = index.search(query_embedding, top_k=top_k)
    
    # Convert to RetrievedChunk objects
    retrieved_chunks = [
        RetrievedChunk(chunk=chunk, score=score)
        for chunk, score in search_results
    ]
    
    if not retrieved_chunks:
        return QueryResponse(
            answer="문서에서 해당 내용을 찾을 수 없습니다.",
            used_chunks=[],
            compression_decisions={},
            prompt_token_count=0,
            llm_latency_ms=0.0
        )
    
    # Step 2: Sense - Decide compression/expansion
    compression_decisions = compression_policy.decide(retrieved_chunks)
    
    # Step 3: Compress/Expand - Build prompt
    system_prompt, messages = build_refrag_prompt(
        query,
        retrieved_chunks,
        compression_decisions
    )
    
    # Step 4: Generate answer
    full_prompt_text = system_prompt + "\n\n" + messages[0]["content"]
    prompt_token_count = count_tokens(full_prompt_text)
    
    answer_coro = llm.chat(
        system_prompt=system_prompt,
        messages=messages,
        max_tokens=512,
        temperature=0.2
    )
    
    answer, llm_latency_ms = await measure_llm_latency(answer_coro)
    
    # Extract used chunks
    used_chunks = [rc.chunk for rc in retrieved_chunks]
    
    return QueryResponse(
        answer=answer,
        used_chunks=used_chunks,
        compression_decisions=compression_decisions,
        prompt_token_count=prompt_token_count,
        llm_latency_ms=llm_latency_ms
    )

