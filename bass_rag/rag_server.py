#!/usr/bin/env python3
"""
RAG Server - FastAPI backend for RAG queries using Ollama
"""

import json
import os
from pathlib import Path
from typing import List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

ROOT = Path("/home/keti_spark1/j309")
INDEX_DIR = ROOT / "index"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:27b"

# Global variables
faiss_index: Optional[faiss.Index] = None
passages: List[dict] = []
embedding_model: Optional[SentenceTransformer] = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]


app = FastAPI(
    title="RAG Server",
    description="RAG server using Ollama (gemma3:27b) and FAISS",
    version="1.0.0"
)


def load_index():
    """Load FAISS index, passages, and embedding model."""
    global faiss_index, passages, embedding_model
    
    # Check if index exists
    faiss_path = INDEX_DIR / "faiss.index"
    passages_path = INDEX_DIR / "passages.json"
    model_name_path = INDEX_DIR / "model_name.txt"
    
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}. Run build_index.py first.")
    
    if not passages_path.exists():
        raise FileNotFoundError(f"Passages file not found: {passages_path}. Run build_index.py first.")
    
    if not model_name_path.exists():
        raise FileNotFoundError(f"Model name file not found: {model_name_path}. Run build_index.py first.")
    
    # Load model name
    with open(model_name_path, "r", encoding="utf-8") as f:
        model_name = f.read().strip()
    
    print(f"[*] Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    
    # Load FAISS index
    print(f"[*] Loading FAISS index...")
    faiss_index = faiss.read_index(str(faiss_path))
    print(f"[+] FAISS index loaded: {faiss_index.ntotal} vectors")
    
    # Load passages
    print(f"[*] Loading passages...")
    with open(passages_path, "r", encoding="utf-8") as f:
        passages = json.load(f)
    print(f"[+] Loaded {len(passages)} passages")
    
    print(f"[+] Index loading complete!")


@app.on_event("startup")
async def startup_event():
    """Load index on server startup."""
    try:
        load_index()
    except Exception as e:
        print(f"[ERROR] Failed to load index: {e}")
        raise


def search_top_k(query: str, top_k: int) -> List[str]:
    """Search top-k similar chunks."""
    if faiss_index is None or embedding_model is None:
        raise RuntimeError("Index not loaded. Please restart the server.")
    
    # Encode query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    # Search
    k = min(top_k, faiss_index.ntotal)
    distances, indices = faiss_index.search(query_embedding, k)
    
    # Get top-k passages
    results = []
    for idx in indices[0]:
        if idx < len(passages):
            results.append(passages[idx]["text"])
    
    return results


def build_rag_prompt(question: str, contexts: List[str]) -> str:
    """Build Korean RAG prompt."""
    context_text = "\n\n".join([f"- {ctx}" for ctx in contexts])
    
    prompt = f"""너는 EV Battery·BMS·BAAS 전문 어시스턴트다.

아래 문서를 참고하여 정확하고 자세하게 한국어로 답변해라.

문서에 없는 정보는 지어내지 말고 
'문서에서 해당 내용을 찾을 수 없습니다' 라고 말해라.



[참고 문서]

{context_text}



[질문]

{question}



[답변]

"""
    return prompt


def query_ollama(prompt: str) -> str:
    """Send prompt to Ollama API and get response."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        answer = result.get("response", "")
        
        return answer.strip()
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama API error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying Ollama: {str(e)}"
        )


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "RAG Server is running",
        "model": OLLAMA_MODEL,
        "index_size": faiss_index.ntotal if faiss_index else 0,
        "passages_count": len(passages)
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    RAG query endpoint.
    
    - question: User question
    - top_k: Number of top chunks to retrieve (default: 3)
    """
    try:
        # Validate top_k
        top_k = max(1, min(request.top_k, 10))  # Limit to 1-10
        
        # Search top-k chunks
        contexts = search_top_k(request.question, top_k)
        
        if not contexts:
            return QueryResponse(
                answer="문서에서 해당 내용을 찾을 수 없습니다.",
                contexts=[]
            )
        
        # Build RAG prompt
        prompt = build_rag_prompt(request.question, contexts)
        
        # Query Ollama
        answer = query_ollama(prompt)
        
        return QueryResponse(
            answer=answer,
            contexts=contexts
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/health")
def health():
    """Health check with index status."""
    return {
        "status": "healthy",
        "index_loaded": faiss_index is not None,
        "model_loaded": embedding_model is not None,
        "index_size": faiss_index.ntotal if faiss_index else 0,
        "passages_count": len(passages)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

