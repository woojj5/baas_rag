# REFRAG RAG 시스템 구현

## 개요

REFRAG (Rethinking RAG based Decoding) 논문의 **Compress - Sense - Expand** 구조를 구현한 RAG 시스템입니다.

## 아키텍처

### 핵심 구조

```
질문 입력
  ↓
[1] Retrieve: 벡터 검색으로 관련 청크 검색
  ↓
[2] Sense: 압축/확장 정책으로 각 청크 결정
  ↓
[3] Compress: 관련성 낮은 청크 압축 (요약)
  ↓
[4] Expand: 관련성 높은 청크 확장 (전체 내용)
  ↓
[5] Prompt: REFRAG 스타일 프롬프트 생성
  ↓
[6] Generate: Ollama LLM으로 답변 생성
  ↓
답변 출력
```

## 디렉터리 구조

```
app/
├── models/
│   └── schemas.py          # Pydantic 모델 (Document, Chunk, RetrievedChunk)
├── llm/
│   ├── base.py            # LLMClient 추상 클래스
│   └── ollama_client.py   # Ollama chat API 구현
├── embeddings/
│   ├── base.py            # EmbeddingClient 인터페이스
│   └── st_client.py       # SentenceTransformers 구현
├── index/
│   └── vector_index.py    # FAISS 기반 VectorIndex
├── compression/
│   ├── policy.py          # 압축/확장 정책 (Heuristic, RL skeleton)
│   └── compressor.py      # 청크 압축/확장 로직
├── rag/
│   ├── adapter.py         # 기존 인덱스 → REFRAG 형식 변환
│   ├── pipeline.py        # REFRAG 파이프라인
│   └── prompt_builder.py  # REFRAG 프롬프트 생성
├── utils/
│   ├── tokenizer.py       # 토큰 수 계산
│   └── latency.py         # LLM 호출 시간 측정
└── refrag_server.py       # FastAPI 서버
```

## 사용 방법

### 1. 서버 실행

```bash
# 기존 인덱스가 있어야 함 (build_index.py 실행 필요)
uvicorn app.refrag_server:app --host 0.0.0.0 --port 8011
```

### 2. API 호출

```bash
# Health check
curl http://localhost:8011/health

# Query
curl -X POST http://localhost:8011/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "BMS 테이블에 존재는 하지만 실제로는 사용되지 않는 변수는?",
    "top_k": 8
  }'
```

### 3. Python 테스트

```bash
python test_refrag.py
```

## 주요 기능

### 1. Compress - Sense - Expand

- **Sense**: Heuristic 정책으로 상위 N개 청크는 EXPAND, 나머지는 COMPRESS
- **Compress**: 관련성 낮은 청크를 첫 N문장으로 요약
- **Expand**: 관련성 높은 청크는 전체 내용 사용

### 2. Ollama Chat API

- `/api/chat` 엔드포인트 사용
- System prompt + User messages 구조
- Gemma3 27B 모델 사용

### 3. 기존 시스템 통합

- 기존 FAISS 인덱스와 호환
- 기존 인덱스를 REFRAG 형식으로 자동 변환
- 기존 `/query` 엔드포인트는 유지 (포트 8010)

## 설정

`.env` 파일 또는 환경 변수:

```bash
# REFRAG 설정
CHUNK_TOKEN_SIZE=256
MAX_RETRIEVED_CHUNKS=32
MAX_EXPANDED_CHUNKS=5

# Ollama 설정
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:27b

# Embedding 설정
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

## 응답 형식

```json
{
  "answer": "답변 텍스트",
  "used_chunks": [
    {
      "id": "chunk_0",
      "document_id": "doc_0",
      "text": "청크 텍스트",
      "token_count": 150,
      "start_offset": 0,
      "end_offset": 500,
      "metadata": {...}
    }
  ],
  "compression_decisions": {
    "chunk_0": "EXPAND",
    "chunk_1": "COMPRESS"
  },
  "prompt_token_count": 1200,
  "llm_latency_ms": 3500.5
}
```

## 기존 시스템과의 차이점

| 기능 | 기존 RAG (포트 8010) | REFRAG (포트 8011) |
|------|---------------------|-------------------|
| 프롬프트 | 모든 청크 전체 내용 | 상위 N개 확장, 나머지 압축 |
| LLM API | `/api/generate` | `/api/chat` |
| 청크 구조 | 문자열 리스트 | 구조화된 Chunk 객체 |
| 메타데이터 | 문자열에 포함 | 별도 필드 |
| 압축/확장 | 없음 | Heuristic 정책 |

## 향후 개선 사항

1. **RL 기반 정책**: `RLCompressionPolicy` 구현
2. **LLM 기반 압축**: Compressor에서 LLM 요약 모드 추가
3. **Ollama Embedding**: `/api/embeddings` 엔드포인트 지원
4. **오프라인 인덱스 구축**: REFRAG 형식으로 직접 인덱스 구축

## 참고

- 기존 RAG 서버 (`app/rag_server.py`)는 포트 8010에서 계속 동작
- REFRAG 서버는 포트 8011에서 별도로 실행
- 두 서버는 같은 인덱스를 공유하지만 다른 방식으로 처리

