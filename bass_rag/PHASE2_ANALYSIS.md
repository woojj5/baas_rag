# Phase 2 작업 분석 및 계획

## 📊 현재 상황 분석

### 2.1 전역 변수 동시성 문제

#### 전역 변수 목록
**refrag_server.py:**
- `faiss_index: faiss.Index | None = None`
- `passages: List[str] = []`
- `passages_lower: List[str] = []`
- `embedding_model: SentenceTransformer | None = None`
- `hybrid_retriever: HybridRetriever | None = None`
- `reranker: DomainReranker | None = None`
- `domain_dict = None`

**rag_server.py:**
- `faiss_index: faiss.Index | None = None`
- `passages: List[str] = []`
- `embedding_model: SentenceTransformer | None = None`
- `hybrid_retriever: HybridRetriever | None = None`
- `reranker: DomainReranker | None = None`
- `ollama_chat_client: OllamaLLMClient | None = None`
- `compression_policy: HeuristicCompressionPolicy | None = None`
- `doc_id_to_original: Dict[str, str] = {}`

#### 동시 접근 시나리오
1. **읽기-읽기**: 여러 `query` 요청이 동시에 `faiss_index`, `passages` 읽기 → 안전 (FAISS는 읽기 전용)
2. **읽기-쓰기**: `query`가 읽는 동안 `ingest`가 `faiss_index`, `passages` 수정 → **위험!**
3. **쓰기-쓰기**: 여러 `ingest` 요청이 동시에 인덱스 수정 → **위험!**

#### 문제점
- `ingest` 엔드포인트에서 `faiss_index.add()` 및 `passages.append()` 수행
- `query` 엔드포인트에서 `faiss_index.search()` 및 `passages[idx]` 접근
- Lock 없이 동시 접근 시 인덱스 불일치, 메모리 오류 가능

### 2.2 서버 파일 중복 문제

#### 중복된 기능 비교

| 기능 | rag_server.py | refrag_server.py | 중복 여부 |
|------|---------------|------------------|----------|
| `postprocess_answer` | ✅ | ✅ | 중복 |
| `find_not_used_passages` | ✅ | ❌ | 부분 중복 |
| 하이브리드 검색 | ✅ | ✅ | 중복 |
| Domain reranker | ✅ | ✅ | 중복 |
| 프롬프트 빌더 | ❌ | ✅ (REFRAG 전용) | 다름 |
| `ingest` 엔드포인트 | ✅ | ❌ | 다름 |
| `upload` 엔드포인트 | ✅ | ❌ | 다름 |
| SQL 생성 | ✅ | ❌ | 다름 |
| 전처리 | ✅ | ❌ | 다름 |

#### 공통 로직 후보
1. `postprocess_answer` - 완전 동일
2. `expand_query_semantically` - 확인 필요
3. 하이브리드 검색 로직 - 확인 필요
4. Domain dictionary 사용 - 확인 필요

---

## 🎯 Phase 2 작업 계획

### 2.1 전역 변수 동시성 문제 해결

#### 전략: Read-Write Lock 패턴
- **읽기 Lock**: `query` 엔드포인트에서 사용 (여러 요청 동시 허용)
- **쓰기 Lock**: `ingest` 엔드포인트에서 사용 (단독 접근)

#### 구현 계획
1. `asyncio.Lock` 도입 (FastAPI는 async이므로)
2. 읽기-쓰기 Lock 클래스 구현 또는 `asyncio`의 기본 Lock 사용
3. `query` 엔드포인트: 읽기 Lock
4. `ingest` 엔드포인트: 쓰기 Lock
5. 인덱스 업데이트 시 Copy-on-Write 패턴 고려

#### 파일 수정 위치
- `app/refrag_server.py`: 전역 변수 선언 부분, `query` 함수, `ingest` 함수 (없으면 추가)
- `app/rag_server.py`: 전역 변수 선언 부분, `query` 함수, `ingest` 함수

### 2.2 서버 파일 중복 해결

#### 전략: 공통 로직 추출 (옵션 C)
- 두 서버 모두 유지 (기능 차이가 있음)
- 공통 로직만 추출하여 재사용

#### 추출 대상
1. `postprocess_answer` → `app/utils/postprocess.py`
2. `find_not_used_passages` → `app/utils/passage_filter.py` (확장)
3. 하이브리드 검색 로직 → 이미 `app/hybrid_retrieval.py`에 있음 (확인 필요)

#### 작업 순서
1. 공통 로직 추출
2. 두 서버에서 import하여 사용
3. 테스트 및 검증

---

## 📝 다음 단계

1. **동시성 문제 해결부터 시작** (더 위험한 문제)
2. **공통 로직 추출은 이후 진행** (코드 품질 개선)

