# REFRAG Server 개선 계획서

**작성일**: 2024년 12월  
**대상 파일**: `app/refrag_server.py`  
**목적**: IMPROVEMENT_PLAN.md의 계획을 refrag_server.py에 적용

---

## 📋 현재 상태 분석

### ✅ 이미 완료된 항목
1. **Phase 1: 테이블 필터링 버그 수정** ✅
   - `postprocess_answer` 함수에 테이블별 변수 필터링 로직 추가됨
   - `find_not_used_passages` 함수에 GPS 테이블 지원 추가됨
   - 프롬프트에 GPS/BMS 테이블별 규칙 추가됨

2. **Phase 2.1: 전역 변수 동시성 문제 해결** ✅
   - `asyncio.Lock` (index_lock) 도입됨
   - `query` 엔드포인트에 읽기 lock 적용됨
   - `ingest`, `upload`, `delete_passages`, `delete_all_passages` 엔드포인트에 쓰기 lock 적용됨
   - `passages_snapshot` 패턴으로 Copy-on-Write 구현됨

3. **Phase 2.2: 공통 로직 추출 (부분 완료)** ✅
   - `postprocess_answer` → `app/utils/postprocess.py`로 추출됨 ✅
   - `expand_query_semantically`는 아직 중복 (refrag_server.py와 rag_server.py에 모두 존재) ⚠️

### ⚠️ 미완료 항목

#### Phase 2.2: 서버 파일 중복 해결 (부분 미완료)
- **문제**: `expand_query_semantically` 함수가 `refrag_server.py`와 `rag_server.py`에 중복 존재
- **위치**: 
  - `app/refrag_server.py:919-962`
  - `app/rag_server.py:849-883`
- **작업 필요**: `app/utils/query_expansion.py`로 추출

#### Phase 3.1: 에러 처리 및 로깅 전략 수립 (미완료)
- **현재 상태**:
  - `print()` 문만 사용 (구조화된 로깅 없음)
  - 기본적인 `try-except`만 있음
  - 예외 타입별 처리 없음
  - 민감 정보 필터링 없음
  - 에러 로깅 없음
- **작업 필요**:
  - 구조화된 로깅 도입 (`app/utils/logger.py` 생성 또는 기존 파일 사용)
  - 모든 엔드포인트에 try-except 추가
  - 예외 타입별 처리 (ValueError, HTTPException 등)
  - 민감 정보 필터링 함수 추가 (`app/utils/security.py`의 `sanitize_error_message` 사용)
  - 에러 로깅 추가

#### Phase 3.2: 보안 이슈 해결 (미완료)
- **현재 상태**:
  - 하드코딩된 경로: `Config.ROOT`, `Config.DATA_DIR` 사용 (환경 변수로 관리되어야 함)
  - 파일 업로드 보안:
    - 파일 크기 제한 없음
    - 파일 타입 검증 부족 (확장자만 체크)
    - 경로 traversal 방지 없음
  - 환경 변수 검증 없음
- **작업 필요**:
  - `app/utils/security.py` 생성 또는 기존 파일 사용
  - 파일 업로드 엔드포인트에 보안 검증 추가:
    - 파일 크기 제한 (`Config.MAX_UPLOAD_SIZE` 사용)
    - 파일 타입 검증 (`Config.ALLOWED_FILE_EXTENSIONS` 사용)
    - 경로 traversal 방지 (`sanitize_filename`, `validate_save_path` 사용)
  - `app/config.py`에 환경 변수 검증 추가 (`validate_config()` 메서드)
  - 서버 시작 시 환경 변수 검증 실행

#### Phase 3.3: REFRAG 파이프라인 완성도 개선 (미완료)
- **현재 상태**:
  - 재검색 단계 없음 (LLM 응답 기반 재검색 없음)
  - Context Refinement가 단순 휴리스틱에 의존
  - Query Rewriting이 제한적 (`expand_query_semantically`만 사용)
- **작업 필요**:
  - LLM 응답 기반 재검색 로직 구현
  - Context Refinement 개선 (여러 번의 샘플링/조합)
  - Query Rewriting 개선 (LLM 기반 의미 확장)

#### Phase 4.1: 성능 최적화 (미완료)
- **현재 상태**:
  - `convert_existing_index_to_refrag`에서 재임베딩 수행 (서버 시작 시)
  - `ingest` 시 매번 전체 인덱스 저장
  - 캐싱 전략 부족
- **작업 필요**:
  - 불필요한 재임베딩 제거
  - 증분 인덱스 업데이트 구현
  - 캐싱 전략 개선

---

## 🎯 적용 계획

### Phase 2.2 완료: 공통 로직 추출 (우선순위: P1)

#### 작업 1: `expand_query_semantically` 함수 추출
**파일 수정**:
- `app/utils/query_expansion.py` (신규 생성)
  - `expand_query_semantically` 함수 이동
  - `detect_query_types` 함수도 함께 이동 고려

- `app/refrag_server.py`
  - `expand_query_semantically` 함수 제거 (라인 919-962)
  - `from app.utils.query_expansion import expand_query_semantically` 추가

**예상 시간**: 1-2시간  
**검증 방법**: 
- `refrag_server.py`와 `rag_server.py` 모두에서 import 테스트
- 기존 기능 동작 확인

---

### Phase 3.1: 에러 처리 및 로깅 전략 수립 (우선순위: P2)

#### 작업 1: 구조화된 로깅 도입
**파일 생성/수정**:
- `app/utils/logger.py` (신규 생성 또는 기존 파일 확인)
  - 로그 레벨별 분리 (DEBUG, INFO, WARNING, ERROR)
  - 파일 로깅 및 로테이션 설정 (10MB, 5개 백업)
  - 콘솔 출력 (INFO 이상)

- `app/refrag_server.py`
  - `from app.utils.logger import get_logger` 추가
  - `logger = get_logger(__name__)` 추가
  - 모든 `print()` 문을 `logger.info()`, `logger.error()` 등으로 교체
  - 서버 시작 시 로깅 추가

**예상 시간**: 2-3시간

#### 작업 2: 예외 처리 개선
**파일 수정**:
- `app/refrag_server.py`
  - 모든 엔드포인트에 try-except 추가
  - 예외 타입별 처리:
    - `ValueError` → HTTP 400
    - `FileNotFoundError` → HTTP 404
    - `HTTPException` → 그대로 전달
    - 기타 `Exception` → HTTP 500 (민감 정보 필터링 후)
  - `from app.utils.security import sanitize_error_message` 추가
  - 모든 예외 메시지에 `sanitize_error_message()` 적용
  - 에러 로깅 추가 (`logger.error()`)

**영향받는 엔드포인트**:
- `/query` (라인 1082-1444)
- `/ingest` (라인 1447-1474)
- `/upload` (라인 1517-1639)
- `/passages` (라인 1642-1650)
- `/passages/delete` (라인 1653-1685)
- `/passages/delete-all` (라인 1688-1710)
- `/sql/generate` (라인 1713-1799)
- `/sql/preview` (라인 1802-1807)
- `/sql/generate-from-db` (라인 1810-1880)
- `/preprocess` (라인 1883-1897)

**예상 시간**: 3-4시간

---

### Phase 3.2: 보안 이슈 해결 (우선순위: P2)

#### 작업 1: 파일 업로드 보안 강화
**파일 생성/수정**:
- `app/utils/security.py` (신규 생성 또는 기존 파일 확인)
  - `sanitize_filename(filename: str) -> str`: 경로 traversal 방지
  - `validate_save_path(base_dir: Path, filename: str) -> Tuple[Path, bool]`: 저장 경로 검증
  - `sanitize_error_message(e: Exception) -> str`: 민감 정보 필터링

- `app/config.py`
  - `MAX_UPLOAD_SIZE` 추가 (기본값: 50MB)
  - `ALLOWED_FILE_EXTENSIONS` 추가 (기본값: ['.txt', '.md', '.csv'])

- `app/refrag_server.py`
  - `/upload` 엔드포인트에 보안 검증 추가:
    - 파일 크기 제한 체크 (`len(content) > Config.MAX_UPLOAD_SIZE`)
    - 파일 타입 검증 (`filename.endswith(tuple(Config.ALLOWED_FILE_EXTENSIONS))`)
    - 경로 traversal 방지 (`sanitize_filename()`, `validate_save_path()` 사용)
  - 모든 파일 저장 경로에 `validate_save_path()` 적용

**예상 시간**: 2-3시간

#### 작업 2: 환경 변수 검증
**파일 수정**:
- `app/config.py`
  - `validate_config()` 정적 메서드 추가:
    - `OLLAMA_BASE_URL` 형식 검증 (URL 형식, 포트 범위)
    - 경로 존재 여부 확인 (`ROOT`, `DATA_DIR`, `INDEX_DIR`)
  - 서버 시작 시 `validate_config()` 호출

- `app/refrag_server.py`
  - `startup()` 함수에서 `Config.validate_config()` 호출 추가

**예상 시간**: 1-2시간

#### 작업 3: `.env.example` 파일 생성
**파일 생성**:
- `.env.example`
  - 모든 환경 변수 문서화
  - `RAG_ROOT_DIR`, `RAG_DATA_DIR`, `RAG_INDEX_DIR` 추가
  - `MAX_UPLOAD_SIZE`, `ALLOWED_FILE_EXTENSIONS` 추가

**예상 시간**: 30분

---

### Phase 3.3: REFRAG 파이프라인 완성도 개선 (우선순위: P2)

#### 작업 1: 재검색 단계 추가
**파일 수정**:
- `app/refrag_server.py`
  - `/query` 엔드포인트에 재검색 로직 추가:
    - LLM 응답 후, 응답에서 키워드 추출
    - 추출된 키워드로 재검색 수행
    - 재검색 결과를 기존 결과와 병합/정렬
  - `app/rag/pipeline.py` 확장 고려

**예상 시간**: 3-5일

#### 작업 2: Context Refinement 개선
**파일 수정**:
- `app/compression/policy.py`
  - 여러 번의 샘플링/조합 로직 추가
  - 중요도 기반 정렬/압축 전략 개선

**예상 시간**: 2-3일

#### 작업 3: Query Rewriting 개선
**파일 수정**:
- `app/utils/query_expansion.py` (작업 2.2에서 생성)
  - LLM 기반 의미 확장 함수 추가
  - Embedding 기반 유사 쿼리 생성 함수 추가

**예상 시간**: 2-3일

---

### Phase 4.1: 성능 최적화 (우선순위: P3)

#### 작업 1: 불필요한 재임베딩 제거
**파일 수정**:
- `app/rag/adapter.py`
  - `convert_existing_index_to_refrag` 함수 개선:
    - 기존 FAISS 인덱스에서 임베딩 추출 방법 검토
    - 재임베딩 대신 기존 임베딩 재사용

**예상 시간**: 1-2일

#### 작업 2: 인덱스 업데이트 최적화
**파일 수정**:
- `app/refrag_server.py`
  - `/ingest` 엔드포인트:
    - 매번 전체 인덱스 저장 대신 증분 업데이트 구현
    - 배치 업데이트 고려 (여러 ingest 요청을 모아서 처리)

**예상 시간**: 1-2일

#### 작업 3: 캐싱 전략 개선
**파일 수정**:
- `app/refrag_server.py`
  - 쿼리 결과 캐싱 추가 (`@lru_cache` 또는 Redis)
  - 임베딩 캐싱 추가

**예상 시간**: 1-2일

---

## 📊 작업 일정 요약

| Phase | 작업 | 우선순위 | 예상 시간 | 시작 주차 |
|-------|------|---------|----------|----------|
| 2.2 | expand_query_semantically 추출 | P1 | 1-2시간 | 즉시 |
| 3.1 | 구조화된 로깅 도입 | P2 | 2-3시간 | 1주차 |
| 3.1 | 예외 처리 개선 | P2 | 3-4시간 | 1주차 |
| 3.2 | 파일 업로드 보안 강화 | P2 | 2-3시간 | 1주차 |
| 3.2 | 환경 변수 검증 | P2 | 1-2시간 | 1주차 |
| 3.2 | .env.example 생성 | P2 | 30분 | 1주차 |
| 3.3 | 재검색 단계 추가 | P2 | 3-5일 | 2주차 |
| 3.3 | Context Refinement 개선 | P2 | 2-3일 | 2주차 |
| 3.3 | Query Rewriting 개선 | P2 | 2-3일 | 2주차 |
| 4.1 | 불필요한 재임베딩 제거 | P3 | 1-2일 | 3주차 |
| 4.1 | 인덱스 업데이트 최적화 | P3 | 1-2일 | 3주차 |
| 4.1 | 캐싱 전략 개선 | P3 | 1-2일 | 3주차 |

**총 예상 시간**: 
- Phase 2.2 완료: 1-2시간
- Phase 3 (에러 처리, 로깅, 보안): 1-2일
- Phase 3.3 (REFRAG 파이프라인): 7-11일
- Phase 4 (성능 최적화): 3-6일
- **전체**: 약 2-3주

---

## ✅ 체크리스트

### Phase 2.2 완료 기준
- [ ] `expand_query_semantically` 함수를 `app/utils/query_expansion.py`로 추출
- [ ] `refrag_server.py`에서 중복 함수 제거
- [ ] `rag_server.py`에서도 동일하게 import로 변경
- [ ] 기존 기능 동작 확인

### Phase 3.1 완료 기준
- [ ] 구조화된 로깅 설정 완료
- [ ] 모든 `print()` 문을 `logger`로 교체
- [ ] 모든 엔드포인트에 try-except 추가
- [ ] 예외 타입별 처리 구현
- [ ] 민감 정보 필터링 적용
- [ ] 에러 로깅 추가

### Phase 3.2 완료 기준
- [ ] 파일 업로드 보안 검증 추가 (크기, 타입, 경로 traversal)
- [ ] 환경 변수 검증 추가
- [ ] `.env.example` 파일 생성
- [ ] 보안 스캔 통과

### Phase 3.3 완료 기준
- [ ] 재검색 단계 추가
- [ ] Context Refinement 개선
- [ ] Query Rewriting 개선
- [ ] 답변 품질 비교 (Before/After)

### Phase 4.1 완료 기준
- [ ] 불필요한 재임베딩 제거
- [ ] 증분 인덱스 업데이트 구현
- [ ] 캐싱 전략 적용
- [ ] 성능 벤치마크 개선 확인

---

## 🚨 리스크 및 대응 방안

### 리스크 1: 기존 기능 동작 중단
- **대응**: 각 작업마다 철저한 테스트
- **대응**: 단계적 배포 (작업별로 독립적으로 완료 가능하도록 설계)

### 리스크 2: 성능 저하
- **대응**: 로깅 추가 시 성능 영향 최소화 (비동기 로깅 고려)
- **대응**: 보안 검증 추가 시 성능 테스트

### 리스크 3: 코드 복잡도 증가
- **대응**: 공통 로직 추출로 중복 제거
- **대응**: 유틸리티 모듈로 분리하여 가독성 향상

---

## 📝 참고 사항

### 파일 구조
```
app/
├── refrag_server.py          # 메인 서버 파일
├── utils/
│   ├── postprocess.py        # ✅ 이미 생성됨
│   ├── query_expansion.py    # ⚠️ 생성 필요
│   ├── logger.py             # ⚠️ 생성 필요
│   └── security.py           # ⚠️ 생성 필요
├── config.py                 # ⚠️ 수정 필요 (환경 변수 검증 추가)
└── ...
```

### 의존성
- Phase 2.2는 독립적으로 진행 가능
- Phase 3.1과 3.2는 함께 진행 권장 (보안과 로깅은 밀접한 관련)
- Phase 3.3는 Phase 2.2 완료 후 진행 권장
- Phase 4.1는 Phase 3 완료 후 진행 권장

---

## 🎯 최종 목표

1. ✅ Critical 버그 제로 (완료)
2. ✅ 코드 품질 향상 (중복 제거, 구조 개선) - 진행 중
3. ⚠️ 안정성 강화 (동시성 완료, 에러 처리/로깅 진행 필요)
4. ⚠️ 보안 강화 (진행 필요)
5. ⚠️ 성능 최적화 (선택적)

---

**다음 단계**: Phase 2.2 완료 (expand_query_semantically 추출)부터 시작 권장

