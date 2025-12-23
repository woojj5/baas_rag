# REFRAG 속도 최적화 적용 계획서

## 개요

현재 프롬프트 기반 REFRAG 구현에서 속도 향상을 위한 단계별 최적화 계획입니다.
Reference implementation의 핵심 아이디어를 프롬프트 기반 구조에 적용하여 **30-50% 속도 향상**을 목표로 합니다.

---

## Phase 1: 즉시 적용 가능한 최적화 (1-2일)

### 목표
- 프롬프트 토큰 수 30-40% 감소
- 속도 30-50% 향상
- 답변 품질 유지

### 작업 1.1: 프롬프트 토큰 수 최적화

**현재 문제점:**
- `system_prompt`에 200+ 토큰의 긴 규칙 텍스트 포함
- 불필요한 예시와 설명이 많음
- 중복된 설명 반복

**개선 방법:**
```python
# 현재: app/refrag_server.py:1389-1469
# 200+ 토큰의 긴 규칙

# 개선: 핵심만 간결하게
system_prompt += """규칙:
- 변수명 매핑 확인 (camelCase↔snake_case)
- 사용됨 변수 제외: hvacList1, moduleAvgTemp, cellVoltageList 등
- 답변: 변수명만 나열, 서론 없음
- 테이블 필터링: GPS/BMS 구분"""
```

**수정 파일:**
- `app/refrag_server.py` (1389-1469 라인)
- `app/rag/prompt_builder.py` (24-53 라인)

**예상 효과:**
- 프롬프트 토큰: 200+ → 100 토큰 (50% 감소)
- LLM 처리 시간: 20-30% 단축

**검증 방법:**
```python
# 토큰 수 측정
before_tokens = count_tokens(old_system_prompt)
after_tokens = count_tokens(new_system_prompt)
reduction = (before_tokens - after_tokens) / before_tokens * 100
print(f"토큰 수 감소: {reduction:.1f}%")
```

---

### 작업 1.2: 압축 비율 증가 (REFRAG 스타일)

**현재 상태:**
```python
# app/refrag_server.py:170
compression_policy = HeuristicCompressionPolicy(
    max_expanded_chunks=5,  # 8개 중 5개 확장 (62.5%)
    expand_frac=None
)
```

**개선 방법:**
```python
# app/refrag_server.py:170 수정
compression_policy = HeuristicCompressionPolicy(
    max_expanded_chunks=3,      # 5 → 3 (37.5%)
    expand_frac=0.25,            # REFRAG 스타일: 25%만 확장
    score_threshold=0.6          # 높은 스코어만 확장
)
```

**수정 파일:**
- `app/refrag_server.py` (170 라인)
- `app/config.py` (기본값 추가)

**예상 효과:**
- 확장 청크: 5개 → 3개 (40% 감소)
- 프롬프트 토큰: 40-50% 감소
- 속도: 30-40% 향상

**검증 방법:**
```python
# 압축 비율 확인
expanded_count = sum(1 for v in compression_decisions.values() if v == "EXPAND")
total_count = len(compression_decisions)
expand_ratio = expanded_count / total_count
print(f"확장 비율: {expand_ratio:.1%} (목표: 25-37%)")
```

---

### 작업 1.3: 동적 압축 강도 조절

**개선 방법:**
```python
# app/refrag_server.py:1347-1355 수정
# 질문 타입에 따라 압축 강도 조절

if is_uncertain_usage_query:
    # 불확실 질문: 더 공격적 압축 (이미 컬럼 정보 있음)
    compression_policy = HeuristicCompressionPolicy(
        max_expanded_chunks=2,  # 최소한만 확장
        expand_frac=0.2,
        score_threshold=0.7     # 매우 높은 스코어만
    )
elif is_not_used_query:
    # "사용하지 않는 변수" 질문: 중간 압축
    compression_policy = HeuristicCompressionPolicy(
        max_expanded_chunks=3,
        expand_frac=0.3,
        score_threshold=0.6
    )
else:
    # 일반 질문: 기본 압축
    compression_policy = HeuristicCompressionPolicy(
        max_expanded_chunks=4,
        expand_frac=0.4,
        score_threshold=0.5
    )

compression_decisions = compression_policy.decide(retrieved_chunks)
```

**수정 파일:**
- `app/refrag_server.py` (1347-1355 라인)

**예상 효과:**
- 질문 타입별 최적화
- 평균 속도: 10-20% 추가 향상

**검증 방법:**
```python
# 질문 타입별 속도 측정
timings_by_type = {
    "uncertain": [],
    "not_used": [],
    "general": []
}
# 각 타입별 평균 속도 비교
```

---

### 작업 1.4: 압축 방식 개선 (더 짧은 압축)

**현재 상태:**
```python
# app/compression/compressor.py:6
def compress_chunk(chunk: Chunk, mode="head", max_sentences: int = 3):
    # 3문장 유지
```

**개선 방법:**
```python
# app/compression/compressor.py:6 수정
def compress_chunk(chunk: Chunk, mode="head", max_sentences: int = 1):
    # 1문장만 유지 (더 공격적)
    # 또는 핵심 키워드만 추출
```

**옵션 1: 더 짧은 압축**
```python
# max_sentences를 3 → 1로 변경
compressed_text = '.'.join(sentences[:1])  # 1문장만
```

**옵션 2: 키워드 추출 (고급)**
```python
# 구조화된 데이터의 경우 핵심 필드만 추출
if has_structured_data:
    # 변수명, 테이블구분, 비고만 유지
    key_fields = ['변수명:', '테이블구분:', '비고:']
    preserved_lines = [line for line in lines 
                      if any(line.startswith(field) for field in key_fields)]
```

**수정 파일:**
- `app/compression/compressor.py` (6, 18-53 라인)

**예상 효과:**
- 압축된 청크 토큰: 50-70% 감소
- 전체 프롬프트: 20-30% 감소

---

## Phase 1 완료 체크리스트

- [ ] 작업 1.1: 프롬프트 토큰 수 최적화 완료
- [ ] 작업 1.2: 압축 비율 증가 완료
- [ ] 작업 1.3: 동적 압축 강도 조절 완료
- [ ] 작업 1.4: 압축 방식 개선 완료
- [ ] 성능 테스트: 속도 30-50% 향상 확인
- [ ] 품질 테스트: 답변 품질 유지 확인

**예상 완료 시간:** 1-2일

---

## Phase 2: 중기 개선 (1주)

### 목표
- 추가 20-30% 속도 향상
- 토큰 수 기반 동적 최적화
- 프롬프트 템플릿 최적화

### 작업 2.1: 토큰 수 기반 동적 조절

**개선 방법:**
```python
# app/refrag_server.py:1471-1472 이후 추가
prompt_token_count = count_tokens(full_prompt_text)

# 토큰 수에 따라 압축 강도 조절
if prompt_token_count > 2000:
    # 토큰 수가 많으면 더 공격적 압축
    logger.info(f"High token count ({prompt_token_count}), applying aggressive compression")
    # 재압축 또는 추가 압축
    compression_decisions = apply_aggressive_compression(
        retrieved_chunks, compression_decisions
    )
elif prompt_token_count > 1500:
    # 중간 압축
    compression_decisions = apply_moderate_compression(
        retrieved_chunks, compression_decisions
    )
```

**수정 파일:**
- `app/refrag_server.py` (1471-1472 라인 이후)
- `app/compression/policy.py` (새 함수 추가)

**예상 효과:**
- 긴 프롬프트 자동 최적화
- 평균 속도: 10-15% 추가 향상

---

### 작업 2.2: 프롬프트 템플릿 최적화

**현재 상태:**
```python
# app/rag/prompt_builder.py:74-82
context_parts.append("=== 관련성 높은 청크 (전체 내용) ===\n")
context_parts.append("=== 관련성 낮은 청크 (요약) ===\n")
```

**개선 방법:**
```python
# 불필요한 헤더/구분자 제거 또는 간소화
if expanded_chunks:
    context_parts.append("[HIGH]\n")  # 또는 아예 제거
    context_parts.extend(expanded_chunks)

if compressed_chunks:
    context_parts.append("[LOW]\n")  # 또는 아예 제거
    context_parts.extend(compressed_chunks)
```

**수정 파일:**
- `app/rag/prompt_builder.py` (74-82 라인)

**예상 효과:**
- 프롬프트 토큰: 5-10% 추가 감소

---

### 작업 2.3: 청크 수 동적 제한

**개선 방법:**
```python
# app/refrag_server.py:1297 라인 이후
# top_k를 동적으로 조절

# 프롬프트 토큰 수 예측
estimated_tokens = estimate_prompt_tokens(query_text, retrieved_chunks)

if estimated_tokens > 2000:
    # 청크 수 줄이기
    retrieved_chunks = retrieved_chunks[:min(5, len(retrieved_chunks))]
    logger.info(f"Reduced chunks to {len(retrieved_chunks)} due to high token count")
```

**수정 파일:**
- `app/refrag_server.py` (1297 라인 이후)

**예상 효과:**
- 긴 질문에서 자동 최적화
- 속도: 5-10% 추가 향상

---

## Phase 2 완료 체크리스트

- [ ] 작업 2.1: 토큰 수 기반 동적 조절 완료
- [ ] 작업 2.2: 프롬프트 템플릿 최적화 완료
- [ ] 작업 2.3: 청크 수 동적 제한 완료
- [ ] 성능 테스트: 추가 20-30% 향상 확인

**예상 완료 시간:** 1주

---

## Phase 3: 장기 전환 (선택적, 2-4주)

### 목표
- 완전한 REFRAG 구현 (TokenProjector 도입)
- 5-10배 속도 향상
- Reference implementation 수준

### 작업 3.1: TokenProjector 구현

**필요 작업:**
1. `app/refrag/token_projector.py` 생성
2. 인코더 임베딩 → 디코더 임베딩 공간 투영
3. `inputs_embeds` 사용

**주의사항:**
- 큰 구조 변경 필요
- LLM 모델 수정 필요 (Ollama 지원 확인)
- 학습/파인튜닝 필요할 수 있음

**예상 효과:**
- 5-10배 속도 향상
- 하지만 복잡도 대폭 증가

---

## 성능 측정 및 검증

### 측정 지표

1. **토큰 수 감소율**
```python
token_reduction = (before_tokens - after_tokens) / before_tokens * 100
```

2. **속도 향상율**
```python
speedup = (before_time - after_time) / before_time * 100
```

3. **답변 품질 (BLEU, ROUGE 등)**
- 답변 품질 유지 확인

### 테스트 케이스

```python
test_cases = [
    {
        "query": "BMS 테이블에 존재하는 변수지만, 사용하지 않는 변수는?",
        "expected_speedup": 0.3,  # 30% 이상 향상
        "quality_check": True
    },
    {
        "query": "GPS 테이블에 존재하는 변수지만, 사용하지 않는 변수는?",
        "expected_speedup": 0.3,
        "quality_check": True
    },
    # 추가 테스트 케이스...
]
```

---

## 리스크 관리

### 잠재적 리스크

1. **답변 품질 저하**
   - 리스크: 과도한 압축으로 정보 손실
   - 대응: 단계적 적용, 품질 테스트 필수

2. **특정 질문 타입에서 성능 저하**
   - 리스크: 복잡한 질문에서 답변 품질 저하
   - 대응: 질문 타입별 최적화, 폴백 메커니즘

3. **캐시 무효화**
   - 리스크: 변경으로 인한 캐시 불일치
   - 대응: 캐시 버전 관리

---

## 우선순위 요약

### 즉시 적용 (Phase 1)
1. ✅ 프롬프트 토큰 수 최적화 (가장 효과적)
2. ✅ 압축 비율 증가 (REFRAG 스타일)
3. ✅ 동적 압축 강도 조절
4. ✅ 압축 방식 개선

**예상 효과:** 30-50% 속도 향상

### 중기 개선 (Phase 2)
5. 토큰 수 기반 동적 조절
6. 프롬프트 템플릿 최적화
7. 청크 수 동적 제한

**예상 효과:** 추가 20-30% 향상

### 장기 전환 (Phase 3, 선택적)
8. 완전한 REFRAG 전환 (TokenProjector)

**예상 효과:** 5-10배 향상 (하지만 복잡도 증가)

---

## 다음 단계

**권장 시작점:** Phase 1 작업 1.1 (프롬프트 토큰 수 최적화)

이 작업부터 시작하면:
- 즉시 효과 확인 가능
- 리스크 낮음
- 답변 품질 유지 가능

Phase 1 작업을 시작할까요?

