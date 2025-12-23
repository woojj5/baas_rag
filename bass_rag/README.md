# RAG Server (Gemma3 27B 기반)

로컬 Ollama (Gemma3 27B)를 사용하는 RAG 백엔드 시스템입니다.

## 프로젝트 목적

EV Battery·BMS·BAAS 전문 어시스턴트를 위한 로컬 RAG 백엔드입니다. 문서를 인덱싱하고, 질문에 대해 관련 문서를 참고하여 답변을 생성합니다.

## 기본 사용법

### 1. 가상환경 생성 및 패키지 설치

```bash
cd /home/keti_spark1/j309
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 문서 준비

`data/` 폴더에 `.txt` 또는 `.md` 파일을 넣습니다.

```bash
mkdir -p data
echo "EV Battery는 전기차의 핵심 부품입니다." > data/battery.txt
```

### 3. 인덱스 생성

```bash
python build_index.py
```

인덱스는 `index/` 폴더에 자동으로 생성됩니다.

### 4. FastAPI 서버 실행

```bash
uvicorn app.rag_server:app --host 0.0.0.0 --port 8000
```

### 5. API 사용 예시

#### 질문하기

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"EV Battery란 무엇인가요?","top_k":5}'
```

#### 문서 추가 (Ingest)

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"BMS는 배터리 관리 시스템입니다."}'
```

#### 헬스 체크

```bash
curl http://localhost:8000/health
```

## 주의사항

- **Ollama 설치 필요**: Ollama가 설치되어 있고 실행 중이어야 합니다.
- **모델 다운로드**: `gemma3:27b` 모델이 다운로드되어 있어야 합니다.
  ```bash
  ollama pull gemma3:27b
  ```
- **Ollama API**: 기본적으로 `http://localhost:11434`에서 실행되어야 합니다.

## 환경 변수 설정 (선택)

`.env` 파일을 생성하여 설정을 변경할 수 있습니다:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:27b
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=400
CHUNK_OVERLAP=100
TOP_K_DEFAULT=5
```

## API 엔드포인트

- `GET /health`: 서버 상태 확인
- `POST /query`: RAG 질문 처리
- `POST /ingest`: 새 문서 추가

API 문서는 서버 실행 후 `http://localhost:8000/docs`에서 확인할 수 있습니다.
