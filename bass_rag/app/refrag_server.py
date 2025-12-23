"""REFRAG-style RAG FastAPI server."""
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import asyncio
import time

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from app.models.schemas import QueryRequest, QueryResponse, RetrievedChunk
from app.llm.ollama_client import OllamaLLMClient
from app.llm.hf_client import HuggingFaceLLMClient
from app.embeddings.st_client import SentenceTransformersEmbeddingClient
from app.index.vector_index import VectorIndex
from app.compression.policy import HeuristicCompressionPolicy
from app.rag.adapter import convert_existing_index_to_refrag
from app.rag_index import load_index, chunk_text, save_index
from app.config import Config
from app.domain_dict import get_domain_dict
from app.hybrid_retrieval import HybridRetriever
from app.reranker import DomainReranker
from app.rag.prompt_builder import build_refrag_prompt
from app.rag.token_projector import get_token_projector, project_embeddings
from app.utils.tokenizer import count_tokens, estimate_prompt_tokens
from app.utils.latency import measure_llm_latency
from app.data_columns import get_actual_used_columns, format_columns_for_prompt, format_missing_variables_info, format_unused_variables_info
from app.utils.postprocess import postprocess_answer
from app.utils.query_expansion import expand_query_semantically, expand_query_with_embeddings
from app.utils.reretrieval import extract_keywords_from_answer, should_reretrieve, build_reretrieval_query
from app.utils.logger import get_logger
from app.utils.security import sanitize_filename, validate_save_path, sanitize_error_message
from app.utils.index_saver import index_save_manager
from app.utils.cache import embedding_cache, chunk_embedding_cache
from app.sql_tools import (
    csv_preview, 
    infer_baas_schema,
    generate_basic_exploration_sql,
    generate_basic_stats_sql,
    generate_baas_domain_sql,
    generate_sql_from_csv_preview,
    db_basic_stats,
    generate_sql_from_db,
    influxdb_basic_stats,
    generate_flux_from_influxdb
)
from app.preprocessor import preprocess_all_files, preprocess_bms_file, preprocess_gps_file
import re

logger = get_logger(__name__)

app = FastAPI(title="REFRAG-style RAG API", version="0.1.0")


# postprocess_answer is now imported from app.utils.postprocess

# Pydantic models
class IngestRequest(BaseModel):
    text: str


class DeletePassageRequest(BaseModel):
    indices: List[int]


class SQLRequest(BaseModel):
    csv_path: Optional[str] = None
    table_name: Optional[str] = None
    question: Optional[str] = None


class DBConnectionRequest(BaseModel):
    db_url: str
    table_name: str
    schema_name: Optional[str] = None  # Renamed from 'schema' to avoid BaseModel conflict


class InfluxDBConnectionRequest(BaseModel):
    url: str  # InfluxDB URL (e.g., "http://localhost:8086" or "influxdb://token@host:port/bucket?org=org")
    bucket: str
    measurement: Optional[str] = None
    org: Optional[str] = None
    token: Optional[str] = None


# Global instances
llm_client: OllamaLLMClient | None = None
embedding_client: SentenceTransformersEmbeddingClient | None = None
vector_index: VectorIndex | None = None
compression_policy: HeuristicCompressionPolicy | None = None
# For hybrid search
faiss_index: faiss.Index | None = None
passages: List[str] = []
passages_lower: List[str] = []
embedding_model: SentenceTransformer | None = None
hybrid_retriever: HybridRetriever | None = None
reranker: DomainReranker | None = None
domain_dict = None
# Phase 3: TokenProjector (optional)
token_projector = None

# Lock for thread-safe access to index and passages
# Using asyncio.Lock for async/await compatibility
index_lock = asyncio.Lock()  # Protects faiss_index and passages from concurrent access


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on server shutdown - force save any pending index updates."""
    global faiss_index, passages, passages_lower
    if faiss_index is not None and passages:
        try:
            logger.info("Shutting down - saving any pending index updates")
            # Force immediate save without new_embeddings (save current state)
            await index_save_manager.force_save(faiss_index, passages, new_embeddings=None)
            logger.info("Index saved successfully on shutdown")
        except Exception as e:
            logger.error(f"Error during shutdown save: {e}")
            # Try direct save as fallback
            try:
                from app.rag_index import save_index
                save_index(faiss_index, passages)
                logger.info("Index saved using fallback method")
            except Exception as e2:
                logger.error(f"Fallback save also failed: {e2}")


@app.on_event("startup")
async def startup():
    """Initialize REFRAG components on startup."""
    global llm_client, embedding_client, vector_index, compression_policy
    global faiss_index, passages, passages_lower, embedding_model, hybrid_retriever, reranker, domain_dict
    global token_projector
    
    try:
        # Validate configuration
        logger.info("Validating configuration...")
        Config.validate_config()
        logger.info("Configuration validation passed")
        
        # Initialize LLM client (Ollama or HuggingFace)
        logger.info(f"Initializing LLM client (backend: {Config.LLM_BACKEND})...")
        if Config.LLM_BACKEND == "huggingface":
            try:
                llm_client = HuggingFaceLLMClient(
                    model_name=Config.HF_MODEL_NAME,
                    device=Config.HF_DEVICE,
                    use_token_projector=Config.USE_TOKEN_PROJECTOR
                )
                logger.info(f"HuggingFace LLM client initialized: {Config.HF_MODEL_NAME}")
            except Exception as e:
                logger.error(f"Failed to initialize HuggingFace client: {e}")
                logger.warning("Falling back to Ollama client")
                llm_client = OllamaLLMClient()
        else:
            llm_client = OllamaLLMClient()
            logger.info(f"Ollama LLM client initialized: {Config.OLLAMA_MODEL}")
        
        # Initialize embedding client
        logger.info("Initializing embedding client...")
        embedding_client = SentenceTransformersEmbeddingClient()
        
        # Phase 3: Initialize TokenProjector (if enabled)
        if Config.USE_TOKEN_PROJECTOR:
            logger.info("Initializing TokenProjector...")
            token_projector = get_token_projector(
                encoder_dim=Config.ENCODER_EMBEDDING_DIM,
                decoder_dim=Config.DECODER_EMBEDDING_DIM,
                use_pretrained=Config.TOKEN_PROJECTOR_PATH is not None,
                projector_path=Config.TOKEN_PROJECTOR_PATH
            )
            logger.info("TokenProjector initialized (Note: Ollama doesn't support inputs_embeds, so TokenProjector is currently used for embedding matching only)")
        else:
            logger.info("TokenProjector disabled (set USE_TOKEN_PROJECTOR=true to enable)")
            token_projector = None
        
        # Load existing index and convert to REFRAG format
        logger.info("Loading existing index...")
        loaded_faiss_index, loaded_passages, model_name, doc_id_to_original, saved_embeddings = load_index()
        
        if saved_embeddings is not None:
            logger.info(f"Found saved embeddings ({saved_embeddings.shape[0]} vectors) - will skip re-embedding")
        else:
            logger.info("No saved embeddings found - will re-embed chunks (this may take time)")
        
        # Store for hybrid search
        faiss_index = loaded_faiss_index
        passages = loaded_passages
        # Precompute lowercased passages once for faster per-query checks
        passages_lower = [p.lower() for p in passages]
        embedding_model = SentenceTransformer(model_name)
        logger.info(f"Loaded {len(passages)} passages")
        
        # Build domain dictionary
        logger.info("Building domain dictionary...")
        domain_dict = get_domain_dict()
        logger.info(f"Domain dictionary built: {len(domain_dict.variable_to_info)} variables, {len(domain_dict.table_to_variables)} tables")
        
        # Initialize hybrid retriever
        logger.info("Initializing hybrid retriever...")
        hybrid_retriever = HybridRetriever(passages, domain_dict)
        logger.info("Hybrid retriever initialized")
        
        # Initialize reranker
        logger.info("Initializing reranker...")
        reranker = DomainReranker(domain_dict, use_cross_encoder=True)
        logger.info("Reranker initialized")
        
        logger.info("Converting index to REFRAG format...")
        vector_index = convert_existing_index_to_refrag(
            loaded_faiss_index,
            loaded_passages,
            embedding_client,
            saved_embeddings=saved_embeddings  # Pass saved embeddings to avoid re-embedding
        )
        logger.info(f"Index loaded: {len(vector_index)} chunks")
        
        # Initialize compression policy (REFRAG-style: expand_frac=0.25)
        compression_policy = HeuristicCompressionPolicy(
            max_expanded_chunks=3,  # Reduced from 5 for better performance
            expand_frac=Config.EXPAND_FRAC,  # REFRAG-style: 25% only
            score_threshold=0.6  # Higher threshold for expansion
        )
        
        logger.info("REFRAG server initialized successfully")
    except FileNotFoundError as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Index not found: {error_msg}")
        raise RuntimeError(
            f"Index not found. Please run 'python build_index.py' first. "
            f"Error: {error_msg}"
        )
    except ValueError as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Configuration validation failed: {error_msg}")
        raise RuntimeError(f"Configuration error: {error_msg}")
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.exception(f"Failed to initialize REFRAG server: {error_msg}")
        raise RuntimeError(f"Failed to initialize REFRAG server: {error_msg}")


@app.get("/", response_class=HTMLResponse)
def root_page() -> str:
    """REFRAG RAG frontend page."""
    return r"""
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <title>REFRAG RAG Demo</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      background: #0f172a;
      color: #e5e7eb;
    }
    .container {
      max-width: 960px;
      margin: 0 auto;
      padding: 24px 16px 40px;
    }
    h1 {
      font-size: 1.8rem;
      margin-bottom: 4px;
    }
    .subtitle {
      color: #9ca3af;
      margin-bottom: 24px;
      font-size: 0.95rem;
    }
    .card {
      background: #020617;
      border-radius: 14px;
      padding: 18px 18px 16px;
      border: 1px solid #1f2937;
      box-shadow: 0 18px 45px rgba(15,23,42,0.85);
      margin-bottom: 16px;
    }
    .card h2 {
      font-size: 1.05rem;
      margin: 0 0 10px;
    }
    label {
      display: block;
      font-size: 0.8rem;
      margin-bottom: 4px;
      color: #9ca3af;
    }
    textarea, input {
      width: 100%;
      box-sizing: border-box;
      border-radius: 10px;
      border: 1px solid #4b5563;
      background: #020617;
      color: #e5e7eb;
      padding: 8px 10px;
      font-size: 0.9rem;
      outline: none;
      resize: vertical;
      min-height: 60px;
    }
    textarea:focus, input:focus {
      border-color: #38bdf8;
      box-shadow: 0 0 0 1px rgba(56,189,248,0.35);
    }
    input[type="number"] {
      max-width: 120px;
    }
    button {
      border-radius: 999px;
      border: none;
      padding: 8px 16px;
      font-size: 0.9rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: linear-gradient(135deg, #8b5cf6, #6366f1);
      color: #fff;
      font-weight: 600;
      margin-top: 8px;
    }
    button.secondary {
      background: #111827;
      color: #e5e7eb;
      border: 1px solid #374151;
    }
    button:disabled {
      opacity: 0.6;
      cursor: default;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: flex-end;
      margin-top: 6px;
    }
    .flex-1 { flex: 1; }
    .answer, .contexts {
      white-space: pre-wrap;
      font-size: 0.9rem;
      line-height: 1.5;
    }
    .contexts ul {
      padding-left: 18px;
    }
    .contexts li {
      margin-bottom: 4px;
    }
    .status-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.75rem;
      color: #9ca3af;
      margin-top: 6px;
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 999px;
      margin-right: 4px;
      background: #22c55e;
      box-shadow: 0 0 0 4px rgba(34,197,94,0.2);
    }
    .status-dot.offline {
      background: #f97373;
      box-shadow: 0 0 0 4px rgba(239,68,68,0.16);
    }
    .status-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .small {
      font-size: 0.75rem;
    }
    .muted {
      color: #6b7280;
    }
    .info-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    .info-item {
      background: #111827;
      padding: 8px;
      border-radius: 6px;
      font-size: 0.75rem;
    }
    .info-label {
      color: #9ca3af;
    }
    .info-value {
      color: #e5e7eb;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>REFRAG RAG Assistant</h1>
    <div class="subtitle">Compress · Sense · Expand · Gemma3 27B · Ollama</div>

    <div class="card">
      <div class="status-bar">
        <div class="status-chip">
          <span id="apiStatusDot" class="status-dot"></span>
          <span id="apiStatusText">API 연결 확인 중...</span>
        </div>
        <div class="muted small">/query · /health</div>
      </div>
    </div>

    <div class="card">
      <h2>질문하기</h2>
      <label for="question">질문</label>
      <textarea id="question" placeholder="예) BMS 테이블에 존재는 하지만 실제로는 사용되지 않는 변수는?"></textarea>
      <div class="row">
        <div>
          <label for="topK">Top K</label>
          <input id="topK" type="number" min="1" max="32" value="8" />
        </div>
        <div class="flex-1"></div>
        <button id="askBtn">
          <span>질문 보내기</span>
        </button>
      </div>
      <div id="queryStatus" class="small muted" style="margin-top:6px;"></div>
    </div>

    <div class="card">
      <h2>답변</h2>
      <div id="answer" class="answer muted">아직 질문이 없습니다.</div>
      <div id="refragInfo" class="info-grid" style="display:none; margin-top:12px;"></div>
    </div>

    <div class="card">
      <h2>사용된 청크</h2>
      <div id="contexts" class="contexts muted small">질문을 보내면, 여기 사용된 청크가 표시됩니다.</div>
    </div>

    <div class="card">
      <h2>문서 추가 (Ingest)</h2>
      <label for="ingestText">새 텍스트 블록</label>
      <textarea id="ingestText" placeholder="새로 추가할 문서를 여기에 붙여넣으세요."></textarea>
      <div class="row">
        <button id="ingestBtn" class="secondary">
          <span>인덱스에 추가</span>
        </button>
        <div id="ingestStatus" class="small muted" style="margin-left:8px;"></div>
      </div>
    </div>

    <div class="card">
      <h2>파일 업로드</h2>
      <label for="fileUpload">문서 파일 (.txt, .md) 또는 CSV 파일 (.csv)</label>
      <input type="file" id="fileUpload" accept=".txt,.md,.csv" style="margin-bottom:8px;" />
      <div class="small muted" style="margin-bottom:8px;">
        • .txt, .md: 파이프(|) 구분자 형식이면 before_preprocess/ 디렉토리에 저장, 아니면 RAG 인덱스에 추가<br>
        • .csv: 파이프(|) 구분자 형식이거나 before_preprocess 형식(bms.{device_no}.{year}-{month}.csv 또는 gps.{device_no}.{year}-{month}.csv)이면 before_preprocess/ 디렉토리에 저장
      </div>
      <div class="row">
        <button id="uploadBtn" class="secondary">
          <span>파일 업로드</span>
        </button>
        <div id="uploadStatus" class="small muted" style="margin-left:8px;"></div>
      </div>
    </div>

    <div class="card">
      <h2>패시지 관리</h2>
      <div class="row">
        <button id="listPassagesBtn" class="secondary">
          <span>패시지 목록 보기</span>
        </button>
        <button id="deleteAllBtn" class="secondary" style="background:#dc2626; color:#fff; border-color:#dc2626;">
          <span>전체 삭제</span>
        </button>
        <div id="passagesStatus" class="small muted" style="margin-left:8px;"></div>
      </div>
      <div id="passagesList" style="margin-top:12px; max-height:300px; overflow-y:auto; display:none;"></div>
    </div>

    <div class="card">
      <h2>데이터 전처리</h2>
      <div style="margin-bottom:8px;">
        <label>before_preprocess 디렉토리의 파일들을 전처리합니다.</label>
      </div>
      <div class="row">
        <button id="preprocessBtn" class="secondary" style="background:linear-gradient(135deg, #f59e0b, #ef4444);">
          <span>전처리 실행</span>
        </button>
        <div id="preprocessStatus" class="small muted" style="margin-left:8px;"></div>
      </div>
      <div id="preprocessResult" style="margin-top:12px; display:none; font-size:0.85rem;"></div>
    </div>

    <div class="card">
      <h2 id="queryTitle">쿼리문 생성</h2>
      <div style="margin-bottom:16px;">
        <label style="display:block; margin-bottom:8px; font-weight:600;">데이터 소스 선택</label>
        <div style="display:flex; gap:8px;">
          <button id="sourceCsvBtn" class="secondary" style="background:#22c55e; color:#020617;">CSV 파일</button>
          <button id="sourceDbBtn" class="secondary">PostgreSQL</button>
          <button id="sourceInfluxBtn" class="secondary">InfluxDB</button>
        </div>
      </div>

      <div id="csvSource" style="display:block;">
        <label for="csvPath">CSV 파일 경로</label>
        <input type="text" id="csvPath" placeholder="/home/keti_spark1/j309/data/preprocessed_bms.01241248529.2022-12.csv" style="margin-bottom:8px;" />
        <label for="tableName">테이블 이름 (선택)</label>
        <input type="text" id="tableName" placeholder="baas_data" value="baas_data" style="margin-bottom:8px;" />
      </div>

      <div id="dbSource" style="display:none;">
        <label for="dbUrl">PostgreSQL URL</label>
        <input type="text" id="dbUrl" placeholder="postgresql://user:password@host:port/dbname" style="margin-bottom:8px;" />
        <label for="dbTableName">테이블 이름</label>
        <input type="text" id="dbTableName" placeholder="baas_data" style="margin-bottom:8px;" />
        <label for="dbSchema">스키마 (선택)</label>
        <input type="text" id="dbSchema" placeholder="public" style="margin-bottom:8px;" />
      </div>

      <div id="influxSource" style="display:none;">
        <label for="influxUrl">InfluxDB URL</label>
        <input type="text" id="influxUrl" placeholder="http://localhost:8086" style="margin-bottom:8px;" />
        <label for="influxBucket">Bucket</label>
        <input type="text" id="influxBucket" placeholder="my_bucket" style="margin-bottom:8px;" />
        <label for="influxMeasurement">Measurement (선택)</label>
        <input type="text" id="influxMeasurement" placeholder="sensor_data" style="margin-bottom:8px;" />
        <label for="influxOrg">Organization (선택)</label>
        <input type="text" id="influxOrg" placeholder="myorg" style="margin-bottom:8px;" />
        <label for="influxToken">Token (선택)</label>
        <input type="password" id="influxToken" placeholder="your-token" style="margin-bottom:8px;" />
      </div>

      <div class="row">
        <button id="generateSqlBtn" class="secondary" style="background:linear-gradient(135deg, #8b5cf6, #6366f1);">
          <span id="generateBtnText">SQL 생성</span>
        </button>
        <div id="sqlStatus" class="small muted" style="margin-left:8px;"></div>
      </div>
      <div id="sqlResult" style="margin-top:12px; display:none;">
        <div style="background:#111827; padding:12px; border-radius:8px; border:1px solid #1f2937;">
          <div id="sqlSchema" style="margin-bottom:12px; font-size:0.85rem; white-space:pre-wrap; color:#e5e7eb;"></div>
          <div id="sqlCode" style="font-size:0.8rem; white-space:pre-wrap; color:#9ca3af; font-family:monospace; background:#020617; padding:10px; border-radius:6px; overflow-x:auto;"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const apiStatusDot = document.getElementById("apiStatusDot");
    const apiStatusText = document.getElementById("apiStatusText");
    const askBtn = document.getElementById("askBtn");
    const ingestBtn = document.getElementById("ingestBtn");
    const uploadBtn = document.getElementById("uploadBtn");
    const listPassagesBtn = document.getElementById("listPassagesBtn");
    const deleteAllBtn = document.getElementById("deleteAllBtn");
    const preprocessBtn = document.getElementById("preprocessBtn");
    const preprocessStatusEl = document.getElementById("preprocessStatus");
    const preprocessResultEl = document.getElementById("preprocessResult");
    const generateSqlBtn = document.getElementById("generateSqlBtn");
    const sourceCsvBtn = document.getElementById("sourceCsvBtn");
    const sourceDbBtn = document.getElementById("sourceDbBtn");
    const sourceInfluxBtn = document.getElementById("sourceInfluxBtn");
    const csvSourceEl = document.getElementById("csvSource");
    const dbSourceEl = document.getElementById("dbSource");
    const influxSourceEl = document.getElementById("influxSource");
    const influxUrlEl = document.getElementById("influxUrl");
    const influxBucketEl = document.getElementById("influxBucket");
    const influxMeasurementEl = document.getElementById("influxMeasurement");
    const influxOrgEl = document.getElementById("influxOrg");
    const influxTokenEl = document.getElementById("influxToken");
    const csvPathEl = document.getElementById("csvPath");
    const tableNameEl = document.getElementById("tableName");
    const dbUrlEl = document.getElementById("dbUrl");
    const dbTableNameEl = document.getElementById("dbTableName");
    const dbSchemaEl = document.getElementById("dbSchema");
    const sqlResultEl = document.getElementById("sqlResult");
    const sqlSchemaEl = document.getElementById("sqlSchema");
    const sqlCodeEl = document.getElementById("sqlCode");
    const sqlStatusEl = document.getElementById("sqlStatus");
    const generateBtnTextEl = document.getElementById("generateBtnText");
    const queryTitleEl = document.getElementById("queryTitle");
    const fileUploadEl = document.getElementById("fileUpload");
    const passagesListEl = document.getElementById("passagesList");
    const passagesStatusEl = document.getElementById("passagesStatus");
    const questionEl = document.getElementById("question");
    const topKEl = document.getElementById("topK");
    const answerEl = document.getElementById("answer");
    const contextsEl = document.getElementById("contexts");
    const ingestTextEl = document.getElementById("ingestText");
    const queryStatusEl = document.getElementById("queryStatus");
    const ingestStatusEl = document.getElementById("ingestStatus");
    const uploadStatusEl = document.getElementById("uploadStatus");
    const refragInfoEl = document.getElementById("refragInfo");

    async function checkHealth() {
      try {
        const res = await fetch("/health");
        if (!res.ok) throw new Error();
        const data = await res.json();
        apiStatusDot.classList.remove("offline");
        // Try to get num_passages from /passages endpoint
        try {
          const passagesRes = await fetch("/passages");
          if (passagesRes.ok) {
            const passagesData = await passagesRes.json();
            apiStatusText.textContent =
              data.index_loaded
                ? `온라인 · 청크 ${data.num_chunks}개 · 패시지 ${passagesData.count || 0}개`
                : "온라인 · 인덱스 미로딩";
          } else {
            apiStatusText.textContent =
              data.index_loaded
                ? `온라인 · 청크 ${data.num_chunks}개`
                : "온라인 · 인덱스 미로딩";
          }
        } catch (e) {
          apiStatusText.textContent =
            data.index_loaded
              ? `온라인 · 청크 ${data.num_chunks}개`
              : "온라인 · 인덱스 미로딩";
        }
      } catch (e) {
        apiStatusDot.classList.add("offline");
        apiStatusText.textContent = "오프라인 · 서버/인덱스 확인 필요";
      }
    }

    async function sendQuery() {
      const query = questionEl.value.trim();
      const topK = parseInt(topKEl.value || "8", 10);
      if (!query) {
        alert("질문을 입력하세요.");
        return;
      }
      askBtn.disabled = true;
      queryStatusEl.textContent = "질문 처리 중...";
      answerEl.textContent = "";
      contextsEl.textContent = "";
      refragInfoEl.style.display = "none";

      const clientStartTime = performance.now();

      try {
        const res = await fetch("/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, top_k: topK })
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "요청 실패");
        }
        const data = await res.json();
        
        const clientEndTime = performance.now();
        const clientTotalTime = ((clientEndTime - clientStartTime) / 1000).toFixed(2);
        
        answerEl.classList.remove("muted");
        answerEl.textContent = data.answer || "(빈 응답)";

        // REFRAG 정보 표시
        if (data.prompt_token_count !== undefined || data.llm_latency_ms !== undefined || data.compression_decisions) {
          let infoHtml = "";
          if (data.prompt_token_count !== undefined) {
            infoHtml += `<div class="info-item"><div class="info-label">프롬프트 토큰</div><div class="info-value">${data.prompt_token_count}</div></div>`;
          }
          if (data.llm_latency_ms !== undefined) {
            infoHtml += `<div class="info-item"><div class="info-label">LLM 지연시간</div><div class="info-value">${data.llm_latency_ms.toFixed(0)}ms</div></div>`;
          }
          if (data.compression_decisions) {
            const expanded = Object.values(data.compression_decisions).filter(v => v === "EXPAND").length;
            const compressed = Object.values(data.compression_decisions).filter(v => v === "COMPRESS").length;
            infoHtml += `<div class="info-item"><div class="info-label">확장 청크</div><div class="info-value">${expanded}</div></div>`;
            infoHtml += `<div class="info-item"><div class="info-label">압축 청크</div><div class="info-value">${compressed}</div></div>`;
          }
          refragInfoEl.innerHTML = infoHtml;
          refragInfoEl.style.display = "grid";
        }

        if (Array.isArray(data.used_chunks) && data.used_chunks.length > 0) {
          const items = data.used_chunks.map((chunk, idx) => {
            const decision = data.compression_decisions?.[chunk.id] || "UNKNOWN";
            const decisionBadge = decision === "EXPAND" ? '<span style="color:#22c55e;">[EXPAND]</span>' : 
                                 decision === "COMPRESS" ? '<span style="color:#f59e0b;">[COMPRESS]</span>' : '';
            const preview = chunk.text.length > 150 ? chunk.text.substring(0, 150) + "..." : chunk.text;
            return `<li>${decisionBadge} ${preview.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</li>`;
          }).join("");
          contextsEl.classList.remove("muted");
          contextsEl.innerHTML = `<ul>${items}</ul>`;
        } else {
          contextsEl.classList.add("muted");
          contextsEl.textContent = "사용된 청크가 없습니다.";
        }
        
        let timingText = `완료 · 총 시간: ${clientTotalTime}초`;
        if (data.llm_latency_ms !== undefined) {
          timingText += ` (LLM: ${(data.llm_latency_ms / 1000).toFixed(2)}초)`;
        }
        queryStatusEl.textContent = timingText;
      } catch (e) {
        queryStatusEl.textContent = "에러: " + e.message;
        answerEl.classList.add("muted");
        answerEl.textContent = "요청 중 에러가 발생했습니다.";
      } finally {
        askBtn.disabled = false;
      }
    }

    async function sendIngest() {
      const text = ingestTextEl.value.trim();
      if (!text) {
        alert("추가할 텍스트를 입력하세요.");
        return;
      }
      ingestBtn.disabled = true;
      ingestStatusEl.textContent = "인덱스에 추가 중...";
      try {
        const res = await fetch("/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text })
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "요청 실패");
        }
        const data = await res.json();
        ingestStatusEl.textContent =
          `완료 · 추가된 청크: ${data.added_chunks ?? 0}개`;
        ingestTextEl.value = "";
        checkHealth();
      } catch (e) {
        ingestStatusEl.textContent = "에러: " + e.message;
      } finally {
        ingestBtn.disabled = false;
      }
    }

    async function sendUpload() {
      const file = fileUploadEl.files[0];
      if (!file) {
        alert("파일을 선택하세요.");
        return;
      }
      if (!file.name.match(/\.(txt|md|csv)$/i)) {
        alert("텍스트 파일(.txt, .md) 또는 CSV 파일(.csv)만 업로드 가능합니다.");
        return;
      }
      uploadBtn.disabled = true;
      uploadStatusEl.textContent = "파일 업로드 중...";
      try {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch("/upload", {
          method: "POST",
          body: formData
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "업로드 실패");
        }
        const data = await res.json();
        
        // Handle CSV files differently
        if (file.name.toLowerCase().endsWith('.csv')) {
          uploadStatusEl.textContent = `완료 · ${data.message || `파일: ${data.filename} · 저장 위치: ${data.saved_to || 'N/A'}`}`;
        } else {
          // Text files added to index
          uploadStatusEl.textContent =
            `완료 · 파일: ${data.filename} · 추가된 청크: ${data.added_chunks ?? 0}개`;
        }
        
        fileUploadEl.value = "";
        checkHealth();
      } catch (e) {
        uploadStatusEl.textContent = "에러: " + e.message;
      } finally {
        uploadBtn.disabled = false;
      }
    }

    async function listPassages() {
      listPassagesBtn.disabled = true;
      passagesStatusEl.textContent = "로딩 중...";
      try {
        const res = await fetch("/passages");
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "요청 실패");
        }
        const data = await res.json();
        const list = data.passages || [];
        if (list.length === 0) {
          passagesListEl.style.display = "none";
          passagesStatusEl.textContent = "패시지가 없습니다.";
          return;
        }
        let html = "<div style='font-size:0.85rem;'>";
        list.forEach((p, idx) => {
          const preview = p.length > 100 ? p.substring(0, 100) + "..." : p;
          html += `<div style='padding:8px; margin-bottom:6px; background:#111827; border-radius:6px; border:1px solid #1f2937;'>
            <div style='display:flex; justify-content:space-between; align-items:start;'>
              <div style='flex:1;'>
                <div style='color:#9ca3af; font-size:0.7rem; margin-bottom:4px;'>#${idx}</div>
                <div style='color:#e5e7eb; line-height:1.4;'>${preview.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>
              </div>
              <button onclick='deletePassage(${idx})' style='margin-left:8px; padding:4px 8px; font-size:0.7rem; background:#dc2626; color:#fff; border:none; border-radius:4px; cursor:pointer;'>삭제</button>
            </div>
          </div>`;
        });
        html += "</div>";
        passagesListEl.innerHTML = html;
        passagesListEl.style.display = "block";
        passagesStatusEl.textContent = `총 ${list.length}개 패시지`;
      } catch (e) {
        passagesStatusEl.textContent = "에러: " + e.message;
        passagesListEl.style.display = "none";
      } finally {
        listPassagesBtn.disabled = false;
      }
    }

    async function deletePassage(idx) {
      if (!confirm(`패시지 #${idx}를 삭제하시겠습니까?`)) return;
      try {
        const res = await fetch("/passages/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ indices: [idx] })
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "삭제 실패");
        }
        const data = await res.json();
        alert(`삭제 완료. 남은 패시지: ${data.remaining_count}개`);
        listPassages();
        checkHealth();
      } catch (e) {
        alert("삭제 실패: " + e.message);
      }
    }

    async function deleteAllPassages() {
      if (!confirm("모든 패시지를 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.")) return;
      try {
        const res = await fetch("/passages/delete-all", {
          method: "POST"
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "삭제 실패");
        }
        const data = await res.json();
        alert(`전체 삭제 완료. 삭제된 패시지: ${data.deleted_count}개`);
        passagesListEl.style.display = "none";
        listPassages();
        checkHealth();
      } catch (e) {
        alert("삭제 실패: " + e.message);
      }
    }

    window.deletePassage = deletePassage;

    async function runPreprocess() {
      preprocessBtn.disabled = true;
      preprocessStatusEl.textContent = "전처리 실행 중...";
      preprocessResultEl.style.display = "none";
      
      try {
        const res = await fetch("/preprocess", {
          method: "POST"
        });
        
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "전처리 실패");
        }
        
        const data = await res.json();
        let resultHtml = "<div style='background:#111827; padding:10px; border-radius:6px;'>";
        resultHtml += `<div style='color:#22c55e; margin-bottom:8px;'>✅ 성공: ${data.processed_count}개 파일</div>`;
        
        if (data.processed && data.processed.length > 0) {
          resultHtml += "<div style='margin-top:8px;'>";
          data.processed.forEach(item => {
            resultHtml += `<div style='margin-bottom:4px; color:#e5e7eb;'>`;
            resultHtml += `[${item.type}] ${item.file}<br>`;
            resultHtml += `<span style='color:#9ca3af; font-size:0.8rem;'>행: ${item.stats.original_rows}, 컬럼: ${item.stats.columns}</span>`;
            resultHtml += `</div>`;
          });
          resultHtml += "</div>";
        }
        
        if (data.errors && data.errors.length > 0) {
          resultHtml += `<div style='color:#f97316; margin-top:8px;'>❌ 실패: ${data.errors.length}개</div>`;
        }
        
        resultHtml += "</div>";
        preprocessResultEl.innerHTML = resultHtml;
        preprocessResultEl.style.display = "block";
        preprocessStatusEl.textContent = "완료";
      } catch (e) {
        preprocessStatusEl.textContent = "에러: " + e.message;
        preprocessResultEl.style.display = "none";
      } finally {
        preprocessBtn.disabled = false;
      }
    }

    let currentSource = "csv";
    
    sourceCsvBtn.addEventListener("click", () => {
      currentSource = "csv";
      csvSourceEl.style.display = "block";
      dbSourceEl.style.display = "none";
      influxSourceEl.style.display = "none";
      sourceCsvBtn.style.background = "#22c55e";
      sourceCsvBtn.style.color = "#020617";
      sourceDbBtn.style.background = "#111827";
      sourceDbBtn.style.color = "#e5e7eb";
      sourceInfluxBtn.style.background = "#111827";
      sourceInfluxBtn.style.color = "#e5e7eb";
      generateBtnTextEl.textContent = "SQL 생성";
      queryTitleEl.textContent = "쿼리문 생성";
    });
    
    sourceDbBtn.addEventListener("click", () => {
      currentSource = "db";
      csvSourceEl.style.display = "none";
      dbSourceEl.style.display = "block";
      influxSourceEl.style.display = "none";
      sourceDbBtn.style.background = "#22c55e";
      sourceDbBtn.style.color = "#020617";
      sourceCsvBtn.style.background = "#111827";
      sourceCsvBtn.style.color = "#e5e7eb";
      sourceInfluxBtn.style.background = "#111827";
      sourceInfluxBtn.style.color = "#e5e7eb";
      generateBtnTextEl.textContent = "SQL 생성";
      queryTitleEl.textContent = "쿼리문 생성";
    });

    sourceInfluxBtn.addEventListener("click", () => {
      currentSource = "influx";
      csvSourceEl.style.display = "none";
      dbSourceEl.style.display = "none";
      influxSourceEl.style.display = "block";
      sourceInfluxBtn.style.background = "#22c55e";
      sourceInfluxBtn.style.color = "#020617";
      sourceCsvBtn.style.background = "#111827";
      sourceCsvBtn.style.color = "#e5e7eb";
      sourceDbBtn.style.background = "#111827";
      sourceDbBtn.style.color = "#e5e7eb";
      generateBtnTextEl.textContent = "Flux 생성";
      queryTitleEl.textContent = "쿼리문 생성";
    });

    async function generateSQL() {
      generateSqlBtn.disabled = true;
      const statusText = currentSource === "influx" ? "Flux 생성 중..." : "SQL 생성 중...";
      sqlStatusEl.textContent = statusText;
      sqlResultEl.style.display = "none";
      
      try {
        let res;
        if (currentSource === "csv") {
          const csvPath = csvPathEl.value.trim();
          const tableName = tableNameEl.value.trim() || "data_table";
          
          if (!csvPath) {
            alert("CSV 파일 경로를 입력하세요.");
            return;
          }
          
          res = await fetch("/sql/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ csv_path: csvPath, table_name: tableName })
          });
        } else if (currentSource === "db") {
          const dbUrl = dbUrlEl.value.trim();
          const dbTableName = dbTableNameEl.value.trim();
          const dbSchema = dbSchemaEl.value.trim() || null;
          
          if (!dbUrl || !dbTableName) {
            alert("데이터베이스 URL과 테이블 이름을 입력하세요.");
            return;
          }
          
          res = await fetch("/sql/generate-from-db", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
              db_url: dbUrl, 
              table_name: dbTableName,
              schema_name: dbSchema
            })
          });
        } else if (currentSource === "influx") {
          const influxUrl = influxUrlEl.value.trim();
          const influxBucket = influxBucketEl.value.trim();
          const influxMeasurement = influxMeasurementEl.value.trim() || null;
          const influxOrg = influxOrgEl.value.trim() || null;
          const influxToken = influxTokenEl.value.trim() || null;
          
          if (!influxUrl || !influxBucket) {
            alert("InfluxDB URL과 Bucket을 입력하세요.");
            return;
          }
          
          res = await fetch("/influxdb/generate-flux", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
              url: influxUrl,
              bucket: influxBucket,
              measurement: influxMeasurement,
              org: influxOrg,
              token: influxToken
            })
          });
        }
        
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "SQL 생성 실패");
        }
        
        const data = await res.json();
        sqlSchemaEl.textContent = data.schema || "";
        // For InfluxDB, use 'flux' instead of 'sql'
        sqlCodeEl.textContent = data.flux || data.sql || "";
        sqlResultEl.style.display = "block";
        sqlStatusEl.textContent = "완료";
      } catch (e) {
        sqlStatusEl.textContent = "에러: " + e.message;
        sqlResultEl.style.display = "none";
      } finally {
        generateSqlBtn.disabled = false;
      }
    }

    askBtn.addEventListener("click", () => { sendQuery(); });
    ingestBtn.addEventListener("click", () => { sendIngest(); });
    uploadBtn.addEventListener("click", () => { sendUpload(); });
    listPassagesBtn.addEventListener("click", () => { listPassages(); });
    deleteAllBtn.addEventListener("click", () => { deleteAllPassages(); });
    preprocessBtn.addEventListener("click", () => { runPreprocess(); });
    generateSqlBtn.addEventListener("click", () => { generateSQL(); });
    questionEl.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        sendQuery();
      }
    });

    checkHealth();
    setInterval(checkHealth, 15000);
  </script>
</body>
</html>
    """


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "index_loaded": vector_index is not None,
        "num_chunks": len(vector_index) if vector_index else 0,
        "llm_ready": llm_client is not None,
        "embedding_ready": embedding_client is not None
    }


# expand_query_semantically is now imported from app.utils.query_expansion

# Helper functions for query processing
def detect_query_types(question: str) -> Dict[str, bool]:
    """Detect query types once and return as dictionary."""
    question_lower = question.lower()
    
    is_and_exclusion = any(term in question_lower for term in [
        '둘 모두 속하지 않는', '둘 다 속하지 않는', '모두 속하지 않는',
        '둘 모두 사용하지 않는', '둘 다 사용하지 않는', '모두 사용하지 않는',
        '와', '과'
    ]) and ('bms' in question_lower and 'gps' in question_lower)
    
    is_or_exclusion = any(term in question_lower for term in [
        '또는', '혹은', 'or'
    ]) and ('속하지' in question_lower or '포함되지' in question_lower)
    
    is_not_used_query = (
        any(keyword in question_lower for keyword in [
            '사용하지 않는', '사용되지 않는', '사용하지 않는', '미사용', 
            '사용 안', '사용안', '안 쓰는', '안쓰는', '쓰지 않는', '쓰지 않는'
        ]) 
        and not is_and_exclusion
        and not is_or_exclusion
        and ('속하지' not in question_lower and '포함되지' not in question_lower)
    )
    
    is_uncertain_query = any(keyword in question_lower for keyword in [
        '불분명', '불확실', '모호', '확실하지', '명확하지', '사용할지 불분명'
    ]) and ('사용' in question_lower or '비고' in question_lower)
    
    is_field_query = any(keyword in question_lower for keyword in ['변수', '필드', '사용', '미사용', '비고'])
    
    return {
        'is_and_exclusion': is_and_exclusion,
        'is_or_exclusion': is_or_exclusion,
        'is_not_used_query': is_not_used_query,
        'is_uncertain_query': is_uncertain_query,
        'is_field_query': is_field_query
    }


def is_not_used_passage(passage: str, passage_lower: str = None) -> Tuple[bool, bool, bool]:
    """
    Check if passage is a "not used" passage.
    Returns: (has_not_used, is_bms, has_excluded)
    """
    if passage_lower is None:
        passage_lower = passage.lower()
    
    # Check for "실제 사용하지 않음" keywords
    not_used_keywords = [
        '실제 사용하지 않음',
        '개발상이유로 존재',
        '개발상의 이유로 존재',
        '개발상 이유로 존재'
    ]
    has_not_used = any(keyword in passage_lower for keyword in not_used_keywords)
    
    # Check if BMS
    is_bms = (
        '테이블구분: bms' in passage_lower or 
        '테이블구분:bms' in passage_lower or 
        '변수명: seq' in passage_lower
    )
    
    # Check for excluded keywords (only exclude if doesn't have "실제 사용하지 않음")
    excluded_keywords = [
        '순수BMS데이터', '단말 처리상 정의 항목', 'DB상 정의 항목',
        'GPS데이터', '상세해석', '데이터 상세해석', 'GPS데이터 상세해석'
    ]
    has_excluded = any(keyword in passage for keyword in excluded_keywords) and '실제 사용하지 않음' not in passage_lower
    
    return has_not_used, is_bms, has_excluded


def find_not_used_passages(passages: List[str], exclude_indices: set = None, table_type: str = None) -> List[int]:
    """Find all passages with '실제 사용하지 않음' that match the specified table and are not excluded.
    
    Args:
        passages: List of passage strings
        exclude_indices: Set of indices to exclude
        table_type: "bms" or "gps" or None (None means both)
    """
    if exclude_indices is None:
        exclude_indices = set()
    
    not_used_indices = []
    for idx, passage in enumerate(passages):
        if idx in exclude_indices:
            continue
        
        passage_lower = passage.lower()
        has_not_used, is_bms, has_excluded = is_not_used_passage(passage, passage_lower)
        
        # 테이블 구분 확인
        is_gps = '테이블구분: gps' in passage_lower or '테이블구분:gps' in passage_lower or '테이블구분: gps_텔레매틱스' in passage_lower
        
        # 테이블 필터링
        table_match = False
        if table_type is None:
            # 테이블 타입이 지정되지 않으면 BMS만 (기존 동작 유지)
            table_match = is_bms
        elif table_type.lower() == "bms":
            table_match = is_bms
        elif table_type.lower() == "gps":
            table_match = is_gps
        
        if has_not_used and table_match and not has_excluded:
            not_used_indices.append(idx)
    
    return not_used_indices


def has_table_designation(passage: str, table: str) -> bool:
    """Check if passage contains table designation."""
    passage_lower = passage.lower()
    table_lower = table.lower()
    
    patterns = [
        f"테이블구분: {table}",
        f"테이블구분:{table}",
        f"테이블구분: {table_lower}"
    ]
    
    return any(pattern in passage or pattern.lower() in passage_lower for pattern in patterns)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    REFRAG-style RAG query endpoint with hybrid search.
    
    Implements Compress - Sense - Expand pipeline with domain-specific enhancements.
    """
    if vector_index is None or llm_client is None or embedding_client is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    if faiss_index is None or hybrid_retriever is None or reranker is None:
        raise HTTPException(status_code=503, detail="Hybrid search components not initialized")
    
    try:
        query_text = request.query
        top_k = min(request.top_k, Config.MAX_RETRIEVED_CHUNKS)
        
        # Use hybrid search like RAG server
        question_lower = query_text.lower()
        is_field_query = any(keyword in question_lower for keyword in ['변수', '필드', '사용', '미사용', '비고'])
        is_uncertain_usage_query = (
            any(keyword in question_lower for keyword in ['불확실', '확실하지 않은', '불명확'])
            and any(keyword in question_lower for keyword in ['사용', '변수'])
        )
        # For speed: limit top_k more aggressively (최대 공격적으로 줄임)
        if is_uncertain_usage_query:
            # 불확실 질문: 최소한만 검색 (3 → 2)
            top_k = min(2, len(passages))
        elif is_field_query:
            # 더 줄임: 4 → 3 chunks max for field queries
            top_k = min(3, len(passages))
        else:
            # 더 줄임: 3 → 2 chunks for general queries
            top_k = min(top_k, 2)
        
        # Detect query types early (before using them)
        # Note: is_uncertain_usage_query is already detected above for top_k optimization
        is_not_used_query = (
            any(keyword in question_lower for keyword in ['사용하지 않는', '사용되지 않는', '미사용'])
        )
        
        # Detect mileage/odometer related queries
        is_mileage_query = (
            any(keyword in question_lower for keyword in [
                '주행거리', 'odometer', '오도미터', 'mileage', '거리', 
                '전력', 'p_kw', 'pkw', 'pack_voltage', 'pack_current', 'emobility_spd',
                '세그먼트', '월별', '누적', '상태 분류', '충전', '방전'
            ])
            and (
                any(keyword in question_lower for keyword in ['bms', '계산', '산정', '로직', '규칙', '구하는', '공식', '어떻게', '방법'])
                or any(keyword in question_lower for keyword in ['mileage', '주행거리', 'odometer', '오도미터'])  # mileage 관련 키워드가 있으면 자동으로 포함
            )
        )
        
        # Query expansion (optimized for speed - reduced expansions)
        if is_uncertain_usage_query:
            # Skip expansion for uncertain usage queries (they already have column info)
            max_expansions = 0  # No expansion needed
        elif is_not_used_query:
            # Minimal expansion for "not used" queries
            max_expansions = 1  # Reduced from 2
        else:
            max_expansions = 1  # Reduced from 2 for faster processing
        
        # Query expansion: domain-specific + semantic + embedding-based
        domain_expansions = domain_dict.expand_query(query_text) if domain_dict else []
        semantic_expansions = expand_query_semantically(query_text, max_expansions=max_expansions)
        
        # Embedding-based expansion (disabled for speed optimization)
        # Skip embedding-based expansion to reduce latency
        embedding_expansions = []
        
        # Merge and deduplicate all expansions
        all_expansions = list(dict.fromkeys(
            domain_expansions + semantic_expansions + embedding_expansions
        ))[:Config.MAX_QUERY_EXPANSIONS]
        
        # Dense retrieval with caching
        query_embeddings_list = []
        expansions_to_embed = []
        expansion_indices = []
        
        # Check cache for each expansion
        for i, expansion in enumerate(all_expansions):
            cached = embedding_cache.get(expansion)
            if cached is not None:
                query_embeddings_list.append((i, cached))
            else:
                expansions_to_embed.append(expansion)
                expansion_indices.append(i)
        
        # Embed uncached expansions
        if expansions_to_embed:
            new_embeddings = embedding_model.encode(
                expansions_to_embed,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=len(expansions_to_embed)
            )
            faiss.normalize_L2(new_embeddings)
            
            # Store in cache and add to results
            for idx, expansion, embedding in zip(expansion_indices, expansions_to_embed, new_embeddings):
                embedding_cache.set(expansion, embedding)
                query_embeddings_list.append((idx, embedding))
        
        # Sort by original index and create array
        query_embeddings_list.sort(key=lambda x: x[0])
        query_embeddings = np.array([emb for _, emb in query_embeddings_list])
        
        # Average the embeddings for better semantic matching
        query_embedding = np.mean(query_embeddings, axis=0, keepdims=True)
        faiss.normalize_L2(query_embedding)
        
        # Phase 3: Apply TokenProjector if enabled (for better embedding matching)
        # Note: Ollama doesn't support inputs_embeds, so we use TokenProjector for embedding matching only
        if token_projector is not None and Config.USE_TOKEN_PROJECTOR:
            try:
                # Project encoder embeddings to decoder space for better matching
                # This is a simplified use case - full REFRAG would use inputs_embeds
                projected_embedding = token_projector.project_batch(query_embedding)
                # Normalize projected embedding
                faiss.normalize_L2(projected_embedding)
                # Use projected embedding for search (if dimension matches)
                # For now, we keep using original embedding since FAISS index uses encoder dimension
                # In full REFRAG, we would rebuild the index with projected embeddings
                logger.debug("TokenProjector applied to query embedding (for matching only)")
            except Exception as e:
                logger.warning(f"TokenProjector failed (non-critical): {e}. Using original embedding.")
        
        # Hybrid search (optimized for uncertain usage queries)
        # Acquire read lock for index access and create snapshot
        async with index_lock:
            if is_uncertain_usage_query:
                # Reduce search range for uncertain usage queries
                search_k = min(int(top_k * 1.5), len(passages))
            else:
                search_k = min(top_k * Config.SEARCH_K_MULTIPLIER, len(passages))
            distances, indices = faiss_index.search(query_embedding, search_k)
            
            # Create a snapshot of passages for this query (to avoid changes during processing)
            passages_snapshot = passages.copy()
            passages_lower_snapshot = passages_lower.copy() if passages_lower else [p.lower() for p in passages_snapshot]
        
        dense_results = [(int(indices[0][i]), float(distances[0][i])) for i in range(len(indices[0]))]
        
        # Use snapshot for hybrid search (no lock needed after snapshot)
        hybrid_results = hybrid_retriever.hybrid_search(
            query_text,
            dense_results,
            top_k=search_k,
            dense_weight=0.6,
            sparse_weight=0.3,
            exact_boost=1.5
        )
        
        # For "not used" queries, add passages with "실제 사용하지 않음"
        context_scores = [(idx, score, is_exact) for idx, score, is_exact in hybrid_results]
        if is_not_used_query and passages_snapshot:
            existing_indices = {idx for idx, _, _ in context_scores}
            
            # 테이블 타입 결정 (질문에서 추론)
            is_gps_query = 'gps' in question_lower and 'bms' not in question_lower
            is_bms_query = 'bms' in question_lower and 'gps' not in question_lower
            
            # Find "not used" passages (single scan, using pre-lowered text)
            # Use snapshot instead of original passages
            for idx, passage in enumerate(passages_snapshot):
                if idx in existing_indices:
                    continue

                passage_lower = passages_lower_snapshot[idx] if idx < len(passages_lower_snapshot) else passage.lower()
                has_not_used = any(kw in passage_lower for kw in [
                    '실제 사용하지 않음', '개발상이유로 존재', '개발상의 이유로 존재'
                ])
                
                # 테이블 구분 확인 (BMS 또는 GPS)
                is_bms = '테이블구분: bms' in passage_lower or '테이블구분:bms' in passage_lower
                is_gps = '테이블구분: gps' in passage_lower or '테이블구분:gps' in passage_lower or '테이블구분: gps_텔레매틱스' in passage_lower
                
                has_excluded = any(kw in passage for kw in [
                    '순수BMS데이터', '단말 처리상 정의 항목', 'DB상 정의 항목',
                    'GPS데이터', '상세해석'
                ]) and '실제 사용하지 않음' not in passage_lower

                # BMS 질문이면 BMS 변수만, GPS 질문이면 GPS 변수만 포함
                if has_not_used and not has_excluded:
                    if is_bms_query and is_bms:
                        context_scores.append((idx, 1.0, True))
                        existing_indices.add(idx)
                    elif is_gps_query and is_gps:
                        context_scores.append((idx, 1.0, True))
                        existing_indices.add(idx)
                    # 테이블이 명시되지 않은 경우 (일반 질문)는 기존 로직 유지
                    elif not is_bms_query and not is_gps_query and is_bms:
                        context_scores.append((idx, 1.0, True))
                        existing_indices.add(idx)
        
        # Filter and rerank
        # Sort: exact matches first, then by combined score
        context_scores.sort(key=lambda x: (not x[2], -x[1]), reverse=False)
        
        # Table filter for BMS (테이블 필터링 우선, is_exact는 테이블 일치 시에만 적용)
        if "bms" in question_lower and "gps" not in question_lower:
            filtered_scores = []
            for idx, score, is_exact in context_scores:
                # Use snapshot to avoid index changes during processing
                if idx < len(passages_snapshot):
                    passage = passages_snapshot[idx]
                    # 테이블 일치 확인 (is_exact는 테이블 일치 시에만 고려)
                    if "테이블구분: BMS" in passage or "테이블구분:BMS" in passage:
                        filtered_scores.append((idx, score, is_exact))
            context_scores = filtered_scores
        
        # Table filter for GPS (테이블 필터링 우선, is_exact는 테이블 일치 시에만 적용)
        elif "gps" in question_lower and "bms" not in question_lower:
            filtered_scores = []
            for idx, score, is_exact in context_scores:
                # Use snapshot to avoid index changes during processing
                if idx < len(passages_snapshot):
                    passage = passages_snapshot[idx]
                    # 테이블 일치 확인 (is_exact는 테이블 일치 시에만 고려)
                    if "테이블구분: GPS" in passage or "테이블구분:GPS" in passage or "테이블구분: GPS_텔레매틱스" in passage:
                        filtered_scores.append((idx, score, is_exact))
            context_scores = filtered_scores
        
        # Limit to top_k
        filtered_indices = [idx for idx, _, _ in context_scores[:top_k]]
        
        # Rerank (optimized - more aggressive limiting)
        if is_uncertain_usage_query:
            # Skip reranking for uncertain usage queries to save time
            # Column information in prompt is sufficient for these queries
            final_indices = filtered_indices[:top_k]
        elif Config.USE_RERANKER and len(filtered_indices) >= Config.RERANKER_MIN_CANDIDATES:
            # Optimized reranking: limit candidates more aggressively
            # Skip if too few candidates (reranking overhead not worth it)
            if len(filtered_indices) <= 3:
                # Too few candidates - skip reranking
                final_indices = filtered_indices[:top_k]
            else:
                # Limit reranking to max 10 candidates for better performance
                # This reduces reranking cost while maintaining quality
                rerank_candidates = min(len(filtered_indices), top_k, 10)
                
                # Use snapshot for reranking
                reranked_results = reranker.rerank(
                    query_text,
                    passages_snapshot,
                    filtered_indices[:rerank_candidates],
                    top_k=rerank_candidates
                )
                final_indices = [idx for idx, _ in reranked_results]
        else:
            final_indices = filtered_indices[:top_k]
        
        # 속도 관련 질문인 경우, 속도와 무관한 변수가 포함된 passage 제외
        is_speed_query_retrieval = any(keyword in question_lower for keyword in ['속도', 'speed', 'velocity', 'vel'])
        if is_speed_query_retrieval:
            filtered_final_indices = []
            for idx in final_indices:
                if idx < len(passages_snapshot):
                    passage = passages_snapshot[idx]
                    passage_lower = passage.lower()
                    
                    # 속도와 무관한 변수가 포함된 passage 제외
                    speed_unrelated_vars = [
                        'fastchargingportconnected', 'cumulativecurrentcharged', 
                        'cumulativecurrentdischarged', 'cumulativepowercharged',
                        'cumulativepowerdischarged'
                    ]
                    
                    has_unrelated_var = any(var in passage_lower for var in speed_unrelated_vars)
                    
                    # 속도 관련 변수가 있는지 확인
                    speed_related_vars = [
                        'emobility_spd', 'emobilityspeed', 'speed', 'velocity',
                        'drive_motor_spd', 'drivemotorspd'
                    ]
                    has_speed_var = any(var in passage_lower for var in speed_related_vars)
                    
                    # 속도 관련 변수가 있으면 항상 포함 (우선)
                    if has_speed_var:
                        filtered_final_indices.append(idx)
                        continue
                    
                    # 속도와 무관한 변수가 있고, 속도 관련 변수가 없으면 제외
                    if has_unrelated_var:
                        continue
                    
                    # cumulative로 시작하는 변수가 있고, 속도 관련 변수가 없으면 제외
                    if 'cumulative' in passage_lower:
                        continue
                    
                    # fastCharging, chargingPort, connected 관련 변수가 있고, 속도 관련 변수가 없으면 제외
                    if any(keyword in passage_lower for keyword in ['fastcharging', 'chargingport', 'connected']):
                        continue
                    
                    # 그 외의 경우는 포함 (속도 관련 변수가 없어도 다른 정보일 수 있음)
                
                filtered_final_indices.append(idx)
            
            final_indices = filtered_final_indices
        
        # Build a fast lookup map for scores: idx -> score
        score_lookup = {}
        for idx, score, _ in context_scores:
            # Keep the highest score per passage index if duplicated
            if idx not in score_lookup or score > score_lookup[idx]:
                score_lookup[idx] = score

        # Convert passage indices to RetrievedChunk objects
        # Note: convert_existing_index_to_refrag creates chunks in the same order as passages
        # So passage index == chunk index
        retrieved_chunks: list[RetrievedChunk] = []
        num_chunks = len(vector_index.chunks)
        for passage_idx in final_indices:
            if 0 <= passage_idx < num_chunks:
                chunk = vector_index.chunks[passage_idx]
                # Clamp score to [0.0, 1.0] range for Pydantic validation
                raw_score = float(score_lookup.get(passage_idx, 0.5))
                score = max(0.0, min(1.0, raw_score))
                retrieved_chunks.append(RetrievedChunk(chunk=chunk, score=score))
        
        # Check if this is a distance-related or p_kw calculation query that can be answered from column info
        is_distance_query_only = any(keyword in question_lower for keyword in ['거리', 'distance', 'mileage', '주행거리', 'odometer', '오도미터'])
        is_p_kw_query = any(keyword in question_lower for keyword in ['p_kw', 'pkw', '전력.*계산', '전력.*구하는', '전력.*공식'])
        if not retrieved_chunks:
            # For distance-related, battery-related, mileage, and p_kw calculation queries, try to answer from column info
            is_battery_query_only = any(keyword in question_lower for keyword in ['배터리', 'battery', 'batt', '셀', 'cell', '팩', 'pack', '모듈', 'module', 'soc', 'soh', 'socd'])
            if is_distance_query_only or is_battery_query_only or is_p_kw_query or is_mileage_query:
                # Don't return immediately - let LLM answer from column info
                logger.info(f"No retrieved chunks for {'distance' if is_distance_query_only else 'battery'} query, will use column info")
            else:
                return QueryResponse(
                    answer="문서에서 해당 내용을 찾을 수 없습니다.",
                    used_chunks=[],
                    compression_decisions={},
                    prompt_token_count=0,
                    llm_latency_ms=0.0
                )
        
        # Phase 2.3: 청크 수 동적 제한 (REFRAG_SPEED_OPTIMIZATION_PLAN.md)
        # 프롬프트 토큰 수 예측
        estimated_tokens = estimate_prompt_tokens(query_text, retrieved_chunks)
        
        if estimated_tokens > 1500:
            # 토큰 수가 많으면 청크 수 줄이기 (더 공격적으로)
            original_count = len(retrieved_chunks)
            retrieved_chunks = retrieved_chunks[:min(3, len(retrieved_chunks))]
            logger.info(f"Reduced chunks from {original_count} to {len(retrieved_chunks)} due to high estimated token count ({estimated_tokens})")
        elif estimated_tokens > 1000:
            # 중간 토큰 수면 약간 줄이기
            original_count = len(retrieved_chunks)
            retrieved_chunks = retrieved_chunks[:min(4, len(retrieved_chunks))]
            logger.info(f"Reduced chunks from {original_count} to {len(retrieved_chunks)} due to moderate estimated token count ({estimated_tokens})")
        
        # Use REFRAG pipeline for compression/expansion and generation
        # Dynamic compression based on query type (REFRAG optimization)
        from app.compression.policy import HeuristicCompressionPolicy
        
        if is_uncertain_usage_query:
            # Uncertain usage queries: most aggressive compression (최대 공격적)
            # 모든 청크를 압축 (확장 비율 0% 보장)
            # 불확실 질문은 컬럼 정보만으로도 충분하므로 청크 확장 불필요
            sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.score, reverse=True)
            compression_decisions = {chunk.chunk.id: "COMPRESS" for chunk in sorted_chunks}
            logger.info(f"Uncertain query: all {len(compression_decisions)} chunks compressed (0% expansion)")
        elif is_not_used_query:
            # "Not used" queries: 더 공격적 압축
            temp_policy = HeuristicCompressionPolicy(
                max_expanded_chunks=1,  # 2 → 1 (더 줄임)
                expand_frac=0.15,      # 0.2 → 0.15 (15%만 확장)
                score_threshold=0.75    # 0.7 → 0.75 (더 높은 기준)
            )
            compression_decisions = temp_policy.decide(retrieved_chunks)
        else:
            # General queries: 더 공격적 압축
            temp_policy = HeuristicCompressionPolicy(
                max_expanded_chunks=1,  # 2 → 1 (더 줄임)
                expand_frac=0.15,       # 0.2 → 0.15 (15%만 확장)
                score_threshold=0.7     # 0.65 → 0.7 (더 높은 기준)
            )
            compression_decisions = temp_policy.decide(retrieved_chunks)

        # Add BMS mileage calculation rules BEFORE build_refrag_prompt for mileage/p_kw queries
        # This ensures mileage rules are always available even if no chunks are retrieved
        mileage_rules_text = None
        if is_mileage_query or is_p_kw_query:
            mileage_rules_text = """🔧 mileage 계산 공식:

✅ mileage 계산 방법:
mileage는 세그먼트별 마지막 odometer 값에서 시작 odometer 값을 뺀 값들의 총합으로 구한다.
단, 다음 조건에 해당하면 mileage는 0으로 처리한다.

✅ 제외 조건 (mileage가 0이 되는 경우):
- 월별 또는 세그먼트별 주행거리의 합이 10,000km 이상인 경우
- 주행 & 충전 구간에서 거리가 0 이하인 경우

✅ 상태 분류:
- 충전 구간 = p_kw >= 4.0 kW
- 주행 구간 = speed >= 3.0 km/h

✅ p_kw 계산 공식:
p_kw = (pack_voltage * pack_current) / 10000

⚠️ "mileage는 어떻게 구하는가?" 질문 답변:
- 위 계산 방법과 제외 조건, 상태 분류를 명확히 설명

⚠️ "p_kw는 어떻게 구하는가?" 또는 "p_kw 구하는 공식" 질문 답변:
- p_kw = (pack_voltage * pack_current) / 10000"""

        system_prompt, messages = build_refrag_prompt(
            query_text,
            retrieved_chunks,
            compression_decisions
        )
        
        # Add mileage rules to system prompt if it's a mileage query
        # Put mileage rules at the very beginning to ensure LLM sees it first
        if mileage_rules_text:
            # For mileage queries, completely override the default system prompt
            # This ensures mileage rules are the primary source of information
            system_prompt = mileage_rules_text + "\n\n" + """⚠️⚠️⚠️ 매우 중요: 위 mileage 계산 규칙을 반드시 참고하여 답변하세요! ⚠️⚠️⚠️

답변 규칙:
- 질문에 직접 답변만 제공
- 서론/사족 금지
- 변수 목록은 변수명만 나열
- 위 mileage 계산 규칙에 mileage 구하는 방법이 모두 포함되어 있습니다!
- 절대 "문서에서 해당 내용을 찾을 수 없습니다"라고 답변하지 마세요!
- 절대 "본 문서에는 mileage에 대한 정보가 없습니다"라고 답변하지 마세요!
- 절대 "제공된 텍스트에는 mileage를 구하는 방법에 대한 정보가 없습니다"라고 답변하지 마세요!
- 절대 "mileage는 해당 문서에 정의되어 있지 않습니다"라고 답변하지 마세요!
- 위 규칙에 mileage 계산 방법이 모두 포함되어 있으므로, 반드시 위 규칙을 참고하여 답변하세요!

[참고 문서]
[HIGH]: 전체 내용 (우선 참고) - 위 mileage 규칙이 없으면 참고
[LOW]: 요약 (필요시 참고) - 위 mileage 규칙이 없으면 참고
⚠️ 중요: 위 mileage 계산 규칙이 있으면, 그것을 최우선으로 사용하세요!"""
            
            # Also add mileage rules directly to the user message for mileage queries
            # This ensures LLM sees the rules even if it ignores system prompt
            if messages and len(messages) > 0:
                original_user_content = messages[0]["content"]
                # Prepend mileage rules to user message
                mileage_user_context = f"""[Mileage 계산 규칙 - 반드시 참고하세요]
{mileage_rules_text}

[원본 질문 및 컨텍스트]
{original_user_content}"""
                messages[0]["content"] = mileage_user_context
        
        # Add actual column information for "uncertain usage", "not used", "distance-related", "speed-related", and "battery-related" queries
        is_distance_query_for_columns = any(keyword in question_lower for keyword in ['거리', 'distance', 'mileage', '주행거리', 'odometer', '오도미터'])
        is_speed_query_for_columns = any(keyword in question_lower for keyword in ['속도', 'speed', 'velocity', 'vel'])
        is_battery_query_for_columns = any(keyword in question_lower for keyword in ['배터리', 'battery', 'batt', '셀', 'cell', '팩', 'pack', '모듈', 'module', 'soc', 'soh', 'socd'])
        if is_uncertain_usage_query or is_not_used_query or is_distance_query_for_columns or is_speed_query_for_columns or is_battery_query_for_columns:
            # Determine table type from query
            if "gps" in question_lower and "bms" not in question_lower:
                table_type = "gps"
            elif "bms" in question_lower:
                table_type = "bms"
            elif (is_distance_query_for_columns or is_speed_query_for_columns or is_battery_query_for_columns) and "bms" not in question_lower and "gps" not in question_lower:
                # For distance/speed/battery queries without table specification, use BMS (battery is BMS-specific)
                table_type = "bms"  # Default for battery queries
            else:
                table_type = "bms"
            
            # Get actual columns from preprocessed files (includes specification files)
            file_columns = get_actual_used_columns(table_type)
            if file_columns:
                columns_info = format_columns_for_prompt(file_columns)
                # Add to system prompt
                system_prompt += f"\n\n{columns_info}\n\n"
                
                # For battery queries, explicitly list all battery-related variables (excluding individual cell/module vars)
                if is_battery_query_for_columns and table_type == "bms":
                    columns = file_columns.get("columns", set()) if isinstance(file_columns, dict) else file_columns
                    battery_keywords = ['soc', 'socd', 'soh', 'cell', 'volt', 'pack', 'module', 'batt', 'temp', 'insul', 'deter']
                    
                    # 개별 셀/모듈 변수 제외하고 배터리 관련 변수만 추출
                    battery_vars = []
                    for col in sorted(columns):
                        col_lower = col.lower()
                        # 개별 셀 전압 변수 제외 (cell_v_001, cell_v_002 등)
                        if col_lower.startswith('cell_v_') and col_lower[7:].isdigit():
                            continue
                        # 개별 모듈 온도 변수 제외 (mod_temp_01, mod_temp_02 등)
                        if col_lower.startswith('mod_temp_') and (col_lower[9:].isdigit() or len(col_lower[9:]) == 2):
                            continue
                        # 배터리 관련 키워드가 있는 변수만 포함
                        if any(kw in col_lower for kw in battery_keywords):
                            battery_vars.append(col)
                    
                    if battery_vars:
                        # 핵심 변수 우선 정렬
                        priority_vars = []
                        other_vars = []
                        for v in battery_vars:
                            v_lower = v.lower()
                            if any(priority in v_lower for priority in ['soc', 'soh', 'pack_volt', 'pack_current', 'cell_volt_list', 'mod_temp_list', 'batt_internal', 'modulemax', 'modulemin', 'moduleavg', 'max_cell_volt', 'min_cell_volt', 'insul_resistance', 'deter']):
                                priority_vars.append(v)
                            else:
                                other_vars.append(v)
                        
                        sorted_battery_vars = sorted(priority_vars) + sorted(other_vars)
                        battery_vars_str = ', '.join(sorted_battery_vars[:30])  # 최대 30개까지
                        if len(sorted_battery_vars) > 30:
                            battery_vars_str += f", ... 외 {len(sorted_battery_vars) - 30}개"
                        
                        system_prompt += f"""\n\n🔋 배터리 관련 변수 목록 (BMS 테이블):
다음은 BMS 테이블에 있는 배터리 관련 변수들입니다. "배터리와 관련된 변수는?" 질문에 반드시 이 변수들을 포함하세요:

{battery_vars_str}

⚠️ 매우 중요: 
- 위 변수들은 모두 배터리와 관련된 변수입니다. 답변에 반드시 포함하세요!
- packVoltage만 나열하는 것은 절대 금지! 위 목록의 변수들을 모두 나열하세요!
- 개별 셀 전압 변수(cell_v_001 등)나 개별 모듈 온도 변수(mod_temp_01 등)는 제외하고, 대표 변수(cell_volt_list, mod_temp_list 등)만 포함하세요!\n\n"""
            
            # For distance/speed queries without table specification, also add GPS columns
            if (is_distance_query_for_columns or is_speed_query_for_columns) and "bms" not in question_lower and "gps" not in question_lower:
                gps_columns = get_actual_used_columns("gps")
                if gps_columns:
                    gps_columns_info = format_columns_for_prompt(gps_columns)
                    system_prompt += f"\n\n{gps_columns_info}\n\n"
                
                # Add missing variables info (variables in spec but not in rules or CSV)
                # Use cached domain_dict to avoid rebuilding
                missing_info = format_missing_variables_info(table_type, domain_dict=domain_dict)
                if missing_info:
                    system_prompt += f"\n{missing_info}\n\n"
                
                # Add unused variables info (variables in rules Excel but not in specification files)
                # This is the key information for "사용하지 않는 변수" queries
                unused_info = format_unused_variables_info(table_type, domain_dict=domain_dict)
                if unused_info:
                    system_prompt += f"\n{unused_info}\n\n"
                
                # 변수 관련 규칙 및 답변 형식 (최적화: 더 간결하고 엄격하게)
                if is_not_used_query:
                    system_prompt += """⚠️⚠️⚠️ "사용하지 않는 변수" 질문 답변 형식 (최우선) ⚠️⚠️⚠️:

1. 답변은 반드시 다음 형식으로 작성:
   "BMS 테이블에 존재하지만 사용하지 않는 변수: seq"
   
   또는 변수가 여러 개인 경우:
   "BMS 테이블에 존재하지만 사용하지 않는 변수: seq, 변수2, 변수3"
   
2. ⚠️ 절대 규칙:
   - "비고:" 필드에 "실제 사용하지 않음" 또는 "개발상이유로 존재"가 정확히 포함된 변수만 나열
   - "순수BMS데이터", "단말 처리상 정의 항목", "DB상 정의 항목" 비고가 있는 변수는 절대 포함하지 말 것
   - 변수명만 간결하게 나열 (설명, 비고 내용 불필요)
   - 서론, 사족, 추측성 표현 절대 금지
   
3. 올바른 답변 예시:
   "BMS 테이블에 존재하지만 사용하지 않는 변수: seq"
   
4. 잘못된 답변 예시 (절대 하지 말 것):
   "acceptableChargingPower, acceptableDischargingPower, batteryPower, chargeCount..." (X - 이들은 사용되는 변수임!)
   "제공된 정보에서..." (X - 서론 금지!)
   "다음은 잠재적으로..." (X - 추측 금지!)

변수 규칙:
- 제외: hvacList1/hvac_list1, moduleAvgTemp/mod_avg_temp, cellVoltageList/cell_volt_list, deviceNo/device_no, messageTime/msg_time
- 매핑: camelCase↔snake_case 동일, 대소문자 무시
- 기준: 규격파일 있음→사용됨, 금지목록 있음→사용됨, rules만 있고 "실제 사용하지 않음"→사용안됨
- 필터: GPS 질문→GPS 변수만, BMS 질문→BMS 변수만
- GPS: "실제 사용하지 않음" 없으면 "GPS 테이블의 모든 변수는 사용됨" 명확히 답변

⚠️⚠️⚠️ 절대 포함 금지 변수 목록 (BMS 테이블) ⚠️⚠️⚠️:
다음 변수들은 모두 "순수BMS데이터" 비고를 가지고 있거나 규격 파일에 존재하므로 절대 "사용하지 않는 변수" 목록에 포함하지 말 것!
- moduleMinTemp, moduleMaxTemp, moduleAvgTemp, moduleTempList (모듈 온도 관련 - 모두 사용됨)
- maxCellVoltage, maxCellVoltageNo, minCellVoltage, minCellVoltageNo (셀 전압 관련 - 모두 사용됨)
- battInternalTemp, battFanRunning, battPw (배터리 관련 - 모두 사용됨)
- cellVoltageList, cellVoltageDispersion (셀 전압 관련 - 모두 사용됨)
- subBattVoltage (보조배터리 전압 - 사용됨)
- cumulativeCurrentCharged, cumulativeCurrentDischarged (누적 전류 - 사용됨)
- cumulativePowerCharged, cumulativePowerDischarged (누적 전력 - 사용됨)
- soh (State of Health - 사용됨)
- maxDeteriorationCellNo, minDeteriorationCellNo (열화 관련 - 사용됨)
- 모든 "순수BMS데이터" 비고가 있는 변수 (이것은 "사용됨"을 의미함)
- 규격 파일(bms_specification.csv)에 있는 모든 변수 (이것은 "사용됨"을 의미함)

⚠️ 매우 중요: 위 변수들을 포함하면 답변이 완전히 잘못된 것입니다!
⚠️ 올바른 답변은 오직 "seq"만 포함해야 합니다!"""
                else:
                    # "거리와 관련된 변수" 질문 처리
                    is_distance_query_prompt = any(keyword in question_lower for keyword in ['거리', 'distance', 'mileage', '주행거리', 'odometer', '오도미터'])
                    is_speed_query_prompt = any(keyword in question_lower for keyword in ['속도', 'speed', 'velocity', 'vel'])
                    is_battery_query_prompt = any(keyword in question_lower for keyword in ['배터리', 'battery', 'batt', '셀', 'cell', '팩', 'pack', '모듈', 'module', 'soc', 'soh', 'socd'])
                    if is_distance_query_prompt:
                        system_prompt += """⚠️⚠️⚠️ "거리와 관련된 변수" 질문 답변 규칙 ⚠️⚠️⚠️:

✅ 포함해야 할 변수 (BMS 테이블에 실제 존재하는 컬럼):
   - odometer (누적 주행거리, km) - 직접 거리 변수
   - pack_volt (팩 전압, V) - 거리 계산에 사용 (p_kw 계산용)
   - pack_current (팩 전류, A) - 거리 계산에 사용 (p_kw 계산용)
   - emobility_spd (속도, km/h) - 거리 계산에 사용

✅ GPS 테이블의 거리 관련 변수 (테이블이 명시되지 않았을 경우):
   - lat, lon (위도, 경도) - 위치 정보
   - speed (속도) - 거리 계산에 사용

❌ 절대 포함하지 말 것:
   - p_kw (BMS 테이블에 실제 컬럼이 아님! 계산된 값임!)
   - cumulativeCurrentCharged, cumulativeCurrentDischarged (누적 전류량 - 거리와 무관)
   - cumulativePowerCharged, cumulativePowerDischarged (누적 전력량 - 거리와 무관)

   ⚠️ 매우 중요: 
   - "거리와 관련된 변수" 질문에 p_kw를 포함하면 완전히 잘못된 답변입니다!
   - 규격 파일(bms_specification.csv, gps_specification.csv)에 있는 변수만 포함하세요!
   - 절대 "문서에서 해당 내용을 찾을 수 없습니다"라고 답변하지 말고, 위 변수들을 나열하세요!"""
                    elif is_speed_query_prompt:
                        system_prompt += """⚠️⚠️⚠️ "속도와 관련된 변수" 질문 답변 규칙 ⚠️⚠️⚠️:

✅ 포함해야 할 변수 (실제 속도 변수만):
   - BMS 테이블: emobility_spd (속도, km/h)
   - GPS 테이블: speed (속도)
   - 구동 모터 속도: drive_motor_spd1, drive_motor_spd2 (BMS 테이블)

❌ 절대 포함하지 말 것:
   - fastChargingPortConnected, chargingPortConnected (충전 포트 연결 상태 - 속도와 무관)
   - charging, charge, 충전 관련 변수 (속도와 무관)
   - port, 포트, connected, 연결 관련 변수 (속도와 무관)
   - relay, 릴레이, cable, 케이블 관련 변수 (속도와 무관)
   - temp, 온도, voltage, 전압, current, 전류, power, 전력 관련 변수 (속도와 무관)
   - cell, 셀, module, 모듈, battery, 배터리 관련 변수 (속도와 무관)

⚠️ 매우 중요: 
   - "속도와 관련된 변수" 질문에는 실제 속도 변수만 포함하세요!
   - 충전, 포트, 연결, 전압, 전류, 전력 등과 관련된 변수는 절대 포함하지 마세요!
   - 규격 파일(bms_specification.csv, gps_specification.csv)에 있는 속도 변수만 포함하세요!
   - 절대 "문서에서 해당 내용을 찾을 수 없습니다"라고 답변하지 말고, 위 속도 변수들을 나열하세요!"""
                    elif is_battery_query_prompt:
                        system_prompt += """⚠️⚠️⚠️ "배터리와 관련된 변수" 질문 답변 규칙 ⚠️⚠️⚠️:

✅ 포함해야 할 변수 (배터리 관련 변수):
   - SOC/충전 상태: soc (SOC율), socd (SOC디스플레이율), soh (SOH율)
   - 셀 전압: cellVoltageList (셀 전압 List), cellVoltageDispersion (셀 전압 분산), 
             maxCellVoltage (최대 셀 전압), maxCellVoltageNo (최대 전압 셀 번호),
             minCellVoltage (최소 셀 전압), minCellVoltageNo (최소 전압 셀 번호)
   - 팩 전압/전류: packVoltage (팩 전압), packCurrent (팩 전류), pack_volt, pack_current
   - 모듈 온도: moduleMaxTemp (모듈최대온도), moduleMinTemp (모듈최소온도), 
              moduleTempList (모듈 온도 List), moduleAvgTemp (모듈 평균 온도)
   - 배터리 온도: battInternalTemp (배터리 내부 온도), battFanRunning (배터리 팬 상태)
   - 보조배터리: subBattVoltage (보조배터리 전압)
   - 절연 저항: insulatedResistance (절연저항), insul_resistance
   - 열화 관련: maxDeteriorationCellNo (최대 열화 Cell 번호), minDeteriorationCellNo (최소 열화 Cell 번호)
   - 기타: cell_volt_list, cell_volt_dispersion, max_cell_volt, min_cell_volt 등

⚠️ 매우 중요: 
   - "배터리와 관련된 변수" 질문에는 배터리, 셀, 팩, 모듈, SOC, SOH, 전압, 전류, 온도, 열화 관련 변수를 모두 포함하세요!
   - 규격 파일(bms_specification.csv)에 있는 배터리 관련 변수들을 모두 나열하세요!
   - 절대 "문서에서 해당 내용을 찾을 수 없습니다"라고 답변하지 말고, 위 배터리 관련 변수들을 나열하세요!
   - 답변에는 최소 10개 이상의 배터리 관련 변수를 포함해야 합니다!
   - packVoltage만 나열하는 것은 절대 금지! soc, socd, soh, cellVoltageList, cellVoltageDispersion, maxCellVoltage, minCellVoltage, moduleMaxTemp, moduleMinTemp, moduleTempList, battInternalTemp, subBattVoltage, insulatedResistance 등 모든 배터리 관련 변수를 포함하세요!"""
                    else:
                        system_prompt += """변수 규칙:
- 제외: hvacList1/hvac_list1, moduleAvgTemp/mod_avg_temp, cellVoltageList/cell_volt_list, deviceNo/device_no, messageTime/msg_time
- 매핑: camelCase↔snake_case 동일, 대소문자 무시
- 기준: 규격파일 있음→사용됨, 금지목록 있음→사용됨, rules만 있고 "실제 사용하지 않음"→사용안됨
- 필터: GPS 질문→GPS 변수만, BMS 질문→BMS 변수만
- GPS: "실제 사용하지 않음" 없으면 "GPS 테이블의 모든 변수는 사용됨" 명확히 답변"""
        
        # Note: mileage_rules_text is already added to system_prompt before build_refrag_prompt
        # No need to add again here
        
        full_prompt_text = system_prompt + "\n\n" + messages[0]["content"]
        prompt_token_count = count_tokens(full_prompt_text)
        
        # Phase 2.1: 토큰 수 기반 동적 조절 (REFRAG_SPEED_OPTIMIZATION_PLAN.md)
        # 최적화: 프롬프트 재빌드 오버헤드를 줄이기 위해 컬럼 정보를 변수에 저장
        columns_info_cached = None
        missing_info_cached = None
        unused_info_cached = None
        table_type_cached = None
        
        if is_uncertain_usage_query or is_not_used_query:
            table_type_cached = "bms" if "bms" in question_lower and "gps" not in question_lower else "gps" if "gps" in question_lower else "bms"
            file_columns = get_actual_used_columns(table_type_cached)
            if file_columns:
                columns_info_cached = format_columns_for_prompt(file_columns)
                missing_info_cached = format_missing_variables_info(table_type_cached, domain_dict=domain_dict)
                unused_info_cached = format_unused_variables_info(table_type_cached, domain_dict=domain_dict)
        
        # 재빌드 조건을 더 보수적으로 설정 (4000+ 토큰에서만 재빌드)
        # 초기 압축 정책이 이미 공격적이므로 재빌드는 매우 높은 경우만
        # 일반 질문의 경우 재빌드하지 않음 (컬럼 정보가 없어서 재빌드 효과가 제한적)
        if prompt_token_count > 4000 and (is_uncertain_usage_query or is_not_used_query):
            # 토큰 수가 매우 많고, 컬럼 정보가 있는 질문에서만 재빌드 (안전장치)
            logger.info(f"Very high token count ({prompt_token_count}), applying aggressive compression")
            compression_decisions = temp_policy.apply_aggressive_compression(
                retrieved_chunks, compression_decisions
            )
            # 재빌드 프롬프트 (캐시된 컬럼 정보 재사용)
            system_prompt, messages = build_refrag_prompt(
                query_text,
                retrieved_chunks,
                compression_decisions
            )
            # 컬럼 정보 다시 추가 (캐시 사용)
            if columns_info_cached:
                system_prompt += f"\n\n{columns_info_cached}\n\n"
                if missing_info_cached:
                    system_prompt += f"\n{missing_info_cached}\n\n"
                if unused_info_cached:
                    system_prompt += f"\n{unused_info_cached}\n\n"
                system_prompt += """변수 규칙:
- 제외: hvacList1/hvac_list1, moduleAvgTemp/mod_avg_temp, cellVoltageList/cell_volt_list, deviceNo/device_no, messageTime/msg_time
- 매핑: camelCase↔snake_case 동일, 대소문자 무시
- 기준: 규격파일 있음→사용됨, 금지목록 있음→사용됨, rules만 있고 "실제 사용하지 않음"→사용안됨
- 필터: GPS 질문→GPS 변수만, BMS 질문→BMS 변수만
- GPS: "실제 사용하지 않음" 없으면 "GPS 테이블의 모든 변수는 사용됨" 명확히 답변"""
            full_prompt_text = system_prompt + "\n\n" + messages[0]["content"]
            prompt_token_count = count_tokens(full_prompt_text)
            logger.info(f"After aggressive compression: {prompt_token_count} tokens")
        elif prompt_token_count > 3500 and (is_uncertain_usage_query or is_not_used_query):
            # 중간-높은 토큰 수: 중간 압축 (컬럼 정보가 있는 질문에서만)
            logger.info(f"High token count ({prompt_token_count}), applying moderate compression")
            compression_decisions = temp_policy.apply_moderate_compression(
                retrieved_chunks, compression_decisions
            )
            # 재빌드 프롬프트 (캐시된 컬럼 정보 재사용)
            system_prompt, messages = build_refrag_prompt(
                query_text,
                retrieved_chunks,
                compression_decisions
            )
            # 컬럼 정보 다시 추가 (캐시 사용)
            if columns_info_cached:
                system_prompt += f"\n\n{columns_info_cached}\n\n"
                if missing_info_cached:
                    system_prompt += f"\n{missing_info_cached}\n\n"
                if unused_info_cached:
                    system_prompt += f"\n{unused_info_cached}\n\n"
                system_prompt += """변수 규칙:
- 제외: hvacList1/hvac_list1, moduleAvgTemp/mod_avg_temp, cellVoltageList/cell_volt_list, deviceNo/device_no, messageTime/msg_time
- 매핑: camelCase↔snake_case 동일, 대소문자 무시
- 기준: 규격파일 있음→사용됨, 금지목록 있음→사용됨, rules만 있고 "실제 사용하지 않음"→사용안됨
- 필터: GPS 질문→GPS 변수만, BMS 질문→BMS 변수만
- GPS: "실제 사용하지 않음" 없으면 "GPS 테이블의 모든 변수는 사용됨" 명확히 답변"""
            full_prompt_text = system_prompt + "\n\n" + messages[0]["content"]
            prompt_token_count = count_tokens(full_prompt_text)
            logger.info(f"After moderate compression: {prompt_token_count} tokens")
        
        # Optimize max_tokens for faster generation (최대 공격적으로 줄임)
        if is_mileage_query or is_p_kw_query:
            max_tokens = 300  # mileage/p_kw 계산 방법 설명에는 더 많은 토큰 필요
        elif is_uncertain_usage_query:
            max_tokens = 60  # 80 → 60 (더 줄임)
        elif is_not_used_query:
            max_tokens = 50  # 100 → 50 (간결한 답변 유도: "BMS 테이블에 존재하지만 사용하지 않는 변수: seq")
        else:
            max_tokens = 120  # 150 → 120 (더 줄임)
        
        # Phase 3: Use TokenProjector to generate inputs_embeds if enabled
        inputs_embeds = None
        if token_projector is not None and Config.USE_TOKEN_PROJECTOR and Config.LLM_BACKEND == "huggingface":
            try:
                # Build full prompt text
                full_prompt_text = system_prompt + "\n\n" + messages[0]["content"]
                
                # Embed the prompt using encoder (sentence-transformers)
                prompt_embedding = embedding_model.encode(
                    full_prompt_text,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                # Project to decoder space using TokenProjector
                # prompt_embedding shape: [encoder_dim] (1D)
                # Reshape to [1, encoder_dim] for batch processing
                prompt_embedding_2d = prompt_embedding.reshape(1, -1)
                projected_embedding = token_projector.project_batch(prompt_embedding_2d)
                
                # projected_embedding shape: [1, decoder_dim] (batch_size=1, single embedding)
                # Reshape for sequence: [batch_size, seq_len, embedding_dim]
                # For now, treat the entire prompt as a single token (simplified)
                # In full REFRAG, we would split into multiple tokens
                # Add sequence dimension: [1, decoder_dim] -> [1, 1, decoder_dim]
                if projected_embedding.ndim == 2:
                    # [batch_size, decoder_dim] -> [batch_size, 1, decoder_dim]
                    inputs_embeds = projected_embedding.reshape(1, 1, -1)
                else:
                    inputs_embeds = projected_embedding
                
                logger.debug(
                    f"Generated inputs_embeds using TokenProjector: "
                    f"original shape={prompt_embedding.shape}, "
                    f"projected shape={projected_embedding.shape}, "
                    f"final shape={inputs_embeds.shape}"
                )
            except Exception as e:
                logger.warning(f"TokenProjector failed to generate inputs_embeds: {e}. Using tokenized input.")
                inputs_embeds = None
        
        answer_coro = llm_client.chat(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
            inputs_embeds=inputs_embeds
        )
        
        answer, llm_latency_ms = await measure_llm_latency(answer_coro)
        
        # Re-retrieval disabled for speed optimization
        # Skip re-retrieval to reduce latency (original retrieval is sufficient)
        original_scores = [rc.score for rc in retrieved_chunks]
        if False and should_reretrieve(answer, original_scores) and Config.USE_RERETRIEVAL:
            try:
                logger.debug("Performing optimized re-retrieval")
                
                # Extract keywords from answer (no LLM call - fast)
                keywords = extract_keywords_from_answer(answer)
                
                if keywords:
                    # Build re-retrieval query
                    reretrieval_query = build_reretrieval_query(query_text, keywords)
                    
                    # Small-scale re-retrieval (limited range for performance)
                    reretrieval_k = min(Config.RERETRIEVAL_SEARCH_K, len(passages))
                    
                    # Quick embedding for re-retrieval query (with caching)
                    cached_reretrieval = embedding_cache.get(reretrieval_query)
                    if cached_reretrieval is not None:
                        reretrieval_embedding = cached_reretrieval.reshape(1, -1)
                    else:
                        reretrieval_embedding = embedding_model.encode(
                            reretrieval_query,
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )
                        faiss.normalize_L2(reretrieval_embedding)
                        embedding_cache.set(reretrieval_query, reretrieval_embedding.flatten())
                        reretrieval_embedding = reretrieval_embedding.reshape(1, -1)
                    
                    # Acquire read lock for re-retrieval
                    async with index_lock:
                        reretrieval_distances, reretrieval_indices = faiss_index.search(
                            reretrieval_embedding, reretrieval_k
                        )
                        passages_snapshot_reretrieval = passages.copy()
                    
                    # Get re-retrieval results
                    reretrieval_results = [
                        (int(reretrieval_indices[0][i]), float(reretrieval_distances[0][i]))
                        for i in range(len(reretrieval_indices[0]))
                    ]
                    
                    # Merge with original results (avoid duplicates, prioritize original)
                    existing_indices = {idx for idx, _, _ in context_scores}
                    reretrieval_chunks_to_add = []
                    
                    for idx, score in reretrieval_results:
                        if idx not in existing_indices and 0 <= idx < len(vector_index.chunks):
                            # Only add if score is reasonable
                            if score > 0.3:  # Threshold to avoid low-quality results
                                chunk = vector_index.chunks[idx]
                                reretrieval_chunks_to_add.append(
                                    RetrievedChunk(chunk=chunk, score=min(score, 0.8))  # Cap score to avoid over-prioritizing
                                )
                    
                    # Add re-retrieval chunks to retrieved_chunks (limit to avoid context bloat)
                    if reretrieval_chunks_to_add:
                        max_reretrieval_chunks = min(2, len(reretrieval_chunks_to_add))  # Limit to 2 chunks
                        retrieved_chunks.extend(reretrieval_chunks_to_add[:max_reretrieval_chunks])
                        logger.debug(f"Added {len(reretrieval_chunks_to_add[:max_reretrieval_chunks])} chunks from re-retrieval")
                        
                        # Re-run compression policy with updated chunks
                        compression_decisions = compression_policy.decide(retrieved_chunks)
                        
                        # Rebuild prompt with updated chunks (only if significant new content)
                        if len(reretrieval_chunks_to_add) > 0:
                            system_prompt, messages = build_refrag_prompt(
                                query_text,
                                retrieved_chunks,
                                compression_decisions
                            )
                            # Note: We don't re-run LLM here to avoid performance impact
                            # Re-retrieval chunks are added to context for potential future queries
            except Exception as e:
                # Fail silently - re-retrieval is optional
                logger.debug(f"Re-retrieval failed (non-critical): {e}")
        
        # 답변 후처리: 서론/사족 제거 및 유사도 기반 필터링
        # 테이블 타입 결정
        question_lower = query_text.lower()
        table_type = "bms" if "bms" in question_lower and "gps" not in question_lower else "gps" if "gps" in question_lower else "bms"
        answer = postprocess_answer(answer, table_type=table_type, is_not_used_query=is_not_used_query, domain_dict=domain_dict, question=query_text)
        
        used_chunks = [rc.chunk for rc in retrieved_chunks]
        
        return QueryResponse(
            answer=answer,
            used_chunks=used_chunks,
            compression_decisions=compression_decisions,
            prompt_token_count=prompt_token_count,
            llm_latency_ms=llm_latency_ms
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Query validation error: {error_msg}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {error_msg}")
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.exception(f"Query processing error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {error_msg}")


@app.post("/ingest")
async def ingest(request: IngestRequest):
    """Ingest new text into the index."""
    try:
        if faiss_index is None or embedding_model is None:
            logger.warning("Ingest request received but index not loaded")
            raise HTTPException(status_code=503, detail="Index not loaded")
        
        if not request.text.strip():
            logger.warning("Ingest request received with empty text")
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f"Ingesting text (length: {len(request.text)} characters)")
        new_chunks = chunk_text(request.text.strip())
        
        if not new_chunks:
            logger.info("No chunks created from ingested text")
            return {"status": "ok", "added_chunks": 0}
        
        logger.info(f"Creating embeddings for {len(new_chunks)} chunks")
        # Use chunk embedding cache to avoid re-embedding
        new_embeddings_list = []
        chunks_to_embed = []
        chunk_indices_to_embed = []
        
        for i, chunk_text in enumerate(new_chunks):
            # Try to get from cache (using chunk text as key for now)
            # In future, we could use chunk ID if available
            cached = chunk_embedding_cache.get(chunk_text)
            if cached is not None:
                new_embeddings_list.append((i, cached))
            else:
                chunks_to_embed.append(chunk_text)
                chunk_indices_to_embed.append(i)
        
        # Embed uncached chunks
        if chunks_to_embed:
            new_embeddings = embedding_model.encode(
                chunks_to_embed,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            faiss.normalize_L2(new_embeddings)
            
            # Store in cache and add to results
            for idx, chunk_text, embedding in zip(chunk_indices_to_embed, chunks_to_embed, new_embeddings):
                chunk_embedding_cache.set(chunk_text, embedding)
                new_embeddings_list.append((idx, embedding))
        
        # Sort by original index and create array
        new_embeddings_list.sort(key=lambda x: x[0])
        new_embeddings = np.array([emb for _, emb in new_embeddings_list])
        faiss.normalize_L2(new_embeddings)
        
        # Acquire write lock for index modification
        async with index_lock:
            faiss_index.add(new_embeddings)
            passages.extend(new_chunks)
            # Update passages_lower
            passages_lower.extend([p.lower() for p in new_chunks])
        
        # Save immediately to ensure persistence (no debounce for ingest)
        # This ensures data is saved even if server is killed
        logger.info("Saving index immediately after ingest")
        await index_save_manager.force_save(faiss_index, passages, new_embeddings)
        
        logger.info(f"Successfully ingested {len(new_chunks)} chunks")
        return {"status": "ok", "added_chunks": len(new_chunks)}
    except HTTPException:
        raise
    except ValueError as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Ingest validation error: {error_msg}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {error_msg}")
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.exception(f"Ingest error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Ingest error: {error_msg}")


def is_pipe_delimited(content: bytes) -> bool:
    """Check if file content uses pipe (|) delimiter format."""
    try:
        text = content.decode("utf-8", errors="ignore")
        lines = text.split("\n")[:20]  # Check first 20 lines
        
        pipe_count = 0
        total_lines = 0
        max_pipes_in_line = 0
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Count pipe characters
            pipes = line.count("|")
            max_pipes_in_line = max(max_pipes_in_line, pipes)
            
            # If a line has 5+ pipes, it's very likely pipe-delimited
            if pipes >= 5:
                pipe_count += 1
            
            total_lines += 1
        
        # If we have at least one line with 5+ pipes, or multiple lines with 3+ pipes
        if max_pipes_in_line >= 5:
            return True
        
        # If more than 30% of non-empty lines have 3+ pipes, it's likely pipe-delimited
        if total_lines > 0 and pipe_count / total_lines > 0.3:
            return True
        
        return False
    except:
        return False


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a document file or CSV file."""
    try:
        if not file.filename:
            logger.warning("Upload request received without filename")
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Sanitize filename
        safe_filename = sanitize_filename(file.filename)
        if safe_filename != file.filename:
            logger.warning(f"Filename sanitized: {file.filename} -> {safe_filename}")
        
        filename_lower = safe_filename.lower()
        
        # Read content first to check format
        content = await file.read()
        
        # Security: Check file size
        if len(content) > Config.MAX_UPLOAD_SIZE:
            logger.warning(f"File too large: {len(content)} bytes (max: {Config.MAX_UPLOAD_SIZE})")
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size ({Config.MAX_UPLOAD_SIZE / 1024 / 1024:.1f}MB)"
            )
        
        # Security: Check file extension
        if not any(filename_lower.endswith(ext.lower()) for ext in Config.ALLOWED_FILE_EXTENSIONS):
            logger.warning(f"Invalid file extension: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed extensions: {', '.join(Config.ALLOWED_FILE_EXTENSIONS)}"
            )
        
        # Check if content uses pipe delimiter format
        is_pipe_format = is_pipe_delimited(content)
        
        # Check if filename matches before_preprocess format
        # Pattern: bms.{device_no}.{year}-{month} or gps.{device_no}.{year}-{month}
        bms_pattern = re.match(r"^bms\.(\d+)\.(\d{4}-\d{2})", safe_filename, re.IGNORECASE)
        gps_pattern = re.match(r"^gps\.(\d+)\.(\d{4}-\d{2})", safe_filename, re.IGNORECASE)
        
        # If file uses pipe delimiter format OR matches filename pattern, save to before_preprocess
        if is_pipe_format or bms_pattern or gps_pattern:
            # Determine file type
            file_type = "BMS" if (bms_pattern or (is_pipe_format and "device_no" in content.decode("utf-8", errors="ignore")[:1000])) else "GPS"
            
            # Save to before_preprocess directory
            before_preprocess_dir = Config.ROOT / "before_preprocess"
            before_preprocess_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure .csv extension for before_preprocess files
            if not filename_lower.endswith('.csv'):
                # Change extension to .csv if it's pipe-delimited
                base_name = Path(safe_filename).stem
                output_filename = f"{base_name}.csv"
            else:
                output_filename = safe_filename
            
            # Security: Validate save path
            output_path, is_valid = validate_save_path(before_preprocess_dir, output_filename)
            if not is_valid:
                logger.error(f"Invalid save path detected: {output_path}")
                raise HTTPException(status_code=400, detail="Invalid file path")
            
            logger.info(f"Saving file to before_preprocess: {output_path}")
            
            with open(output_path, "wb") as f:
                f.write(content)
            
            logger.info(f"File saved to before_preprocess: {output_path}")
            return {
                "status": "ok",
                "filename": safe_filename,
                "saved_to": str(output_path),
                "message": f"파일이 before_preprocess 디렉토리에 저장되었습니다. (형식: {file_type}, 파이프 구분자: {'예' if is_pipe_format else '아니오'})"
            }
        
        # Handle CSV files (non-pipe format)
        elif filename_lower.endswith('.csv'):
            # Save to data directory for other CSV files
            data_dir = Config.DATA_DIR
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Security: Validate save path
            output_path, is_valid = validate_save_path(data_dir, safe_filename)
            if not is_valid:
                logger.error(f"Invalid save path detected: {output_path}")
                raise HTTPException(status_code=400, detail="Invalid file path")
            
            logger.info(f"Saving CSV file to data directory: {output_path}")
            with open(output_path, "wb") as f:
                f.write(content)
            
            logger.info(f"CSV file saved successfully: {output_path}")
            return {
                "status": "ok",
                "filename": safe_filename,
                "saved_to": str(output_path),
                "message": "CSV 파일이 data 디렉토리에 저장되었습니다."
            }
        
        # Handle text files (.txt, .md) - add to RAG index
        elif filename_lower.endswith(('.txt', '.md')):
            if faiss_index is None or embedding_model is None:
                logger.warning("Text file upload requested but index not loaded")
                raise HTTPException(status_code=503, detail="Index not loaded")
            
            try:
                text = content.decode("utf-8").strip()
            except UnicodeDecodeError:
                logger.warning(f"File encoding error: {safe_filename}")
                raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")
            
            if not text:
                logger.warning(f"Empty file uploaded: {safe_filename}")
                raise HTTPException(status_code=400, detail="File is empty")
            
            # Save text file to data directory
            data_dir = Config.DATA_DIR
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Security: Validate save path
            output_path, is_valid = validate_save_path(data_dir, safe_filename)
            if not is_valid:
                logger.error(f"Invalid save path detected: {output_path}")
                raise HTTPException(status_code=400, detail="Invalid file path")
            
            logger.info(f"Saving text file to data directory: {output_path}")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            # Add metadata for ingested text
            metadata = "타입: 사용자 입력 | 형식: 텍스트"
            logger.info(f"Chunking text file: {safe_filename}")
            new_chunks = chunk_text(text, metadata)
            
            if not new_chunks:
                logger.info(f"No chunks created from text file: {safe_filename}")
                return {
                    "status": "ok",
                    "filename": safe_filename,
                    "saved_to": str(output_path),
                    "added_chunks": 0
                }
            
            logger.info(f"Creating embeddings for {len(new_chunks)} chunks from text file")
            # Use chunk embedding cache to avoid re-embedding
            new_embeddings_list = []
            chunks_to_embed = []
            chunk_indices_to_embed = []
            
            for i, chunk_text in enumerate(new_chunks):
                # Try to get from cache (using chunk text as key)
                cached = chunk_embedding_cache.get(chunk_text)
                if cached is not None:
                    new_embeddings_list.append((i, cached))
                else:
                    chunks_to_embed.append(chunk_text)
                    chunk_indices_to_embed.append(i)
            
            # Embed uncached chunks
            if chunks_to_embed:
                new_embeddings = embedding_model.encode(
                    chunks_to_embed,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                faiss.normalize_L2(new_embeddings)
                
                # Store in cache and add to results
                for idx, chunk_text, embedding in zip(chunk_indices_to_embed, chunks_to_embed, new_embeddings):
                    chunk_embedding_cache.set(chunk_text, embedding)
                    new_embeddings_list.append((idx, embedding))
            
            # Sort by original index and create array
            new_embeddings_list.sort(key=lambda x: x[0])
            new_embeddings = np.array([emb for _, emb in new_embeddings_list])
            faiss.normalize_L2(new_embeddings)
            
            # Acquire write lock for index modification
            async with index_lock:
                faiss_index.add(new_embeddings)
                passages.extend(new_chunks)
                # Update passages_lower
                passages_lower.extend([p.lower() for p in new_chunks])
            
            # Save immediately to ensure persistence (no debounce for upload)
            # This ensures data is saved even if server is killed
            logger.info("Saving index immediately after upload")
            await index_save_manager.force_save(faiss_index, passages, new_embeddings)
            
            logger.info(f"Successfully uploaded and indexed text file: {safe_filename} ({len(new_chunks)} chunks)")
            return {
                "status": "ok",
                "filename": safe_filename,
                "saved_to": str(output_path),
                "added_chunks": len(new_chunks)
            }
        else:
            logger.warning(f"Unsupported file type: {safe_filename}")
            raise HTTPException(status_code=400, detail="Only .txt, .md, and .csv files are supported")
    except HTTPException:
        raise
    except ValueError as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Upload validation error: {error_msg}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {error_msg}")
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.exception(f"Upload error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Upload error: {error_msg}")


@app.get("/passages")
async def list_passages():
    """List all passages."""
    try:
        if faiss_index is None:
            logger.warning("List passages requested but index not loaded")
            raise HTTPException(status_code=503, detail="Index not loaded")
        
        # Acquire read lock for safe access
        async with index_lock:
            passages_snapshot = passages.copy()
        
        logger.info(f"Listed {len(passages_snapshot)} passages")
        return {"passages": passages_snapshot, "count": len(passages_snapshot)}
    except HTTPException:
        raise
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.exception(f"Error listing passages: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error listing passages: {error_msg}")


@app.post("/passages/delete")
async def delete_passages(request: DeletePassageRequest):
    """Delete passages by indices and rebuild index."""
    global faiss_index, passages, passages_lower
    
    try:
        if faiss_index is None or embedding_model is None:
            logger.warning("Delete passages requested but index not loaded")
            raise HTTPException(status_code=503, detail="Index not loaded")
        
        if not request.indices:
            logger.warning("Delete passages requested with no indices")
            raise HTTPException(status_code=400, detail="No indices provided")
        
        indices_to_delete = set(request.indices)
        logger.info(f"Deleting {len(indices_to_delete)} passages: {sorted(indices_to_delete)}")
        
        # Acquire write lock for index modification
        async with index_lock:
            if any(idx < 0 or idx >= len(passages) for idx in indices_to_delete):
                logger.warning(f"Invalid indices provided: {indices_to_delete}")
                raise HTTPException(status_code=400, detail="Invalid index")
            
            new_passages = [p for i, p in enumerate(passages) if i not in indices_to_delete]
            
            if len(new_passages) == len(passages):
                logger.info("No passages deleted (all indices were invalid or already deleted)")
                return {"status": "ok", "deleted_count": 0, "remaining_count": len(passages)}
            
            logger.info(f"Rebuilding index with {len(new_passages)} passages")
            new_embeddings = embedding_model.encode(new_passages, convert_to_numpy=True)
            dimension = new_embeddings.shape[1]
            new_index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(new_embeddings)
            new_index.add(new_embeddings)
            
            faiss_index = new_index
            passages.clear()
            passages.extend(new_passages)
            # Rebuild passages_lower
            passages_lower = [p.lower() for p in passages]
        
        # Schedule optimized save (delete operations should save immediately for data integrity)
        await index_save_manager.force_save(faiss_index, passages)
        
        deleted_count = len(passages) - len(new_passages)
        logger.info(f"Successfully deleted {deleted_count} passages, {len(new_passages)} remaining")
        return {
            "status": "ok",
            "deleted_count": deleted_count,
            "remaining_count": len(new_passages)
        }
    except HTTPException:
        raise
    except ValueError as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Delete passages validation error: {error_msg}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {error_msg}")
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.exception(f"Delete passages error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Delete passages error: {error_msg}")


@app.post("/passages/delete-all")
async def delete_all_passages():
    """Delete all passages and create empty index."""
    global faiss_index, passages, passages_lower
    
    try:
        if faiss_index is None or embedding_model is None:
            logger.warning("Delete all passages requested but index not loaded")
            raise HTTPException(status_code=503, detail="Index not loaded")
        
        # Acquire write lock for index modification
        async with index_lock:
            deleted_count = len(passages)
            logger.warning(f"Deleting all {deleted_count} passages")
            
            dimension = 384
            if embedding_model:
                dimension = embedding_model.get_sentence_embedding_dimension()
            
            new_index = faiss.IndexFlatIP(dimension)
            passages.clear()
            passages_lower.clear()
            
            faiss_index = new_index
        
        # Force save immediately for data integrity (delete operations)
        await index_save_manager.force_save(faiss_index, passages)
        
        logger.warning(f"Successfully deleted all {deleted_count} passages")
        return {
            "status": "ok",
            "deleted_count": deleted_count,
            "remaining_count": 0
        }
    except HTTPException:
        raise
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.exception(f"Delete all passages error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Delete all passages error: {error_msg}")


@app.post("/sql/generate")
async def generate_sql(request: SQLRequest):
    """Generate SQL from CSV preview or table name."""
    try:
        if request.csv_path:
            # Security: Validate CSV path
            csv_path_obj = Path(request.csv_path)
            if not csv_path_obj.exists():
                logger.warning(f"CSV file not found: {request.csv_path}")
                raise HTTPException(status_code=404, detail=f"CSV file not found: {request.csv_path}")
            
            logger.info(f"Generating SQL from CSV: {request.csv_path}")
            result = generate_sql_from_csv_preview(
                request.csv_path, 
                table_name=request.table_name or "data_table"
            )
            
            if "error" in result:
                error_msg = sanitize_error_message(Exception(result["error"]))
                logger.error(f"SQL generation error: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
            
            schema_info = result["schema_summary"]
            inferred = schema_info["inferred"]
            time_col = inferred["time_columns"][0] if inferred["time_columns"] else None
            
            # Format output
            schema_output = "## 🧩 Inferred Schema\n\n"
            for col_info in schema_info["schema_details"]:
                col = col_info["column"]
                dtype = col_info["dtype"]
                notes = []
                
                if col in inferred["key_columns"]:
                    notes.append("key")
                if col in inferred["time_columns"]:
                    notes.append("timestamp")
                if col == inferred["baas_fields"]["soc"]:
                    notes.append("SOC")
                if col == inferred["baas_fields"]["soh"]:
                    notes.append("SOH")
                if col == inferred["baas_fields"]["odometer"]:
                    notes.append("odometer")
                if col == inferred["baas_fields"]["voltage"]:
                    notes.append("voltage")
                if col == inferred["baas_fields"]["temperature"]:
                    notes.append("temperature")
                if col in inferred["baas_fields"]["gps"]:
                    notes.append("GPS")
                
                note_str = f" ({', '.join(notes)})" if notes else ""
                schema_output += f"- `{col}`: {dtype}{note_str}\n"
            
            sql_output = f"""## 🧮 Basic Statistics SQL

```sql
{result["exploration_sql"]}

{result["stats_sql"]}
```

## 🔋 BAAS Domain SQL

```sql
{result["baas_sql"]}
```"""
            
            logger.info("SQL generation completed successfully")
            return {
                "schema": schema_output,
                "sql": sql_output,
                "full_result": result
            }
        
        elif request.table_name:
            # For DB tables, we'd need db_basic_stats tool
            logger.warning("Database table analysis requested but not implemented")
            return {
                "error": "Database table analysis requires db_basic_stats tool (not yet implemented)"
            }
        
        else:
            logger.warning("SQL generation requested without csv_path or table_name")
            raise HTTPException(status_code=400, detail="Either csv_path or table_name must be provided")
    except HTTPException:
        raise
    except ValueError as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"SQL generation validation error: {error_msg}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {error_msg}")
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.exception(f"SQL generation error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"SQL generation error: {error_msg}")


@app.get("/sql/preview")
def preview_csv(csv_path: str):
    """Preview CSV file and return schema."""
    try:
        # Security: Validate CSV path
        csv_path_obj = Path(csv_path)
        if not csv_path_obj.exists():
            logger.warning(f"CSV file not found: {csv_path}")
            raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")
        
        logger.info(f"Previewing CSV: {csv_path}")
        result = csv_preview(csv_path)
        if "error" in result:
            error_msg = sanitize_error_message(Exception(result["error"]))
            logger.error(f"CSV preview error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info("CSV preview completed successfully")
        return result
    except HTTPException:
        raise
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.exception(f"CSV preview error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"CSV preview error: {error_msg}")


@app.post("/sql/generate-from-db")
async def generate_sql_from_database(request: DBConnectionRequest):
    """Generate SQL from database table."""
    try:
        # Security: Validate DB URL format (basic check)
        if not request.db_url.startswith(('postgresql://', 'mysql://', 'sqlite://')):
            logger.warning(f"Invalid database URL format: {request.db_url[:50]}...")
            raise HTTPException(status_code=400, detail="Invalid database URL format")
        
        logger.info(f"Generating SQL from database: {request.table_name}")
        result = generate_sql_from_db(
            request.db_url,
            request.table_name,
            request.schema_name
        )
        
        if "error" in result:
            error_msg = sanitize_error_message(Exception(result["error"]))
            logger.error(f"SQL generation from DB error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        schema_info = result["schema_summary"]
        inferred = schema_info["inferred"]
        
        # Format output
        schema_output = "## 🧩 Inferred Schema\n\n"
        schema_output += f"**테이블**: {request.table_name}\n"
        if request.schema_name:
            schema_output += f"**스키마**: {request.schema_name}\n"
        schema_output += f"**행 수**: {schema_info.get('row_count', 'N/A')}\n\n"
        
        for col_info in schema_info["schema_details"]:
            col = col_info["column"]
            dtype = col_info["dtype"]
            notes = []
            
            if col in inferred["key_columns"]:
                notes.append("key")
            if col in inferred["time_columns"]:
                notes.append("timestamp")
            if col == inferred["baas_fields"]["soc"]:
                notes.append("SOC")
            if col == inferred["baas_fields"]["soh"]:
                notes.append("SOH")
            if col == inferred["baas_fields"]["odometer"]:
                notes.append("odometer")
            if col == inferred["baas_fields"]["voltage"]:
                notes.append("voltage")
            if col == inferred["baas_fields"]["temperature"]:
                notes.append("temperature")
            if col in inferred["baas_fields"]["gps"]:
                notes.append("GPS")
            
            note_str = f" ({', '.join(notes)})" if notes else ""
            schema_output += f"- `{col}`: {dtype}{note_str}\n"
        
        sql_output = f"""## 🧮 Basic Statistics SQL

```sql
{result["exploration_sql"]}

{result["stats_sql"]}
```

## 🔋 BAAS Domain SQL

```sql
{result["baas_sql"]}
```"""
        
        logger.info("SQL generation from database completed successfully")
        return {
            "schema": schema_output,
            "sql": sql_output,
            "full_result": result
        }
    except HTTPException:
        raise
    except ValueError as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"SQL generation from DB validation error: {error_msg}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {error_msg}")
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.exception(f"SQL generation from DB error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error: {error_msg}")


@app.post("/influxdb/generate-flux")
async def generate_flux_from_influxdb_endpoint(request: InfluxDBConnectionRequest):
    """Generate Flux queries from InfluxDB bucket/measurement."""
    try:
        logger.info(f"Generating Flux from InfluxDB: bucket={request.bucket}, measurement={request.measurement}")
        result = generate_flux_from_influxdb(
            request.url,
            request.bucket,
            request.measurement,
            request.org,
            request.token
        )
        
        if "error" in result:
            error_msg = sanitize_error_message(Exception(result["error"]))
            logger.error(f"Flux generation from InfluxDB error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        schema_info = result["schema_summary"]
        
        # Format output
        schema_output = "## 🧩 InfluxDB Schema\n\n"
        schema_output += f"**Bucket**: {schema_info.get('bucket', 'N/A')}\n"
        schema_output += f"**Measurement**: {schema_info.get('measurement', 'N/A')}\n"
        schema_output += f"**데이터 포인트 수**: {schema_info.get('row_count', 'N/A')}\n\n"
        
        if schema_info.get("measurements"):
            schema_output += "**측정값(Measurements)**:\n"
            for m in schema_info["measurements"]:
                schema_output += f"- `{m}`\n"
            schema_output += "\n"
        
        if schema_info.get("fields"):
            schema_output += "**필드(Fields)**:\n"
            for f in schema_info["fields"][:20]:  # Limit to 20 fields
                schema_output += f"- `{f}`\n"
            if len(schema_info["fields"]) > 20:
                schema_output += f"- ... (총 {len(schema_info['fields'])}개 필드)\n"
            schema_output += "\n"
        
        if schema_info.get("tags"):
            schema_output += "**태그(Tags)**:\n"
            for t in schema_info["tags"][:20]:  # Limit to 20 tags
                schema_output += f"- `{t}`\n"
            if len(schema_info["tags"]) > 20:
                schema_output += f"- ... (총 {len(schema_info['tags'])}개 태그)\n"
            schema_output += "\n"
        
        flux_output = f"""## 🔍 Exploration Flux Queries

```flux
{result["exploration_flux"]}
```"""
        
        if result.get("stats_flux"):
            flux_output += f"""

## 📊 Statistics Flux Queries

```flux
{result["stats_flux"]}
```"""
        
        if result.get("baas_flux"):
            flux_output += f"""

## 🔋 BAAS Domain Flux Queries

```flux
{result["baas_flux"]}
```"""
        
        logger.info("Flux generation from InfluxDB completed successfully")
        return {
            "schema": schema_output,
            "flux": flux_output,
            "full_result": result
        }
    except HTTPException:
        raise
    except ValueError as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Flux generation from InfluxDB validation error: {error_msg}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {error_msg}")
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.exception(f"Flux generation from InfluxDB error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error: {error_msg}")


@app.post("/influxdb/stats")
async def get_influxdb_stats(request: InfluxDBConnectionRequest):
    """Get basic statistics from InfluxDB bucket/measurement."""
    try:
        logger.info(f"Getting stats from InfluxDB: bucket={request.bucket}, measurement={request.measurement}")
        result = influxdb_basic_stats(
            request.url,
            request.bucket,
            request.measurement,
            request.org,
            request.token
        )
        
        if "error" in result:
            error_msg = sanitize_error_message(Exception(result["error"]))
            logger.error(f"InfluxDB stats error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info("InfluxDB stats retrieved successfully")
        return {
            "status": "ok",
            "bucket": result["bucket"],
            "measurement": result["measurement"],
            "measurements": result["measurements"],
            "fields": result["fields"],
            "tags": result["tags"],
            "row_count": result["row_count"],
            "sample_data": result["sample_data"]
        }
    except HTTPException:
        raise
    except ValueError as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"InfluxDB stats validation error: {error_msg}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {error_msg}")
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.exception(f"InfluxDB stats error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error: {error_msg}")


@app.post("/preprocess")
async def preprocess_data():
    """Preprocess all files in before_preprocess directory."""
    try:
        logger.info("Starting preprocessing of files in before_preprocess directory")
        results = preprocess_all_files()
        
        logger.info(f"Preprocessing completed: {len(results['processed'])} processed, {len(results['errors'])} errors")
        return {
            "status": "ok",
            "processed_count": len(results["processed"]),
            "error_count": len(results["errors"]),
            "processed": results["processed"],
            "errors": results["errors"]
        }
    except HTTPException:
        raise
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.exception(f"Preprocessing error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {error_msg}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.refrag_server:app", host="0.0.0.0", port=8011, reload=True)

