"""FastAPI RAG server."""
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import faiss
import numpy as np
import time
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from app.config import Config
from app.rag_index import load_index, chunk_text, save_index
from app.ollama_client import generate
from app.domain_dict import get_domain_dict
from app.hybrid_retrieval import HybridRetriever
from app.reranker import DomainReranker
from app.llm.ollama_client import OllamaLLMClient
from app.models.schemas import Chunk, RetrievedChunk
from app.compression.policy import HeuristicCompressionPolicy
from app.compression.compressor import compress_chunk, expand_chunk
from app.utils.tokenizer import count_tokens
from app.sql_tools import (
    csv_preview, 
    infer_baas_schema,
    generate_basic_exploration_sql,
    generate_basic_stats_sql,
    generate_baas_domain_sql,
    generate_sql_from_csv_preview,
    db_basic_stats,
    generate_sql_from_db
)
from app.preprocessor import preprocess_all_files, preprocess_bms_file, preprocess_gps_file
from app.data_columns import get_actual_used_columns, format_columns_for_prompt, format_missing_variables_info, format_unused_variables_info
from app.utils.postprocess import postprocess_answer
from app.utils.query_expansion import expand_query_semantically

app = FastAPI(title="RAG Server", version="1.0.0")

faiss_index: faiss.Index | None = None
passages: List[str] = []
embedding_model: SentenceTransformer | None = None
hybrid_retriever: HybridRetriever | None = None
reranker: DomainReranker | None = None
ollama_chat_client: OllamaLLMClient | None = None
compression_policy: HeuristicCompressionPolicy | None = None
doc_id_to_original: Dict[str, str] = {}  # Mapping for original document content

# Lock for thread-safe access to index and passages
import asyncio
index_lock = asyncio.Lock()  # Protects faiss_index and passages from concurrent access


class QueryRequest(BaseModel):
    question: str
    top_k: int = Config.TOP_K_DEFAULT


class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]
    timing: Optional[dict] = None  # Timing information


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


@app.get("/", response_class=HTMLResponse)
def root_page() -> str:
    """Simple frontend page for RAG interaction."""
    return r"""
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <title>EV Battery RAG Demo</title>
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
      background: linear-gradient(135deg, #22c55e, #0ea5e9);
      color: #020617;
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
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      font-size: 0.75rem;
      padding: 4px 8px;
      border-radius: 999px;
      background: #020617;
      border: 1px solid #1f2937;
      color: #9ca3af;
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
  </style>
</head>
<body>
  <div class="container">
    <h1>EV Battery RAG Assistant</h1>
    <div class="subtitle">Gemma3 27B · Ollama · FAISS · Sentence-Transformers</div>

    <div class="card">
      <div class="status-bar">
        <div class="status-chip">
          <span id="apiStatusDot" class="status-dot"></span>
          <span id="apiStatusText">API 연결 확인 중...</span>
        </div>
        <div class="muted small">/query · /ingest · /health</div>
      </div>
    </div>

    <div class="card">
      <h2>질문하기</h2>
      <label for="question">질문</label>
      <textarea id="question" placeholder="예) EV Battery 수명에 영향을 주는 요소가 뭐야?"></textarea>
      <div class="row">
        <div>
          <label for="topK">Top K</label>
          <input id="topK" type="number" min="1" max="10" value="5" />
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
    </div>

    <div class="card">
      <h2>참고 문서 컨텍스트</h2>
      <div id="contexts" class="contexts muted small">질문을 보내면, 여기 컨텍스트가 표시됩니다.</div>
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
      <h2>SQL 생성</h2>
      <div style="margin-bottom:16px;">
        <label style="display:block; margin-bottom:8px; font-weight:600;">데이터 소스 선택</label>
        <div style="display:flex; gap:8px;">
          <button id="sourceCsvBtn" class="secondary" style="background:#22c55e; color:#020617;">CSV 파일</button>
          <button id="sourceDbBtn" class="secondary">데이터베이스</button>
        </div>
      </div>

      <div id="csvSource" style="display:block;">
        <label for="csvPath">CSV 파일 경로</label>
        <input type="text" id="csvPath" placeholder="/home/keti_spark1/j309/data/preprocessed_bms.01241248529.2022-12.csv" style="margin-bottom:8px;" />
        <label for="tableName">테이블 이름 (선택)</label>
        <input type="text" id="tableName" placeholder="baas_data" value="baas_data" style="margin-bottom:8px;" />
      </div>

      <div id="dbSource" style="display:none;">
        <label for="dbUrl">데이터베이스 URL</label>
        <input type="text" id="dbUrl" placeholder="postgresql://user:password@host:port/dbname" style="margin-bottom:8px;" />
        <label for="dbTableName">테이블 이름</label>
        <input type="text" id="dbTableName" placeholder="baas_data" style="margin-bottom:8px;" />
        <label for="dbSchema">스키마 (선택)</label>
        <input type="text" id="dbSchema" placeholder="public" style="margin-bottom:8px;" />
      </div>

      <div class="row">
        <button id="generateSqlBtn" class="secondary" style="background:linear-gradient(135deg, #8b5cf6, #6366f1);">
          <span>SQL 생성</span>
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
    const csvSourceEl = document.getElementById("csvSource");
    const dbSourceEl = document.getElementById("dbSource");
    const csvPathEl = document.getElementById("csvPath");
    const tableNameEl = document.getElementById("tableName");
    const dbUrlEl = document.getElementById("dbUrl");
    const dbTableNameEl = document.getElementById("dbTableName");
    const dbSchemaEl = document.getElementById("dbSchema");
    const sqlResultEl = document.getElementById("sqlResult");
    const sqlSchemaEl = document.getElementById("sqlSchema");
    const sqlCodeEl = document.getElementById("sqlCode");
    const sqlStatusEl = document.getElementById("sqlStatus");
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

    async function checkHealth() {
      try {
        const res = await fetch("/health");
        if (!res.ok) throw new Error();
        const data = await res.json();
        apiStatusDot.classList.remove("offline");
        apiStatusText.textContent =
          data.index_loaded
            ? `온라인 · 패시지 ${data.num_passages}개`
            : "온라인 · 인덱스 미로딩";
      } catch (e) {
        apiStatusDot.classList.add("offline");
        apiStatusText.textContent = "오프라인 · 서버/인덱스 확인 필요";
      }
    }

    async function sendQuery() {
      const question = questionEl.value.trim();
      const topK = parseInt(topKEl.value || "5", 10);
      if (!question) {
        alert("질문을 입력하세요.");
        return;
      }
      askBtn.disabled = true;
      queryStatusEl.textContent = "질문 처리 중...";
      answerEl.textContent = "";
      contextsEl.textContent = "";

      const clientStartTime = performance.now();

      try {
        const res = await fetch("/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, top_k: topK })
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

        if (Array.isArray(data.contexts) && data.contexts.length > 0) {
          const items = data.contexts
            .map((c) => `<li>${c.replace(/</g, "&lt;")}</li>`)
            .join("");
          contextsEl.classList.remove("muted");
          contextsEl.innerHTML = `<ul>${items}</ul>`;
        } else {
          contextsEl.classList.add("muted");
          contextsEl.textContent = "컨텍스트가 없습니다.";
        }
        
        // Display timing information
        let timingText = `완료 · 총 시간: ${clientTotalTime}초`;
        if (data.timing) {
          const t = data.timing;
          timingText += ` (임베딩: ${t.embedding?.toFixed(2) || 0}초, 검색: ${t.search?.toFixed(2) || 0}초, LLM: ${t.llm_generation?.toFixed(2) || 0}초)`;
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
      sourceCsvBtn.style.background = "#22c55e";
      sourceCsvBtn.style.color = "#020617";
      sourceDbBtn.style.background = "#111827";
      sourceDbBtn.style.color = "#e5e7eb";
    });
    
    sourceDbBtn.addEventListener("click", () => {
      currentSource = "db";
      csvSourceEl.style.display = "none";
      dbSourceEl.style.display = "block";
      sourceDbBtn.style.background = "#22c55e";
      sourceDbBtn.style.color = "#020617";
      sourceCsvBtn.style.background = "#111827";
      sourceCsvBtn.style.color = "#e5e7eb";
    });

    async function generateSQL() {
      generateSqlBtn.disabled = true;
      sqlStatusEl.textContent = "SQL 생성 중...";
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
        } else {
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
        }
        
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "SQL 생성 실패");
        }
        
        const data = await res.json();
        sqlSchemaEl.textContent = data.schema || "";
        sqlCodeEl.textContent = data.sql || "";
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


@app.on_event("startup")
async def startup():
    """Load index and build domain dictionary on server startup."""
    global faiss_index, passages, embedding_model, hybrid_retriever, reranker, ollama_chat_client, compression_policy, doc_id_to_original
    
    try:
        index, loaded_passages, model_name, original_mapping, _ = load_index()  # embeddings not used in rag_server
        faiss_index = index
        passages = loaded_passages
        embedding_model = SentenceTransformer(model_name)
        doc_id_to_original = original_mapping
        
        # Build domain dictionary from rules files
        print("Building domain dictionary from rules files...")
        domain_dict = get_domain_dict()
        print(f"Domain dictionary built: {len(domain_dict.variable_to_info)} variables, {len(domain_dict.table_to_variables)} tables")
        
        # Initialize hybrid retriever
        print("Initializing hybrid retriever (BM25)...")
        hybrid_retriever = HybridRetriever(passages, domain_dict)
        print("Hybrid retriever initialized")
        
        # Initialize reranker
        print("Initializing domain reranker...")
        reranker = DomainReranker(domain_dict, use_cross_encoder=True)
        print("Domain reranker initialized")
        
        # Initialize REFRAG components
        print("Initializing REFRAG components...")
        ollama_chat_client = OllamaLLMClient()
        # REFRAG-style: Use expand_frac if configured, otherwise use fixed count
        compression_policy = HeuristicCompressionPolicy(
            max_expanded_chunks=Config.MAX_EXPANDED_CHUNKS,
            expand_frac=Config.EXPAND_FRAC if hasattr(Config, 'EXPAND_FRAC') else None
        )
        print("REFRAG components initialized")
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Index not found. Please run 'python build_index.py' first. "
            f"Error: {e}"
        )


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "index_loaded": faiss_index is not None,
        "num_passages": len(passages) if passages else 0
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


# postprocess_answer is now imported from app.utils.postprocess


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
    """RAG query endpoint with domain-specific enhancements."""
    start_time = time.time()
    timing = {
        "total": 0,
        "embedding": 0,
        "search": 0,
        "exact_match": 0,
        "rerank": 0,
        "context_processing": 0,
        "llm_generation": 0
    }
    
    if faiss_index is None or embedding_model is None or hybrid_retriever is None or reranker is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Get domain dictionary
    domain_dict = get_domain_dict()
    
    top_k = max(1, min(request.top_k, len(passages)))
    
    # For field-related queries, increase top_k to get more context
    is_field_query = any(keyword in request.question.lower() for keyword in ['변수', '필드', '사용', '미사용', '비고'])
    if is_field_query:
        top_k = min(top_k * 2, len(passages))  # Double the context for field queries
    
    # [1] Domain-specific query expansion (optimized - limit expansions)
    domain_expansions = domain_dict.expand_query(request.question)
    # Combine with semantic expansion
    expanded_queries = expand_query_semantically(request.question, max_expansions=2)  # Reduced from 3
    # Merge and deduplicate
    all_expansions = list(dict.fromkeys(domain_expansions + expanded_queries))[:Config.MAX_QUERY_EXPANSIONS]
    
    # [2] Dense retrieval (semantic search) - for hybrid search
    embedding_start = time.time()
    query_embeddings = embedding_model.encode(
        all_expansions, 
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=len(all_expansions)  # Process all at once
    )
    faiss.normalize_L2(query_embeddings)
    
    # Average the embeddings for better semantic matching
    query_embedding = np.mean(query_embeddings, axis=0, keepdims=True)
    faiss.normalize_L2(query_embedding)
    timing["embedding"] = time.time() - embedding_start
    
    # [3] Hybrid retrieval: Dense + Sparse + Exact
    search_start = time.time()
    # Acquire read lock for index access and create snapshot
    async with index_lock:
        # Dense search (optimized - reduce candidate count)
        search_k = min(top_k * Config.SEARCH_K_MULTIPLIER, len(passages))
        distances, indices = faiss_index.search(query_embedding, search_k)
        
        # Create a snapshot of passages for this query (to avoid changes during processing)
        passages_snapshot = passages.copy()
    
    # Convert to list of (index, distance) tuples
    dense_results = [(int(indices[0][i]), float(distances[0][i])) for i in range(len(indices[0]))]
    
    # Hybrid search: combine dense, sparse, and exact (use snapshot)
    hybrid_results = hybrid_retriever.hybrid_search(
        request.question,
        dense_results,
        top_k=search_k,
        dense_weight=0.6,
        sparse_weight=0.3,
        exact_boost=1.5
    )
    timing["search"] = time.time() - search_start
    timing["exact_match"] = 0  # Exact match is now part of hybrid search
    
    # Extract indices and scores from hybrid results
    context_scores = [(idx, score, is_exact) for idx, score, is_exact in hybrid_results]
    
    # Initialize context list and seen indices
    context_list = []
    seen_indices = set()
    
    question_lower = request.question.lower()
    
    # Detect query types once (reused throughout the function)
    query_types = detect_query_types(request.question)
    is_and_exclusion = query_types['is_and_exclusion']
    is_or_exclusion = query_types['is_or_exclusion']
    is_not_used_query = query_types['is_not_used_query']
    is_uncertain_query = query_types['is_uncertain_query']
    is_field_query = query_types['is_field_query']
    
    # [5.5] For "not used" queries, force search for "실제 사용하지 않음" keywords
    if is_not_used_query:
        # Get existing indices to avoid duplicates
        existing_indices = {idx for idx, _, _ in context_scores}
        
        # Find all "not used" passages (use snapshot)
        not_used_indices = find_not_used_passages(passages_snapshot, exclude_indices=existing_indices)
        
        # Add with high score to ensure they're included
        for idx in not_used_indices:
            context_scores.append((idx, 1.0, True))  # High score, mark as exact match
            existing_indices.add(idx)
    
    # [6] Domain-specific filtering
    # Sort: exact matches first, then by relevance score
    context_scores.sort(key=lambda x: (not x[2], -x[1]), reverse=False)  # is_exact=True first, then by score
    table_filter = None
    
    # Detect table filter from query (more robust detection)
    if "bms" in question_lower:
        if "gps" not in question_lower or "속하지 않는" in question_lower or "포함되지 않는" in question_lower:
            table_filter = "BMS"
    elif "gps" in question_lower:
        if "bms" not in question_lower or "속하지 않는" in question_lower or "포함되지 않는" in question_lower:
            table_filter = "GPS"
    
    # Filter by table if specified (but keep exact matches)
    if table_filter:
        filtered_scores = []
        table_vars = domain_dict.table_to_variables.get(table_filter, set())
        
        for idx, score, is_exact in context_scores:
            # Use snapshot to avoid index changes during processing
            if idx < len(passages_snapshot):
                passage = passages_snapshot[idx]
            else:
                continue  # Skip if index is out of bounds
            matches_table = False
            
            # Check if passage contains variables from the specified table
            for var_name in table_vars:
                if var_name in passage or var_name.lower() in passage.lower():
                    matches_table = True
                    break
            
            # Also check metadata for table designation
            if f"테이블구분: {table_filter}" in passage or f"테이블구분:{table_filter}" in passage:
                matches_table = True
            elif f"테이블구분: {table_filter.lower()}" in passage.lower():
                matches_table = True
            
            # Keep if matches table (테이블 필터링 우선, is_exact는 테이블 일치 시에만 적용)
            # 테이블 불일치 시 제외 (GPS 질문에 BMS 변수, BMS 질문에 GPS 변수 제외)
            if matches_table:
                filtered_scores.append((idx, score, is_exact))
            # is_exact=True여도 테이블 불일치면 제외 (테이블 필터링 우선)
        
        context_scores = filtered_scores
    
    # Filter by relevance threshold (optimized - keep top results, limit to reasonable number)
    max_contexts = min(top_k, 10)  # Limit max contexts for performance
    if context_scores:
        # Keep exact matches + top semantic results
        exact_contexts = [(idx, score) for idx, score, is_exact in context_scores if is_exact]
        semantic_contexts = [(idx, score) for idx, score, is_exact in context_scores if not is_exact]
        
        # Apply relevance threshold to semantic results
        if semantic_contexts:
            relevance_threshold = semantic_contexts[0][1] * 0.5
            filtered_semantic = [(idx, score) for idx, score in semantic_contexts if score >= relevance_threshold]
            filtered_semantic = filtered_semantic[:max_contexts - len(exact_contexts)]
        else:
            filtered_semantic = []
        
        filtered_contexts = exact_contexts + filtered_semantic
    else:
        filtered_contexts = []
    
    # Mark filtered contexts as seen
    for idx, _ in filtered_contexts:
        seen_indices.add(idx)
    
    # [7] Rerank passages for better relevance (optimized - conditional reranking)
    rerank_start = time.time()
    filtered_indices = [idx for idx, _ in filtered_contexts]
    
    # Only rerank if we have enough candidates and reranker is enabled
    if Config.USE_RERANKER and len(filtered_indices) >= Config.RERANKER_MIN_CANDIDATES:
        # Rerank the filtered passages (limit to top candidates for speed)
        # Use snapshot for reranking
        reranked_results = reranker.rerank(
            request.question,
            passages_snapshot,
            filtered_indices,
            top_k=min(len(filtered_indices), top_k + 3)  # Reduced from top_k * 2
        )
        # Extract reranked indices
        reranked_indices = [idx for idx, _ in reranked_results]
    else:
        # Skip reranking for speed (use filtered results directly)
        reranked_indices = filtered_indices[:top_k]
    
    timing["rerank"] = time.time() - rerank_start
    
    # [9] Prioritize rules passages (after reranking)
    # Use snapshot for prioritization
    prioritized_indices = domain_dict.prioritize_rules_passages(passages_snapshot, reranked_indices)
    
    # [9.5] For "not used" queries, force include passages with "실제 사용하지 않음"
    if is_not_used_query:
        # Find all "not used" passages (excluding already prioritized ones)
        # 테이블 타입 결정
        table_type_for_search = None
        if "bms" in question_lower and "gps" not in question_lower:
            table_type_for_search = "bms"
        elif "gps" in question_lower and "bms" not in question_lower:
            table_type_for_search = "gps"
        not_used_indices = find_not_used_passages(passages_snapshot, exclude_indices=set(prioritized_indices), table_type=table_type_for_search)
        
        # Add not_used_indices to the beginning of prioritized_indices (highest priority)
        if not_used_indices:
            prioritized_indices = not_used_indices + [idx for idx in prioritized_indices if idx not in set(not_used_indices)]
    
    # [10] Filter out BMS/GPS if AND exclusion query
    if is_and_exclusion:
        filtered_indices = []
        for idx in prioritized_indices:
            if 0 <= idx < len(passages_snapshot):
                passage = passages_snapshot[idx]
                # Exclude if has BMS or GPS table designation
                if not has_table_designation(passage, "BMS") and not has_table_designation(passage, "GPS"):
                    filtered_indices.append(idx)
        prioritized_indices = filtered_indices
    
    # [9.6] For "not used" queries, ensure "실제 사용하지 않음" passages are included even after top_k limit
    if is_not_used_query:
        # Find indices with "실제 사용하지 않음"
        not_used_in_prioritized = [
            idx for idx in prioritized_indices 
            if idx < len(passages_snapshot) and '실제 사용하지 않음' in passages_snapshot[idx].lower()
        ]
        
        # If we have not_used passages, ensure they're in the final list
        if not_used_in_prioritized:
            prioritized_without_not_used = [idx for idx in prioritized_indices if idx not in not_used_in_prioritized]
            prioritized_indices = not_used_in_prioritized + prioritized_without_not_used
        
        # Limit to top_k (but ensure not_used passages are included)
        not_used_indices_final = [
            idx for idx in prioritized_indices 
            if idx < len(passages_snapshot) and '실제 사용하지 않음' in passages_snapshot[idx].lower()
        ]
        other_indices = [idx for idx in prioritized_indices if idx not in not_used_indices_final]
        prioritized_indices = not_used_indices_final + other_indices[:max(0, top_k - len(not_used_indices_final))]
    else:
        prioritized_indices = prioritized_indices[:top_k]
    
    # Store retrieved chunks with scores for REFRAG processing
    retrieved_chunks_with_scores = []
    seen_in_retrieved = set()
    
    # For "not used" queries, force include "실제 사용하지 않음" passages first
    if is_not_used_query:
        # 테이블 타입 결정
        table_type_for_search = None
        if "bms" in question_lower and "gps" not in question_lower:
            table_type_for_search = "bms"
        elif "gps" in question_lower and "bms" not in question_lower:
            table_type_for_search = "gps"
        not_used_indices_for_retrieved = find_not_used_passages(passages_snapshot, exclude_indices=seen_in_retrieved, table_type=table_type_for_search)
        
        for idx in not_used_indices_for_retrieved:
            if idx < len(passages_snapshot):
                passage = passages_snapshot[idx]
                # Find score for this index
                score = 1.0  # High score for not_used passages
                for i, (idx2, score2, _) in enumerate(context_scores):
                    if idx2 == idx:
                        score = max(score2, 1.0)  # Use higher score
                        break
                retrieved_chunks_with_scores.append((idx, score))
                context_list.append(passage)
                seen_in_retrieved.add(idx)
    
    # Add other prioritized indices
    for idx in prioritized_indices:
        if 0 <= idx < len(passages_snapshot) and idx not in seen_in_retrieved:
            # Find score for this index
            score = 0.0
            for i, (idx2, score2, _) in enumerate(context_scores):
                if idx2 == idx:
                    score = score2
                    break
            
            retrieved_chunks_with_scores.append((idx, score))
            context_list.append(passages[idx])  # Keep for backward compatibility
            seen_in_retrieved.add(idx)
    
    # [8] Keyword-based search (sparse retrieval) for field queries
    if is_field_query and len(context_list) < top_k:
        question_lower = request.question.lower()
        keywords = []
        
        # Enhanced keyword extraction using domain dictionary
        if any(term in question_lower for term in ['사용하지', '사용되지', '미사용', '사용 안']):
            keywords = [
                '실제 사용하지 않음', 
                '사용하지 않음', 
                '미사용', 
                '비고: 실제 사용하지 않음', 
                '개발상의 이유로 존재',
                '개발상이유로 존재',
                '개발상 이유로 존재',
                '개발상'
            ]
        elif '비고' in question_lower:
            keywords = ['비고']
        
        # Also check domain dictionary for variable-related keywords
        exact_matches = domain_dict.find_exact_matches(request.question)
        for match_term, match_type in exact_matches:
            if match_type.startswith("variable"):
                var_info = domain_dict.variable_to_info.get(match_term, {})
                if var_info.get("note"):
                    keywords.append(f"비고: {var_info['note']}")
        
        if keywords:
            # Search through passages (prioritize rules passages)
            keyword_matches = []
            # First search in rules passages
            for idx in prioritized_indices:
                if idx not in seen_indices and idx < len(passages):
                    if idx < len(passages_snapshot):
                        passage = passages_snapshot[idx]
                    else:
                        continue
                    passage_lower = passage.lower()
                    
                    # For "not used" queries, filter more strictly
                    if is_not_used_query:
                        has_not_used, _, has_excluded = is_not_used_passage(passage, passage_lower)
                        
                        if has_not_used and not has_excluded:
                            keyword_matches.append(idx)
                            seen_indices.add(idx)
                            if len(keyword_matches) >= (top_k - len(context_list)):
                                break
                    else:
                        # For other queries, use normal keyword matching
                        if any(keyword in passage_lower for keyword in keywords):
                            keyword_matches.append(idx)
                            seen_indices.add(idx)
                            if len(keyword_matches) >= (top_k - len(context_list)):
                                break
            
            # If still need more, search in remaining passages
            if len(keyword_matches) < (top_k - len(context_list)):
                search_limit = min(len(passages), top_k * 3)
                for idx in range(search_limit):
                    if idx not in seen_indices and idx < len(passages):
                        if idx < len(passages_snapshot):
                        passage = passages_snapshot[idx]
                    else:
                        continue
                        passage_lower = passage.lower()
                        
                        # For "not used" queries, filter more strictly
                        if is_not_used_query:
                            has_not_used, _, has_excluded = is_not_used_passage(passage, passage_lower)
                            
                            if has_not_used and not has_excluded:
                                keyword_matches.append(idx)
                                seen_indices.add(idx)
                                if len(keyword_matches) >= (top_k - len(context_list)):
                                    break
                        else:
                            if any(keyword in passage_lower for keyword in keywords):
                                keyword_matches.append(idx)
                                seen_indices.add(idx)
                                if len(keyword_matches) >= (top_k - len(context_list)):
                                    break
            
            # Add keyword matches (lower priority than exact/semantic)
            for idx in keyword_matches:
                if idx not in [i for i, _ in filtered_contexts]:
                    if idx < len(passages_snapshot):
                        context_list.append(passages_snapshot[idx])
    
    if not context_list:
        timing["total"] = time.time() - start_time
        return QueryResponse(
            answer="문서에서 해당 내용을 찾을 수 없습니다.",
            contexts=[],
            timing=timing
        )
    
    # Measure context processing time
    context_start = time.time()
    
    # Query types are already detected above, reuse them
    
    # REFRAG: Convert passages to Chunk objects and apply Compress-Sense-Expand
    if ollama_chat_client is None or compression_policy is None:
        raise HTTPException(status_code=503, detail="REFRAG components not initialized")
    
    # Normalize scores to [0, 1] range for Pydantic validation
    if retrieved_chunks_with_scores:
        scores = [s for _, s in retrieved_chunks_with_scores]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        score_range = max_score - min_score if max_score != min_score else 1.0
    else:
        score_range = 1.0
        min_score = 0.0
    
    # Convert passages to RetrievedChunk objects
    retrieved_chunks = []
    for idx, score in retrieved_chunks_with_scores:
        if 0 <= idx < len(passages):
            if idx < len(passages_snapshot):
                passage_text = passages_snapshot[idx]
            else:
                continue
            
            # Extract metadata
            metadata = {}
            text = passage_text
            is_summary = False
            original_doc_id = None
            
            if "[메타데이터:" in passage_text:
                metadata_start = passage_text.find("[메타데이터:")
                metadata_end = passage_text.find("]", metadata_start)
                if metadata_end != -1:
                    metadata_str = passage_text[metadata_start + len("[메타데이터:"):metadata_end]
                    for part in metadata_str.split("|"):
                        part = part.strip()
                        if ":" in part:
                            key, value = part.split(":", 1)
                            key = key.strip()
                            value = value.strip()
                            # Only add non-empty values
                            if value:
                                metadata[key] = value
                            
                            # Check if this is a summary
                            if key == "요약" and value.lower() == "true":
                                is_summary = True
                            
                            # Get document ID for original content lookup
                            if key == "문서ID":
                                original_doc_id = value
                    
                    text = passage_text[metadata_end + 1:].strip()
            
            # Store original doc_id for later use (when expanding)
            if is_summary:
                # Try to get 원본_문서ID from metadata first, then fallback to 문서ID
                if not original_doc_id:
                    original_doc_id = metadata.get("원본_문서ID") or metadata.get("문서ID")
                
                # Only add if we have a valid original_doc_id
                if original_doc_id:
                    metadata["원본_문서ID"] = original_doc_id
                metadata["요약"] = "true"
            
            # Clean metadata: remove None values and empty strings
            clean_metadata = {k: v for k, v in metadata.items() if v is not None and v != ""} if metadata else None
            
            # Create Chunk object
            chunk = Chunk(
                id=f"chunk_{idx}",
                document_id=clean_metadata.get("문서ID", f"doc_{idx}") if clean_metadata else f"doc_{idx}",
                text=text,
                token_count=count_tokens(text),
                start_offset=0,
                end_offset=len(text),
                metadata=clean_metadata
            )
            
            # Normalize score to [0, 1] range
            normalized_score = (float(score) - min_score) / score_range if score_range > 0 else 0.5
            normalized_score = max(0.0, min(1.0, normalized_score))  # Clamp to [0, 1]
            
            retrieved_chunks.append(RetrievedChunk(chunk=chunk, score=normalized_score))
    
    if not retrieved_chunks:
        timing["context_processing"] = time.time() - context_start
        timing["total"] = time.time() - start_time
        return QueryResponse(
            answer="문서에서 해당 내용을 찾을 수 없습니다.",
            contexts=[],
            timing=timing
        )
    
    # Pre-filter contexts for "not used" queries - only keep passages with "실제 사용하지 않음"
    if is_not_used_query:
        filtered_retrieved_chunks = []
        for retrieved_chunk in retrieved_chunks:
            chunk_text = retrieved_chunk.chunk.text
            has_not_used, _, _ = is_not_used_passage(chunk_text)
            
            # Include if it has "실제 사용하지 않음" keyword
            if has_not_used:
                filtered_retrieved_chunks.append(retrieved_chunk)
        
        # If we filtered out everything, keep original (fallback) - but this shouldn't happen if seq exists
        if filtered_retrieved_chunks:
            retrieved_chunks = filtered_retrieved_chunks
    
    # REFRAG Sense: Decide compression/expansion
    compression_decisions = compression_policy.decide(retrieved_chunks)
    
    # REFRAG Compress/Expand: Build prompt with compressed and expanded chunks
    expanded_chunks = []
    compressed_chunks = []
    
    for retrieved_chunk in retrieved_chunks:
        chunk = retrieved_chunk.chunk
        chunk_id = chunk.id
        decision = compression_decisions.get(chunk_id, "COMPRESS")
        
        if decision == "EXPAND":
            # If expanding a summary, get original content
            if chunk.metadata and chunk.metadata.get("요약") == "true":
                original_doc_id = chunk.metadata.get("원본_문서ID")
                if original_doc_id and doc_id_to_original:
                    from app.document_summarizer import get_original_content
                    original_content = get_original_content(original_doc_id, doc_id_to_original)
                    if original_content:
                        # Create new chunk with original content
                        original_chunk = Chunk(
                            id=chunk.id,
                            document_id=chunk.document_id,
                            text=original_content,
                            token_count=count_tokens(original_content),
                            start_offset=chunk.start_offset,
                            end_offset=len(original_content),
                            metadata={**chunk.metadata, "원본_사용": "true"} if chunk.metadata else {"원본_사용": "true"}
                        )
                        expanded_text = expand_chunk(original_chunk, retrieved_chunk.score)
                    else:
                        # Fallback to summary if original not found
                        expanded_text = expand_chunk(chunk, retrieved_chunk.score)
                else:
                    # No original mapping, use summary
                    expanded_text = expand_chunk(chunk, retrieved_chunk.score)
            else:
                # Not a summary, use as is
                expanded_text = expand_chunk(chunk, retrieved_chunk.score)
            
            expanded_chunks.append(expanded_text)
        else:
            # Compressed chunks use summary (keep it short)
            compressed_text = compress_chunk(chunk, mode="head")
            compressed_chunks.append(compressed_text)
    
    # Build REFRAG-style prompt with context window management
    context_parts = []
    
    if expanded_chunks:
        context_parts.append("=== 관련성 높은 청크 (전체 내용) ===\n")
        context_parts.extend(expanded_chunks)
        context_parts.append("")
    
    if compressed_chunks:
        context_parts.append("=== 관련성 낮은 청크 (요약) ===\n")
        context_parts.extend(compressed_chunks)
    
    context_text = "\n".join(context_parts)
    
    # REFRAG-style: Context window management (truncate if exceeds CTX_MAX_TOKENS)
    ctx_max_tokens = getattr(Config, 'CTX_MAX_TOKENS', 2048)
    context_tokens = count_tokens(context_text)
    
    if context_tokens > ctx_max_tokens:
        # If context exceeds limit, prioritize expanded chunks and truncate compressed ones
        # This is a simple REFRAG-style approach: keep expanded, reduce compressed
        expanded_text = "\n".join(expanded_chunks) if expanded_chunks else ""
        compressed_text = "\n".join(compressed_chunks) if compressed_chunks else ""
        
        # Calculate available tokens for compressed chunks
        expanded_tokens = count_tokens(expanded_text)
        available_for_compressed = max(0, ctx_max_tokens - expanded_tokens - 200)  # Reserve 200 for headers
        
        if available_for_compressed > 0 and compressed_chunks:
            # Truncate compressed chunks to fit
            compressed_lines = compressed_text.split('\n')
            truncated_compressed = []
            current_tokens = 0
            
            for line in compressed_lines:
                line_tokens = count_tokens(line)
                if current_tokens + line_tokens <= available_for_compressed:
                    truncated_compressed.append(line)
                    current_tokens += line_tokens
                else:
                    break
            
            if truncated_compressed:
                compressed_text = '\n'.join(truncated_compressed) + "\n..."
            else:
                compressed_text = ""
        
        # Rebuild context
        context_parts = []
        if expanded_chunks:
            context_parts.append("=== 관련성 높은 청크 (전체 내용) ===\n")
            context_parts.append(expanded_text)
            context_parts.append("")
        if compressed_text:
            context_parts.append("=== 관련성 낮은 청크 (요약) ===\n")
            context_parts.append(compressed_text)
        
        context_text = "\n".join(context_parts)
    
    # Check if any context comes from rules directory
    has_rules = any("rules/" in ctx or "규칙/필드정의" in ctx for ctx in context_list)
    
    # Determine query type for better prompting (reuse already defined variables)
    # question_lower, is_and_exclusion, is_or_exclusion, is_not_used_query are already defined above
    
    # General exclusion query
    is_exclusion_query = any(term in question_lower for term in [
        '속하지 않는', '포함되지 않는', '해당하지 않는'
    ]) and not is_and_exclusion and not is_or_exclusion
    
    # Check for "uncertain/unclear usage" query (사용할지 불분명한)
    is_uncertain_query = any(keyword in question_lower for keyword in [
        '불분명', '불확실', '모호', '확실하지', '명확하지', '사용할지 불분명'
    ]) and ('사용' in question_lower or '비고' in question_lower)
    
    is_field_query = any(keyword in question_lower for keyword in ['변수', '필드', '비고']) or is_not_used_query or is_uncertain_query
    
    # Few-shot examples
    few_shot_examples = ""
    if is_and_exclusion:
        few_shot_examples = """[예시] 질문: "GPS와 BMS 테이블, 둘 모두 사용하지 않는 변수들은 무엇인가?"
의미 해석: "GPS와 BMS 둘 다 속하지 않는 변수들" = AND 조건 (테이블구분이 BMS도 아니고 AND GPS도 아닌 변수)
의도 파악: 질문자는 "테이블구분이 BMS도 아니고 GPS도 아닌 다른 테이블의 변수들"을 찾고 있음
→ 예상되는 테이블: 텔레매틱스, 고객정보, COMMON, SYSTEM 등 (BMS와 GPS를 제외한 모든 테이블)

추론 단계:
1) 질문 의도 파악: "둘 모두 사용하지 않는" = "둘 다 속하지 않는" = "테이블구분이 BMS도 아니고 GPS도 아닌"
2) 모든 변수의 "테이블구분:" 필드를 정확히 확인 (구조화된 데이터에서 파싱)
3) "테이블구분: BMS"인 변수는 절대 제외 (조건에 맞지 않음)
4) "테이블구분: GPS"인 변수도 절대 제외 (조건에 맞지 않음)
5) 나머지 변수들만 선택: "테이블구분: 텔레매틱스", "테이블구분: 고객정보", "테이블구분: COMMON" 등 (BMS도 GPS도 아닌 모든 값)
6) 선택된 변수들을 나열
7) 최종 검증: 답변에 포함된 모든 변수의 테이블구분이 BMS도 아니고 GPS도 아닌지 확인

올바른 답변 예시 (문서에 정확히 나와있는 변수만):
- 변수명: odometer, 테이블구분: 텔레매틱스, 설명: 주행거리km (문서에 "변수명: odometer"로 명시됨)
- 변수명: door, 테이블구분: 텔레매틱스, 설명: 도어 상태 (문서에 "변수명: door"로 명시됨)
- 변수명: 주소지, 테이블구분: 고객정보, 설명: ... (문서에 "변수명: 주소지"로 명시됨)
- 변수명: gps_data, 테이블구분: 텔레매틱스, 설명: ... (문서에 "변수명: gps_data"로 명시됨)
- 변수명: bms_svc_mode, 테이블구분: 텔레매틱스, 설명: ... (문서에 "변수명: bms_svc_mode"로 명시됨)

잘못된 답변 예시 (절대 하지 말 것 - 이런 답변은 완전히 잘못됨):
- 변수명: fastChargingRelayOn, 테이블구분: BMS, 설명: ... (X - BMS는 절대 포함하면 안 됨)
- 변수명: chargingCableConnected, 테이블구분: BMS, 설명: ... (X - BMS는 절대 포함하면 안 됨)
- 변수명: gps_lat, 테이블구분: GPS, 설명: ... (X - GPS는 절대 포함하면 안 됨)
- 변수명: 신r, 테이블구분: 텔레매틱스, 설명: ... (X - 불완전하거나 이상한 변수명, 파싱 오류 가능성)
- 변수명: 신, 테이블구분: 텔레매틱스, 설명: ... (X - 불완전한 변수명, 원본이 "신호"일 수 있음)

⚠️ 매우 중요:
- 문서에 나온 "테이블구분:" 값을 그대로 사용하라. 임의로 변경하거나 추측하지 말 것
- 질문의 의도를 정확히 파악하라: "사용하지 않는"이 "비고 필드의 미사용"이 아니라 "테이블에 속하지 않는"을 의미함
---
"""
    elif is_or_exclusion:
        few_shot_examples = """[예시] 질문: "GPS 또는 BMS 테이블에 속하지 않는 변수들은 무엇인가?"
의미 해석: "GPS 또는 BMS에 속하지 않는 변수들" = OR 조건 (테이블구분이 BMS가 아니거나 OR GPS가 아닌 변수)
추론: 
1) 모든 변수의 "테이블구분:" 필드를 확인
2) "테이블구분: BMS"가 아니거나 "테이블구분: GPS"가 아닌 변수 찾기 (OR 조건)
3) 조건에 맞는 변수들을 나열
4) 각 변수의 변수명, 테이블구분, 설명을 포함하여 답변

주의: "또는"은 OR 조건이므로, BMS도 아니고 GPS도 아닌 변수 + BMS는 아니지만 GPS인 변수 + BMS는 있지만 GPS가 아닌 변수 모두 포함
---
"""
    elif is_exclusion_query and ('bms' in question_lower and 'gps' in question_lower):
        few_shot_examples = """[예시] 질문: "BMS와 GPS 모두 속하지 않는 것들은 무엇인가?"
의미 해석: "BMS도 아니고 GPS도 아닌 변수들" = AND 조건
추론: 
1) 모든 변수의 "테이블구분:" 필드를 확인
2) "테이블구분: BMS"와 "테이블구분: GPS"를 모두 제외 (AND 조건)
3) 나머지 변수들을 나열
---
"""
    elif is_uncertain_query:
        few_shot_examples = """[예시] 질문: "BMS에 속해있지만, 실제로 사용할지 불분명한 것들을 찾아봐라"
의미 해석: "비고 필드에 사용 여부가 불확실하거나 모호하게 적힌 변수들"을 의미함
→ 명시적으로 "실제 사용하지 않음"이라고 적힌 것은 제외
→ 명시적으로 "사용함"이라고 적힌 것도 제외
→ 불확실하거나 모호한 내용만 포함

추론: 
1) "테이블구분: BMS"인 모든 변수 찾기
2) 각 변수의 "비고:" 필드를 확인:
   - 비고 필드가 비어있거나
   - "검토 필요", "추가 확인 필요", "불확실", "모호" 등의 키워드가 있거나
   - 사용 여부가 명확하지 않은 내용
3) 명시적으로 "실제 사용하지 않음", "미사용"이라고 적힌 것은 제외
4) 명시적으로 "사용함", "사용 중"이라고 적힌 것도 제외
5) 조건에 맞는 변수들만 나열: 변수명, 테이블구분, 설명, 비고 내용 포함

올바른 답변 예시:
- 변수명: xxx, 테이블구분: BMS, 설명: ..., 비고: (비어있음 또는 불확실한 내용)
- 변수명: yyy, 테이블구분: BMS, 설명: ..., 비고: 검토 필요

잘못된 답변 예시:
- "실제 사용하지 않음"이라고 명시된 변수 포함 (X - 명시적이므로 제외)
- 모든 변수를 나열 (X - 불분명한 것만 나열)
---
"""
    elif is_not_used_query and "사용하지" in question_lower:
        few_shot_examples = """[예시] 질문: "BMS 테이블에 존재는 하지만 실제로는 사용되지 않는 변수는?"
의미 해석: "비고 필드에 '실제 사용하지 않음'이라고 명시적으로 적힌 변수들만"을 의미함

⚠️⚠️⚠️ 절대 규칙 ⚠️⚠️⚠️
1. "비고:" 필드에 다음 키워드 중 하나라도 정확히 포함된 변수만 답변에 포함:
   - "실제 사용하지 않음" (정확히 이 문구)
   - "사용하지 않음"
   - "미사용"
   - "개발상의 이유로 존재" (이것은 "실제 사용하지 않음"을 의미)

2. ⚠️⚠️⚠️ 절대 포함하지 말 것 (이것들은 실제로 사용되는 데이터/항목임) ⚠️⚠️⚠️:
   - "순수BMS데이터" → 이것은 "사용된다"는 의미임 (절대 포함 금지)
   - "단말 처리상 정의 항목" → 이것은 "사용된다"는 의미임 (절대 포함 금지, "사용하지 않음"이 아님!)
   - "DB상 정의 항목" → 이것은 "사용된다"는 의미임 (절대 포함 금지, "사용하지 않음"이 아님!)
   - "GPS데이터", "GPS데이터 상세해석", "상세해석" → 이것들은 "사용된다"는 의미임 (절대 포함 금지)
   - 비고 필드가 비어있거나 다른 내용 → 포함 금지

⚠️ 매우 중요: "단말 처리상 정의 항목"과 "DB상 정의 항목"은 "실제 사용하지 않음"이 아닙니다. 이것들은 실제로 사용되는 항목입니다!

추론 단계:
1) "테이블구분: BMS"인 모든 변수 찾기
2) 각 변수의 "비고:" 필드를 정확히 확인
3) "비고:" 필드에 "실제 사용하지 않음" 또는 "개발상의 이유로 존재"가 정확히 포함된 변수만 선택
4) 조건에 맞는 변수들을 나열: 변수명, 테이블구분, 설명, 비고 내용 포함
5) 각 변수에 대해 왜 사용되지 않는지 이유를 "비고:" 필드의 내용을 그대로 인용하여 설명

올바른 답변 예시 (반드시 이 형식으로):
- 변수명: seq, 테이블구분: BMS, 설명: 시퀀스, 비고: 실제 사용하지 않음(개발상이유로 존재)
  → 사용되지 않는 이유: 비고 필드에 "실제 사용하지 않음(개발상이유로 존재)"라고 명시되어 있음

⚠️⚠️⚠️ 매우 중요: 참고 문서에 "변수명: seq"와 "비고: 실제 사용하지 않음(개발상이유로 존재)"가 포함된 내용이 있다면, 반드시 이 변수를 답변에 포함해야 합니다. 이 변수를 누락하면 답변이 완전히 잘못된 것입니다.

⚠️⚠️⚠️ 잘못된 답변 예시 (절대 하지 말 것 - 이런 답변은 완전히 잘못됨) ⚠️⚠️⚠️:
- 변수명: measured_month, 테이블구분: BMS, 설명: 데이터 수신 년월, 비고: DB상 정의 항목 (X - "DB상 정의 항목"은 사용되는 항목임! "사용하지 않음"이 아님!)
- 변수명: messageTime, 테이블구분: BMS, 설명: 메시지 전송 시각, 비고: 단말 처리상 정의 항목 (X - "단말 처리상 정의 항목"은 사용되는 항목임! "사용하지 않음"이 아님!)
- 변수명: start_time, 테이블구분: BMS, 설명: 시작시간, 비고: 단말 처리상 정의 항목 (X - "단말 처리상 정의 항목"은 사용되는 항목임! "사용하지 않음"이 아님!)
- 변수명: soc, 테이블구분: BMS, 설명: SOC율, 비고: 순수BMS데이터 (X - "순수BMS데이터"는 사용되는 데이터임! "사용하지 않음"이 아님!)
- "활용되지 않을 가능성이 높습니다" 같은 추측성 답변 (X - 명시적으로 "실제 사용하지 않음"이라고 적힌 변수만 나열)

⚠️ 최종 검증: 답변에 포함된 모든 변수의 비고 필드를 확인하라. "실제 사용하지 않음" 또는 "개발상의 이유로 존재"가 없으면 즉시 제거하라.

⚠️⚠️⚠️ 필수 확인 사항:
1. 참고 문서를 다시 한 번 정확히 확인하라
2. "변수명: seq"와 "비고: 실제 사용하지 않음" 또는 "비고: 실제 사용하지 않음(개발상이유로 존재)"가 포함된 내용이 있는지 확인하라
3. 만약 있다면, 반드시 이 변수를 답변에 포함하라
4. 이 변수를 누락하면 답변이 완전히 잘못된 것이다
---
"""
    
    # System prompt
    system_prompt = f"""너는 EV Battery·BMS·BAAS 전문 어시스턴트다.

⚠️⚠️⚠️ 답변 형식 규칙 (매우 중요) ⚠️⚠️⚠️:
- 질문에 대한 답변만 제공하라. 불필요한 설명, 사족, 추측성 답변은 절대 금지!
- "제공된 정보에서...", "다만...", "따라서 질문의 의도가...", "따라서..." 같은 불필요한 서론/결론 제거!
- 질문에 직접 답하는 내용만 작성하라!
- 예시:
  ❌ 잘못된 답변: "제공된 정보에서 GPS 테이블에 대한 언급은 없으며... 다만... 따라서..."
  ✅ 올바른 답변: "GPS 테이블에 있는 변수 중 사용하지 않는 변수: [변수 목록]"
- 변수 목록을 나열할 때는 변수명만 간단히 나열하거나, 최소한의 설명만 추가하라!

⚠️⚠️⚠️ 매우 중요한 규칙 ⚠️⚠️⚠️
"실제로 사용되지 않는 변수" 질문에 답할 때:
- "비고:" 필드에 "실제 사용하지 않음" 또는 "개발상의 이유로 존재"가 정확히 있는 변수만 답변에 포함
- "비고:" 필드에 "순수BMS데이터", "단말 처리상 정의 항목", "DB상 정의 항목", "GPS데이터", "상세해석"이 있는 변수는 절대 포함하지 말 것 (이것들은 사용되는 데이터임, "사용하지 않음"이 아님!)
- "단말 처리상 정의 항목" = 사용된다는 의미 (사용하지 않음이 아님!)
- "DB상 정의 항목" = 사용된다는 의미 (사용하지 않음이 아님!)
- "활용되지 않을 가능성이 높습니다" 같은 추측성 답변 절대 금지

[답변 단계] 
1) 질문 이해: 질문의 정확한 의미와 의도 파악
2) 문서 분석: 모든 변수의 정보를 정확히 파싱
   - 구조화된 데이터에서 "변수명:", "테이블구분:", "설명:", "비고:" 등 모든 필드를 정확히 파싱
   - 각 변수의 모든 필드 값을 정확히 추출
3) 필터링: 조건에 맞는 변수만 선택
   - "비고:" 필드에 "실제 사용하지 않음" 또는 "개발상의 이유로 존재"가 있는 변수만 선택
   - "순수BMS데이터", "단말 처리상 정의 항목" 등이 있는 변수는 제외
4) 검증: 선택된 모든 변수가 조건에 맞는지 재확인
   - 각 변수의 "비고:" 필드를 다시 확인
   - 조건에 맞지 않는 변수는 즉시 제거
5) 답변 구성: 검증된 변수들을 나열하고 이유 설명
   - 각 변수에 대해 왜 해당 조건에 맞는지 이유를 반드시 설명하라
   - "비고:" 필드의 내용을 바탕으로 구체적인 이유 제시
   - 변수명만 나열하지 말고, 반드시 이유를 포함하라

[참고 문서 구조] 
- EXPANDED_CHUNK: 관련성이 높아 전체 내용을 제공한 청크 (우선 참고)
- COMPRESSED_CHUNK: 관련성이 낮아 요약된 청크 (필요시 참고)
- 구조화된 데이터 형식: "변수명:", "테이블구분:", "설명:", "단위:", "범위:", "호출주기:", "비고:" 등의 필드:값 형식으로 제공됨
- 각 필드를 정확히 파싱하여 활용하라

{"⚠️ 중요: 참고 문서 중 '규칙/필드정의' 타입의 문서는 프로젝트의 공식 규칙 및 데이터 필드 정의입니다. 이 내용을 최우선으로 참고하여 답변하세요." if has_rules else ""}

{"""
[AND 제외 조건 질문 특별 지시사항 - 둘 다 속하지 않는 변수 찾기]
질문 의미: "GPS와 BMS 둘 다 속하지 않는 변수들" = AND 조건

⚠️ 매우 중요: 문서에 나온 "테이블구분:" 값을 그대로 따르라
- 문서에 "테이블구분: BMS"로 나와있으면 그대로 "테이블구분: BMS"로 사용
- 문서에 "테이블구분: GPS"로 나와있으면 그대로 "테이블구분: GPS"로 사용
- 문서에 "테이블구분: 텔레매틱스"로 나와있으면 그대로 "테이블구분: 텔레매틱스"로 사용
- 테이블구분 값을 임의로 변경하거나 추측하지 말 것

1. 질문 의도: "BMS도 아니고 GPS도 아닌 변수들" 찾기
2. 문서에서 각 변수의 "테이블구분:" 필드를 확인
3. "테이블구분: BMS" 또는 "테이블구분: GPS"가 아닌 변수만 선택
4. 선택된 변수들을 나열할 때 문서에 나온 테이블구분 값을 그대로 사용
""" if is_and_exclusion else ""}

{"""
[OR 제외 조건 질문 특별 지시사항 - 하나라도 속하지 않는 변수 찾기]
질문 의미: "GPS 또는 BMS에 속하지 않는 변수들" = OR 조건

⚠️ 매우 중요: 문서에 나온 "테이블구분:" 값을 그대로 따르라
- 문서에 나온 테이블구분 값을 그대로 사용하라
- 테이블구분 값을 임의로 변경하거나 추측하지 말 것

1. 문서에서 각 변수의 "테이블구분:" 필드를 확인
2. "테이블구분: BMS"가 아니거나 "테이블구분: GPS"가 아닌 변수 찾기
3. 조건에 맞는 변수들을 나열 (문서에 나온 테이블구분 값 그대로 사용)
""" if is_or_exclusion else ""}

{"""
[필드 관련 질문 특별 지시사항 - 실제로 사용할지 불분명한 변수 찾기]
질문 의미: "비고 필드에 사용 여부가 불확실하거나 모호하게 적힌 변수들"

⚠️ 매우 중요: 다음 규칙을 정확히 따르라

1. "테이블구분: BMS"인 모든 변수를 찾아라 (질문에 특정 테이블이 명시된 경우)
2. 각 변수의 "비고:" 필드를 정확히 확인하라
   - 비고 필드가 비어있거나
   - "검토 필요", "추가 확인 필요", "불확실", "모호", "불분명" 등의 키워드가 있거나
   - 사용 여부가 명확하지 않은 내용이 있는 변수만 선택
3. 절대 포함하지 말 것:
   - "비고:" 필드에 "실제 사용하지 않음", "사용하지 않음", "미사용"이라고 명시된 변수 (제외)
   - "비고:" 필드에 "사용함", "사용 중"이라고 명시된 변수 (제외)
   - 비고 필드가 명확하게 사용 여부를 표시한 변수 (제외)
4. 조건에 맞는 변수들을 다음 형식으로 나열하라:
   - 변수명: [변수명] (문서에 나온 그대로)
   - 테이블구분: [테이블구분] (문서에 나온 그대로, 임의로 변경하지 말 것)
   - 설명: [설명] (문서에 나온 그대로)
   - 비고: [비고 내용] (비어있거나 불확실한 내용)
5. 중복 답변 금지: 같은 의미의 변수를 여러 번 나열하지 말고, 각 변수를 명확히 구분하여 나열하라
6. 모든 변수를 나열하지 말고, 불분명한 것만 선택하라
""" if is_uncertain_query else ""}

{"""
[필드 관련 질문 특별 지시사항 - 실제 사용하지 않는 변수 찾기 (비고 필드 기준)]
질문 의미: "비고 필드에 '실제 사용하지 않음'이라고 명시적으로 적힌 변수들만"

⚠️⚠️⚠️ 절대 규칙 - 반드시 지켜야 함 ⚠️⚠️⚠️

1. 문서에서 "테이블구분: BMS"인 모든 변수를 찾아라 (질문에 특정 테이블이 명시된 경우)
   ⚠️ 문서에 나온 "테이블구분:" 값을 그대로 사용하라

2. 각 변수의 "비고:" 필드를 정확히 확인하라
   
   ✅ 포함해야 할 변수 (다음 키워드 중 하나라도 정확히 포함된 변수만):
   - "실제 사용하지 않음" (정확히 이 문구가 있어야 함)
   - "사용하지 않음"
   - "미사용"
   - "개발상의 이유로 존재" (이것은 "실제 사용하지 않음"을 의미)
   
   ❌ 절대 포함하지 말 것 (이것들은 실제로 사용되는 데이터/항목임, "사용하지 않음"이 아님!):
   - "순수BMS데이터" → 이것은 "사용된다"는 의미임 (절대 포함 금지, "사용하지 않음"이 아님!)
   - "단말 처리상 정의 항목" → 이것은 "사용된다"는 의미임 (절대 포함 금지, "사용하지 않음"이 아님!)
   - "DB상 정의 항목" → 이것은 "사용된다"는 의미임 (절대 포함 금지, "사용하지 않음"이 아님!)
   - "GPS데이터", "GPS데이터 상세해석", "상세해석", "데이터 상세해석" → 이것들은 "사용된다"는 의미임 (절대 포함 금지, "사용하지 않음"이 아님!)
   - 비고 필드가 비어있거나 다른 내용 → 포함 금지
   - "활용되지 않을 가능성이 높습니다" 같은 추측성 답변 금지
   - "실제 사용 여부가 불분명합니다" 같은 추측성 답변 금지
   - "활용 여부가 불분명합니다" 같은 추측성 답변 금지
   - "실제 사용되지 않는 변수임을 나타냅니다" 같은 잘못된 해석 금지 (단말 처리상 정의 항목, DB상 정의 항목은 사용되는 항목임!)

3. 조건에 맞는 변수들을 다음 형식으로 나열하라:
   - 변수명: [변수명] (문서에 나온 그대로)
   - 테이블구분: [테이블구분] (문서에 나온 그대로, 임의로 변경하지 말 것)
   - 설명: [설명] (문서에 나온 그대로)
   - 비고: [비고 내용] (문서에 나온 그대로, 반드시 "실제 사용하지 않음" 또는 "개발상의 이유로 존재" 포함)
   - → 사용되지 않는 이유: [비고 필드의 내용을 그대로 인용]

4. ⚠️ 매우 중요: 각 변수에 대해 왜 사용되지 않는지 이유를 반드시 설명하라
   - "비고:" 필드의 내용을 그대로 인용
   - 예: "비고 필드에 '실제 사용하지 않음(개발상의 이유로 존재)'라고 명시되어 있음"

5. 중복 답변 금지: 같은 의미의 변수를 여러 번 나열하지 말고, 각 변수를 명확히 구분하여 나열하라

6. "중복 가능성" 같은 모호한 표현 대신, 실제로 사용하지 않는 변수를 구체적으로 나열하고 이유를 설명하라

7. 변수명만 나열하지 말고, 반드시 이유를 포함하라

8. ⚠️⚠️ 최종 검증 필수 (반드시 수행하라) ⚠️⚠️
   답변을 작성한 후, 다음 단계를 반드시 수행하라:
   a) 답변에 포함된 모든 변수의 "비고:" 필드를 하나씩 확인
   b) "실제 사용하지 않음" 또는 "개발상의 이유로 존재"가 없으면 즉시 제거
   c) "순수BMS데이터", "단말 처리상 정의 항목", "DB상 정의 항목"이 있으면 즉시 제거
   d) 남은 변수들만 최종 답변으로 제시

⚠️⚠️⚠️ 검증 실패 예시 (절대 하지 말 것 - 이런 답변은 완전히 잘못됨) ⚠️⚠️⚠️:
- "순수BMS데이터"가 비고에 있는 변수를 답변에 포함 → 이것은 "사용된다"는 의미이므로 즉시 제거하고 다시 작성
- "단말 처리상 정의 항목"이 비고에 있는 변수를 답변에 포함 → 이것은 "사용된다"는 의미임! "사용하지 않음"이 아님! 즉시 제거하고 다시 작성
- "DB상 정의 항목"이 비고에 있는 변수를 답변에 포함 → 이것은 "사용된다"는 의미임! "사용하지 않음"이 아님! 즉시 제거하고 다시 작성
- "GPS데이터", "GPS데이터 상세해석", "상세해석"이 비고에 있는 변수를 답변에 포함 → 이것들은 "사용된다"는 의미이므로 즉시 제거하고 다시 작성
- "실제 사용하지 않음"이 비고에 없는 변수를 답변에 포함 → 즉시 제거하고 다시 작성
- "단말 처리상 정의 항목" 또는 "DB상 정의 항목"을 "실제 사용하지 않음"으로 잘못 해석 → 절대 하지 말 것! 이것들은 사용되는 항목임!
""" if (is_not_used_query and "사용하지" in question_lower and not is_uncertain_query) else ""}

{"""
[필드 관련 질문 특별 지시사항 - 일반]
- 구조화된 필드(변수명, 테이블구분, 설명, 단위, 범위, 호출주기, 비고)를 정확히 파싱하여 활용하라
- 각 변수의 정보를 명확하게 나열하라
""" if (is_field_query and "사용하지" not in question_lower) else ""}


[중요] 답변 시 주의사항:
- ⚠️ 절대 하지 말 것: 문서에 명시되지 않은 변수명을 지어내거나 추측하지 말라
- ⚠️ 절대 하지 말 것: 불완전하거나 이상한 변수명을 포함하지 말라 (예: "신r", "신" 같은 불완전한 변수명)
- ⚠️ 매우 중요: 문서에 나온 "테이블구분:" 값을 그대로 사용하라. 임의로 변경하거나 추측하지 말 것
- 문서에 정확히 나와있는 변수명만 사용하라 (구조화된 데이터에서 "변수명: xxx" 형식으로 정확히 파싱)
- 변수명은 문서의 "변수명: xxx" 형식에서 정확히 추출하라 (파싱 오류로 인한 불완전한 변수명 제외)
- 테이블구분은 문서의 "테이블구분: xxx" 형식에서 정확히 추출하라 (임의로 변경하지 말 것)
- 문서에 없는 정보는 지어내지 말고, "문서에서 해당 내용을 찾을 수 없습니다"라고 답변하라
- 중복되거나 의미가 같은 내용을 반복하지 말라
- 각 변수/항목을 명확히 구분하여 나열하라
- 구조화된 필드 형식을 정확히 인식하고 활용하라
- 답변에 포함된 모든 변수명이 참고 문서에 정확히 나와있는지 확인하라
- 변수명이 한 글자이거나 불완전하면 제외하라

{"""
⚠️⚠️⚠️ 최종 검증 필수 (반드시 수행하라) ⚠️⚠️⚠️

답변을 작성한 후, 다음 단계를 반드시 수행하라:

1. 답변에 포함된 모든 변수의 "테이블구분:" 값을 문서와 대조하여 확인하라
2. 문서에 나온 "테이블구분:" 값을 그대로 사용했는지 확인하라
3. 테이블구분 값을 임의로 변경하거나 추측하지 않았는지 확인하라

⚠️ 매우 중요: 문서에 나온 테이블구분 값을 그대로 사용하라. 임의로 변경하지 말 것.
""" if is_and_exclusion else ""}"""

    # User message
    user_message = f"""{context_text}

[질문]
{request.question}

[답변]"""
    
    messages = [{"role": "user", "content": user_message}]
    
    # Add actual column information for "uncertain usage" queries
    # Add actual column information for "uncertain usage" and "not used" queries
    if is_uncertain_query or is_not_used_query:
        # Determine table type from query
        question_lower_for_table = request.question.lower()
        table_type = "bms" if "bms" in question_lower_for_table and "gps" not in question_lower_for_table else "gps" if "gps" in question_lower_for_table else "bms"
        
        # Get actual columns from preprocessed files (includes specification files)
        file_columns = get_actual_used_columns(table_type)
        if file_columns:
            columns_info = format_columns_for_prompt(file_columns)
            # Add to system prompt
            system_prompt += f"\n\n{columns_info}\n\n"
            
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
            
            # 개선된 프롬프트 구조: 답변 형식 규칙을 최상단에 배치
            system_prompt += """⚠️⚠️⚠️ 답변 형식 규칙 (최우선) ⚠️⚠️⚠️:
- 질문에 대한 답변만 제공하라. 불필요한 설명, 사족, 추측성 답변은 절대 금지!
- "제공된 정보에서...", "왜냐하면...", "다만...", "따라서 질문의 의도가...", "따라서...", "주의:", "추정", "가능성" 같은 표현 절대 금지!
- 질문에 직접 답하는 내용만 작성하라!
- 예시:
  ❌ 잘못된 답변: "제공된 정보에서 사용하지 않는 변수를 정확히 파악하기는 어렵습니다. 왜냐하면... 다만... 따라서... 주의: 위 목록은 추정이며..."
  ✅ 올바른 답변: "BMS 테이블에 있는 변수 중 사용하지 않는 변수: seq"
- 변수 목록을 나열할 때는 변수명만 간단히 나열하거나, 최소한의 설명만 추가하라!

⚠️⚠️⚠️ 변수명 매핑 (답변 전 필수 확인) ⚠️⚠️⚠️:
다음 변수들은 모두 "사용됨"이므로 절대 "사용하지 않는 변수" 목록에 포함하지 말 것!

[절대 포함 금지 변수 목록]
* hvacList1 = hvac_list1 (사용됨)
* hvacList2 = hvac_list2 (사용됨)
* moduleAvgTemp = mod_avg_temp (사용됨)
* emobilitySpeed = emobility_spd (사용됨)
* maxDeteriorationCellNo = max_deter_cell_no (사용됨)
* minDeterioratation = min_deter (사용됨)
* minDeteriorationCellNo = min_deter_cell_no (사용됨)
* socd = socd (사용됨)
* cellVoltageList = cell_volt_list (사용됨, cell_v_* 모두 사용됨)
* insulatedResistance = insul_resistance (사용됨)
* cellVoltageDispersion = cell_volt_dispersion (사용됨)
* driveMotorSpd1 = drive_motor_spd1 (사용됨)
* driveMotorSpd2 = drive_motor_spd2 (사용됨)
* airbagHWireDuty = airbag_hwire_duty (사용됨)
* deviceNo = device_no = DEVICE_NO (사용됨, GPS/BMS 공통)
* messageTime = msg_time = msgTime (사용됨, BMS)

⚠️⚠️⚠️ 변수명 매칭 규칙 ⚠️⚠️⚠️:
변수명이 완전히 같지 않더라도 구성요소가 거의 같으면 같은 변수로 인식:
1. camelCase와 snake_case는 같은 변수 (예: cellVoltageList = cell_volt_list)
2. 대소문자 차이는 무시 (예: DriveMotorSpd1 = drive_motor_spd1)
3. 핵심 단어가 70% 이상 일치하면 같은 변수

⚠️⚠️⚠️ "사용하지 않는 변수" 판단 기준 (3단계) ⚠️⚠️⚠️:
1단계: 규격 파일(bms_specification.csv 또는 gps_specification.csv)에 있는가?
  → 있으면 "사용됨" (절대 "사용하지 않는 변수" 목록에 포함하지 말 것!)

2단계: 위 "절대 포함 금지 변수 목록"에 있는가? (변수명 매핑 확인)
  → 있으면 "사용됨" (절대 "사용하지 않는 변수" 목록에 포함하지 말 것!)

3단계: rules Excel에만 있고, 규격 파일에도 없고, 변수명 매핑에도 없고, 비고 필드에 "실제 사용하지 않음"이 있는가?
  → 모두 만족하면 "사용 안됨" (이것만 "사용하지 않는 변수" 목록에 포함!)

⚠️⚠️⚠️ 테이블별 질문 규칙 ⚠️⚠️⚠️:
[BMS 테이블 질문]
- "BMS 테이블에 있는 변수 중 사용하지 않는 것" 질문 → BMS 변수만 답변!
- GPS 변수(lat, lon, speed, direction 등)는 절대 언급하지 말 것!

[GPS 테이블 질문]
- "GPS 테이블에 있는 변수 중 사용하지 않는 것" 질문 → GPS 변수만 답변!
- BMS 변수(seq, msg_id, pack_volt, soc, cell_v_*, mod_temp_* 등)는 절대 언급하지 말 것!
- ⚠️ 매우 중요: GPS 테이블에는 "실제 사용하지 않음" 비고가 있는 변수가 없습니다!
- 만약 GPS 테이블 질문에서 "실제 사용하지 않음" 비고가 있는 변수를 찾을 수 없다면:
  → "GPS 테이블에 있는 변수 중 사용하지 않는 변수는 없습니다" 또는
  → "GPS 테이블의 모든 변수는 실제로 사용되고 있습니다"라고 명확히 답변하라!
- 절대 "문서에서 해당 내용을 찾을 수 없습니다" 같은 모호한 답변을 하지 말 것!

⚠️⚠️⚠️ 답변 작성 전 최종 체크리스트 ⚠️⚠️⚠️:
답변을 작성하기 전에 반드시 확인:
1. 위 "절대 포함 금지 변수 목록"의 모든 변수를 제외했는가?
2. 변수명 매핑을 확인했는가? (camelCase ↔ snake_case)
3. 규격 파일에 있는 변수를 제외했는가?
4. 불필요한 서론/사족을 제거했는가?
5. 질문에 직접 답하는 내용만 작성했는가?

✅ 올바른 답변 형식:
- 질문에 직접 답하는 변수명만 나열
- 서론/사족 없이 간결하게 작성
- "실제 사용하지 않음" 비고가 있는 변수만 포함

❌ 잘못된 답변 예시:
"제공된 정보에서 사용하지 않는 변수를 정확히 파악하기는 어렵습니다. 왜냐하면... 다만... 따라서... 다음은 잠재적으로 사용되지 않거나 중요도가 낮은 변수 목록입니다: cellVoltageDispersion, airbagHWireDuty, hvacList1, hvacList2... 주의: 위 목록은 추정이며..."

⚠️⚠️⚠️ 테이블 필터링 오류 예시 (절대 하지 말 것) ⚠️⚠️⚠️:
❌ GPS 질문에 BMS 변수 포함 예시:
질문: "GPS 테이블에 존재하는 변수지만, 사용하지 않는 변수는?"
잘못된 답변: "messageTime, seq" (X - 이들은 BMS 변수이지 GPS 변수가 아님!)
올바른 답변: GPS 변수만 나열 (BMS 변수는 절대 포함하지 말 것!)

❌ BMS 질문에 GPS 변수 포함 예시:
질문: "BMS 테이블에 존재하는 변수지만, 사용하지 않는 변수는?"
잘못된 답변: "lat, lon, speed" (X - 이들은 GPS 변수이지 BMS 변수가 아님!)
올바른 답변: BMS 변수만 나열 (GPS 변수는 절대 포함하지 말 것!)

⚠️ 매우 중요: 질문에서 명시한 테이블의 변수만 답변에 포함하라!
- "GPS 테이블" 질문 → GPS 변수만 (BMS 변수 절대 제외)
- "BMS 테이블" 질문 → BMS 변수만 (GPS 변수 절대 제외) """
    
    timing["context_processing"] = time.time() - context_start
    
    # Measure LLM generation time (REFRAG: use chat API)
    llm_start = time.time()
    try:
        answer = await ollama_chat_client.chat(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=512,
            temperature=0.2
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama API error: {e}")
    timing["llm_generation"] = time.time() - llm_start
    
    # 답변 후처리: 서론/사족 제거 및 유사도 기반 필터링
    # 테이블 타입 결정
    question_lower = request.question.lower()
    table_type = "bms" if "bms" in question_lower and "gps" not in question_lower else "gps" if "gps" in question_lower else "bms"
    # "사용하지 않는 변수" 질문인지 확인
    is_not_used_query = (
        any(keyword in question_lower for keyword in ['사용하지 않는', '사용되지 않는', '미사용'])
        and '속하지' not in question_lower and '포함되지' not in question_lower
    )
    answer = postprocess_answer(answer, table_type=table_type, is_not_used_query=is_not_used_query, domain_dict=domain_dict)
    
    # Calculate total time
    timing["total"] = time.time() - start_time
    
    return QueryResponse(answer=answer, contexts=context_list, timing=timing)


@app.post("/ingest")
async def ingest(request: IngestRequest):
    """Ingest new text into the index."""
    if faiss_index is None or embedding_model is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    new_chunks = chunk_text(request.text.strip())
    
    if not new_chunks:
        return {"status": "ok", "added_chunks": 0}
    
    new_embeddings = embedding_model.encode(new_chunks, convert_to_numpy=True)
    faiss.normalize_L2(new_embeddings)
    
    # Acquire write lock for index modification
    async with index_lock:
        faiss_index.add(new_embeddings)
        passages.extend(new_chunks)
        
        # Save index after modification
        save_index(faiss_index, passages)
    
    return {"status": "ok", "added_chunks": len(new_chunks)}


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
    import re
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    filename_lower = file.filename.lower()
    
    # Read content first to check format
    content = await file.read()
    
    # Check if content uses pipe delimiter format
    is_pipe_format = is_pipe_delimited(content)
    
    # Check if filename matches before_preprocess format
    # Pattern: bms.{device_no}.{year}-{month} or gps.{device_no}.{year}-{month}
    bms_pattern = re.match(r"^bms\.(\d+)\.(\d{4}-\d{2})", file.filename, re.IGNORECASE)
    gps_pattern = re.match(r"^gps\.(\d+)\.(\d{4}-\d{2})", file.filename, re.IGNORECASE)
    
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
            base_name = Path(file.filename).stem
            output_filename = f"{base_name}.csv"
        else:
            output_filename = file.filename
        
        output_path = before_preprocess_dir / output_filename
        
        with open(output_path, "wb") as f:
            f.write(content)
        
        return {
            "status": "ok",
            "filename": file.filename,
            "saved_to": str(output_path),
            "message": f"파일이 before_preprocess 디렉토리에 저장되었습니다. (형식: {file_type}, 파이프 구분자: {'예' if is_pipe_format else '아니오'})"
        }
    
    # Handle CSV files (non-pipe format)
    elif filename_lower.endswith('.csv'):
        try:
            # Save to data directory for other CSV files
            data_dir = Config.DATA_DIR
            data_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = data_dir / file.filename
            
            with open(output_path, "wb") as f:
                f.write(content)
            
            return {
                "status": "ok",
                "filename": file.filename,
                "saved_to": str(output_path),
                "message": "CSV 파일이 data 디렉토리에 저장되었습니다."
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error saving CSV file: {e}")
    
    # Handle text files (.txt, .md) - add to RAG index
    elif filename_lower.endswith(('.txt', '.md')):
        if faiss_index is None or embedding_model is None:
            raise HTTPException(status_code=503, detail="Index not loaded")
        
        try:
            text = content.decode("utf-8").strip()
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {e}")
        
        if not text:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Save text file to data directory
        data_dir = Config.DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = data_dir / file.filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Add metadata for ingested text
        metadata = "타입: 사용자 입력 | 형식: 텍스트"
        new_chunks = chunk_text(text, metadata)
        
        if not new_chunks:
            return {
                "status": "ok",
                "filename": file.filename,
                "saved_to": str(output_path),
                "added_chunks": 0
            }
        
        new_embeddings = embedding_model.encode(new_chunks, convert_to_numpy=True)
        faiss.normalize_L2(new_embeddings)
        
        # Acquire write lock for index modification
        async with index_lock:
            faiss_index.add(new_embeddings)
            passages.extend(new_chunks)
            
            # Save index after modification
            save_index(faiss_index, passages)
        
        return {
            "status": "ok",
            "filename": file.filename,
            "saved_to": str(output_path),
            "added_chunks": len(new_chunks)
        }
    else:
        raise HTTPException(status_code=400, detail="Only .txt, .md, and .csv files are supported")


@app.get("/passages")
async def list_passages():
    """List all passages."""
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Acquire read lock for safe access
    async with index_lock:
        passages_snapshot = passages.copy()
    
    return {"passages": passages_snapshot, "count": len(passages_snapshot)}


@app.post("/passages/delete")
async def delete_passages(request: DeletePassageRequest):
    """Delete passages by indices and rebuild index."""
    global faiss_index, passages
    
    if faiss_index is None or embedding_model is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    if not request.indices:
        raise HTTPException(status_code=400, detail="No indices provided")
    
    indices_to_delete = set(request.indices)
    
    # Acquire write lock for index modification
    async with index_lock:
        if any(idx < 0 or idx >= len(passages) for idx in indices_to_delete):
            raise HTTPException(status_code=400, detail="Invalid index")
        
        new_passages = [p for i, p in enumerate(passages) if i not in indices_to_delete]
        
        if len(new_passages) == len(passages):
            return {"status": "ok", "deleted_count": 0, "remaining_count": len(passages)}
        
        new_embeddings = embedding_model.encode(new_passages, convert_to_numpy=True)
        dimension = new_embeddings.shape[1]
        new_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(new_embeddings)
        new_index.add(new_embeddings)
        
        faiss_index = new_index
        passages.clear()
        passages.extend(new_passages)
        
        save_index(faiss_index, passages)
    
    return {
        "status": "ok",
        "deleted_count": len(passages) - len(new_passages),
        "remaining_count": len(new_passages)
    }


@app.post("/passages/delete-all")
async def delete_all_passages():
    """Delete all passages and create empty index."""
    global faiss_index, passages
    
    if faiss_index is None or embedding_model is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Acquire write lock for index modification
    async with index_lock:
        deleted_count = len(passages)
        
        dimension = 384
        if embedding_model:
            dimension = embedding_model.get_sentence_embedding_dimension()
        
        new_index = faiss.IndexFlatIP(dimension)
        passages.clear()
        
        faiss_index = new_index
        save_index(faiss_index, passages)
    
    return {
        "status": "ok",
        "deleted_count": deleted_count,
        "remaining_count": 0
    }


@app.post("/sql/generate")
async def generate_sql(request: SQLRequest):
    """Generate SQL from CSV preview or table name."""
    if request.csv_path:
        result = generate_sql_from_csv_preview(
            request.csv_path, 
            table_name=request.table_name or "data_table"
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
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
        
        return {
            "schema": schema_output,
            "sql": sql_output,
            "full_result": result
        }
    
    elif request.table_name:
        # For DB tables, we'd need db_basic_stats tool
        return {
            "error": "Database table analysis requires db_basic_stats tool (not yet implemented)"
        }
    
    else:
        raise HTTPException(status_code=400, detail="Either csv_path or table_name must be provided")


@app.get("/sql/preview")
def preview_csv(csv_path: str):
    """Preview CSV file and return schema."""
    result = csv_preview(csv_path)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/sql/generate-from-db")
async def generate_sql_from_database(request: DBConnectionRequest):
    """Generate SQL from database table."""
    try:
        result = generate_sql_from_db(
            request.db_url,
            request.table_name,
            request.schema_name
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
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
        
        return {
            "schema": schema_output,
            "sql": sql_output,
            "full_result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/preprocess")
async def preprocess_data():
    """Preprocess all files in before_preprocess directory."""
    try:
        results = preprocess_all_files()
        
        return {
            "status": "ok",
            "processed_count": len(results["processed"]),
            "error_count": len(results["errors"]),
            "processed": results["processed"],
            "errors": results["errors"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")