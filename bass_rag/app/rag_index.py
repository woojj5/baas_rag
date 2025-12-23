"""RAG index building and loading utilities."""
import json
from pathlib import Path
from typing import List, Tuple, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

from app.config import Config


def load_rules_files() -> List[Tuple[str, str]]:
    """Load rules files from rules directory (Excel, CSV, TXT, MD)."""
    documents = []
    rules_dir = Config.ROOT / "rules"
    
    if not rules_dir.exists():
        return documents
    
    # Load Excel files - convert to JSON format for structured storage
    for file_path in rules_dir.glob("*.xlsx"):
        try:
            all_sheets = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
            
            excel_data = {
                "파일명": file_path.name,
                "시트": {}
            }
            
            for sheet_name, sheet_df in all_sheets.items():
                if sheet_df.empty:
                    continue
                
                # Clean column names - use first row as header if needed
                first_row = sheet_df.iloc[0]
                header_keywords = ['순서', '변수명', '설명', '단위', '범위', '호출주기', '비고', '테이블구분', 'table', 'field', 'name', 'description']
                first_row_str = ' '.join([str(v) for v in first_row.values if pd.notna(v)]).lower()
                
                if any(keyword in first_row_str for keyword in header_keywords):
                    # Use first row as header
                    sheet_df.columns = [str(col).strip() if pd.notna(col) else f'Column_{i}' for i, col in enumerate(first_row.values)]
                    sheet_df = sheet_df.iloc[1:].reset_index(drop=True)
                
                # Convert DataFrame to structured text format
                rows_text = []
                for idx, row in sheet_df.iterrows():
                    if row.isna().all():
                        continue
                    
                    # Skip '순서' column and format in desired order
                    row_parts = []
                    
                    # Define field order (excluding 순서)
                    field_order = ['변수명', '테이블구분', '설명', '단위', '범위', '호출주기', '비고']
                    
                    # First try to match exact column names
                    for field in field_order:
                        # Try to find matching column (case-insensitive, flexible)
                        for col_name in row.index:
                            col_clean = str(col_name).strip()
                            if field.lower() in col_clean.lower() or col_clean.lower() in field.lower():
                                value = row[col_name]
                                if pd.notna(value) and str(value).strip():
                                    row_parts.append(f"{field}: {str(value).strip()}")
                                break
                    
                    # Add any remaining columns that weren't in the field_order (except 순서)
                    for col_name, value in row.items():
                        col_clean = str(col_name).strip()
                        # Skip 순서 and already processed fields
                        if '순서' in col_clean or any(field in col_clean for field in field_order):
                            continue
                        if pd.notna(value) and str(value).strip():
                            row_parts.append(f"{col_clean}: {str(value).strip()}")
                    
                    if row_parts:
                        rows_text.append("\n".join(row_parts))
                
                if rows_text:
                    excel_data["시트"][sheet_name] = rows_text
            
            # Convert to structured text format for RAG
            if excel_data["시트"]:
                content_parts = [f"[파일: {file_path.name}]\n"]
                content_parts.append("=== 데이터 필드 정의 ===\n")
                
                for sheet_name, rows_text in excel_data["시트"].items():
                    content_parts.append(f"\n[시트: {sheet_name}]\n")
                    for row_text in rows_text:
                        content_parts.append(row_text)
                        content_parts.append("")
                
                content = "\n".join(content_parts).strip()
                relative_path = file_path.relative_to(Config.ROOT)
                documents.append((f"rules/{relative_path.name}", content))
        except Exception as e:
            print(f"Warning: Could not load Excel file {file_path}: {e}")
            continue
    
    # Load CSV files - convert to structured format
    for file_path in rules_dir.glob("*.csv"):
        try:
            df = pd.read_csv(file_path)
            
            # Convert to structured text format
            content_parts = [f"[파일: {file_path.name}]\n"]
            content_parts.append("=== 데이터 ===\n")
            
            for idx, row in df.iterrows():
                if row.isna().all():
                    continue
                
                row_parts = [f"  행 {idx+1}:"]
                for col_name, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        col_clean = str(col_name).strip()
                        val_clean = str(value).strip()
                        row_parts.append(f"    {col_clean}: {val_clean}")
                
                if len(row_parts) > 1:
                    content_parts.append("\n".join(row_parts))
                    content_parts.append("")
            
            if len(content_parts) > 2:  # More than just header
                content = "\n".join(content_parts).strip()
                relative_path = file_path.relative_to(Config.ROOT)
                documents.append((f"rules/{relative_path.name}", content))
        except Exception as e:
            print(f"Warning: Could not load CSV file {file_path}: {e}")
            continue
    
    # Load text files
    for ext in ["*.txt", "*.md"]:
        for file_path in rules_dir.glob(ext):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        relative_path = file_path.relative_to(Config.ROOT)
                        documents.append((f"rules/{relative_path.name}", content))
            except Exception:
                continue
    
    return documents


def load_text_files() -> List[Tuple[str, str]]:
    """Load all .txt and .md files from DATA_DIR (recursive)."""
    documents = []
    
    if not Config.DATA_DIR.exists():
        return documents
    
    for ext in ["*.txt", "*.md"]:
        for file_path in Config.DATA_DIR.rglob(ext):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        relative_path = file_path.relative_to(Config.DATA_DIR)
                        documents.append((str(relative_path), content))
            except Exception:
                continue
    
    return documents


def chunk_text(text: str, metadata: str = "") -> List[str]:
    """Split text into overlapping chunks with metadata."""
    if len(text) <= Config.CHUNK_SIZE:
        if metadata:
            return [f"[메타데이터: {metadata}]\n\n{text}"]
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + Config.CHUNK_SIZE
        chunk = text[start:end]
        
        # Add metadata to each chunk
        if metadata:
            chunk = f"[메타데이터: {metadata}]\n\n{chunk}"
        
        chunks.append(chunk)
        
        if end >= len(text):
            break
        
        start = end - Config.CHUNK_OVERLAP
    
    return chunks


def build_passages(use_summaries: bool = True) -> Tuple[List[str], Dict[str, str]]:
    """
    Load all files and chunk them into passages with metadata.
    Optionally creates summaries for cleaner retrieval.
    
    Args:
        use_summaries: If True, create summaries for documents (default: True)
        
    Returns:
        Tuple of (passages, doc_id_to_original_mapping)
        - passages: List of passage strings (summaries if use_summaries=True)
        - doc_id_to_original_mapping: Dict mapping doc_id to original content
    """
    documents = []
    
    # Load rules files first (higher priority)
    rules_docs = load_rules_files()
    documents.extend(rules_docs)
    
    # Load data directory files
    data_docs = load_text_files()
    documents.extend(data_docs)
    
    # Create summaries if requested
    doc_id_to_original = {}
    if use_summaries:
        from app.document_summarizer import create_document_summaries
        summary_docs, doc_id_to_original = create_document_summaries(documents)
        # Use summaries for indexing
        documents = summary_docs
    else:
        # Store original content mapping
        for doc_id, content in documents:
            doc_id_to_original[doc_id] = content
    
    passages = []
    
    for doc_id, content in documents:
        # Extract metadata from doc_id
        metadata_parts = []
        if "rules/" in doc_id:
            metadata_parts.append("타입: 규칙/필드정의")
            if ".xlsx" in doc_id:
                metadata_parts.append("형식: Excel")
            elif ".csv" in doc_id:
                metadata_parts.append("형식: CSV")
        else:
            metadata_parts.append("타입: 문서")
            if doc_id.endswith(".txt"):
                metadata_parts.append("형식: 텍스트")
            elif doc_id.endswith(".md"):
                metadata_parts.append("형식: 마크다운")
        
        # Mark if it's a summary
        if doc_id.endswith("_summary"):
            metadata_parts.append("요약: true")
        
        metadata = " | ".join(metadata_parts)
        
        chunks = chunk_text(content, metadata)
        passages.extend(chunks)
    
    return passages, doc_id_to_original


def build_index(use_summaries: bool = True) -> Tuple[faiss.Index, List[str], SentenceTransformer, Dict[str, str]]:
    """
    Build FAISS index from passages.
    
    Args:
        use_summaries: If True, use document summaries for indexing (default: True)
        
    Returns:
        Tuple of (index, passages, model, doc_id_to_original_mapping)
    """
    passages, doc_id_to_original = build_passages(use_summaries=use_summaries)
    
    if not passages:
        raise ValueError("No passages found. Add documents to data/ directory.")
    
    model = SentenceTransformer(Config.EMBED_MODEL_NAME)
    embeddings = model.encode(passages, show_progress_bar=False, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save embeddings for future use (optimization)
    save_index(index, passages, doc_id_to_original, embeddings)
    
    return index, passages, model, doc_id_to_original


def save_index(
    index: faiss.Index, 
    passages: List[str], 
    doc_id_to_original: Dict[str, str] = None,
    embeddings: np.ndarray = None
) -> None:
    """
    Save FAISS index and passages to disk.
    
    Args:
        index: FAISS index
        passages: List of passage strings
        doc_id_to_original: Optional mapping of doc_id to original content
        embeddings: Optional numpy array of embeddings to save (for optimization)
    """
    faiss_path = Config.INDEX_DIR / "faiss.index"
    passages_path = Config.INDEX_DIR / "passages.json"
    model_name_path = Config.INDEX_DIR / "model_name.txt"
    original_mapping_path = Config.INDEX_DIR / "doc_id_to_original.json"
    embeddings_path = Config.INDEX_DIR / "embeddings.npy"
    
    faiss.write_index(index, str(faiss_path))
    
    with open(passages_path, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)
    
    with open(model_name_path, "w", encoding="utf-8") as f:
        f.write(Config.EMBED_MODEL_NAME)
    
    # Save original content mapping if provided
    if doc_id_to_original:
        with open(original_mapping_path, "w", encoding="utf-8") as f:
            json.dump(doc_id_to_original, f, ensure_ascii=False, indent=2)
    
    # Save embeddings if provided (for optimization - avoid re-embedding)
    if embeddings is not None:
        np.save(str(embeddings_path), embeddings)


def load_index() -> Tuple[faiss.Index, List[str], str, Dict[str, str], np.ndarray]:
    """
    Load FAISS index and passages from disk.
    
    Returns:
        Tuple of (index, passages, model_name, doc_id_to_original_mapping, embeddings)
        embeddings may be None if not saved
    """
    faiss_path = Config.INDEX_DIR / "faiss.index"
    passages_path = Config.INDEX_DIR / "passages.json"
    model_name_path = Config.INDEX_DIR / "model_name.txt"
    original_mapping_path = Config.INDEX_DIR / "doc_id_to_original.json"
    embeddings_path = Config.INDEX_DIR / "embeddings.npy"
    
    if not faiss_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {faiss_path}. "
            "Run 'python build_index.py' to create the index."
        )
    
    if not passages_path.exists():
        raise FileNotFoundError(
            f"Passages file not found at {passages_path}. "
            "Run 'python build_index.py' to create the index."
        )
    
    index = faiss.read_index(str(faiss_path))
    
    with open(passages_path, "r", encoding="utf-8") as f:
        passages = json.load(f)
    
    with open(model_name_path, "r", encoding="utf-8") as f:
        model_name = f.read().strip()
    
    # Load original content mapping if exists
    doc_id_to_original = {}
    if original_mapping_path.exists():
        try:
            with open(original_mapping_path, "r", encoding="utf-8") as f:
                doc_id_to_original = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load original content mapping: {e}")
    
    # Load embeddings if available (for optimization)
    embeddings = None
    embeddings_path = Config.INDEX_DIR / "embeddings.npy"
    if embeddings_path.exists():
        try:
            embeddings = np.load(str(embeddings_path))
            # Verify embeddings match passages count
            if len(embeddings) != len(passages):
                embeddings = None  # Mismatch - will need to re-embed
        except Exception:
            embeddings = None  # Failed to load - will need to re-embed
    
    return index, passages, model_name, doc_id_to_original, embeddings

