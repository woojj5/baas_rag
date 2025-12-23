"""Document summarization for cleaner context retrieval."""
from typing import List, Tuple, Dict
import re
from app.config import Config
from app.ollama_client import generate


def extractive_summary(text: str, max_sentences: int = 3) -> str:
    """
    Create extractive summary by taking first N sentences.
    Simple but effective for structured documents.
    """
    if not text or len(text.strip()) < 50:
        return text
    
    # Split by sentences (Korean and English)
    sentences = re.split(r'[.!?。！？]\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Take first N sentences
    summary_sentences = sentences[:max_sentences]
    summary = '. '.join(summary_sentences)
    
    if len(sentences) > max_sentences:
        summary += "..."
    
    return summary


async def llm_summary(text: str, doc_type: str = "문서") -> str:
    """
    Create LLM-based summary for documents.
    Falls back to extractive if LLM fails.
    """
    if not text or len(text.strip()) < 50:
        return text
    
    # For structured data (rules/field definitions), use extractive
    if "변수명:" in text or "테이블구분:" in text or "rules/" in doc_type:
        return extractive_summary(text, max_sentences=5)
    
    # For long documents, try LLM summarization
    if len(text) > 500:
        try:
            prompt = f"""다음 문서의 핵심 내용을 2-3문장으로 요약하라. 
중요한 정보(변수명, 테이블구분, 설명 등)는 반드시 포함하라.

문서:
{text[:1000]}  # Limit to first 1000 chars for prompt

요약:"""
            
            summary = await generate(prompt)
            if summary and len(summary.strip()) > 20:
                return summary.strip()
        except Exception as e:
            print(f"Warning: LLM summarization failed, using extractive: {e}")
    
    # Fallback to extractive
    return extractive_summary(text, max_sentences=3)


def create_document_summaries(
    documents: List[Tuple[str, str]]
) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """
    Create summaries for all documents.
    
    Args:
        documents: List of (doc_id, content) tuples
        
    Returns:
        Tuple of (summary_documents, doc_id_to_original_mapping)
        - summary_documents: List of (doc_id, summary) tuples
        - doc_id_to_original_mapping: Dict mapping doc_id to original content
    """
    summary_documents = []
    doc_id_to_original = {}
    
    for doc_id, content in documents:
        # Store original content (use original doc_id as key)
        doc_id_to_original[doc_id] = content
        
        # Determine document type
        doc_type = "문서"
        if "rules/" in doc_id:
            doc_type = "규칙/필드정의"
        
        # Create summary based on document type
        if "rules/" in doc_id or "변수명:" in content or "테이블구분:" in content:
            # For structured data, create a more comprehensive summary
            # Include key fields (변수명, 테이블구분, 설명, 비고) in summary
            lines = content.split('\n')
            summary_lines = []
            seen_vars = set()
            current_var_block = []  # Collect all fields for current variable
            
            # Important fields to include in summary
            important_fields = ['변수명:', '테이블구분:', '설명:', '비고:']
            
            for line in lines:
                line_stripped = line.strip()
                
                # If we hit a new variable, save the previous block
                if line_stripped.startswith('변수명:'):
                    if current_var_block:
                        summary_lines.extend(current_var_block)
                        summary_lines.append("")  # Add blank line between variables
                    current_var_block = []
                    var_name = line_stripped.split(':', 1)[1].strip()
                    seen_vars.add(var_name)
                
                # Include important fields for current variable
                if any(line_stripped.startswith(field) for field in important_fields):
                    current_var_block.append(line_stripped)
                
                # Limit total summary size (stop after collecting enough variables)
                if len(seen_vars) >= 50:  # Limit to first 50 variables
                    if current_var_block:
                        summary_lines.extend(current_var_block)
                    break
            
            # Add remaining current block
            if current_var_block:
                summary_lines.extend(current_var_block)
            
            # Add count of variables if many
            if len(seen_vars) > 50:
                summary_lines.append(f"\n... 총 {len(seen_vars)}개 이상의 변수 정의")
            
            summary = '\n'.join(summary_lines) if summary_lines else extractive_summary(content, max_sentences=3)
        else:
            # For regular documents, use extractive summary
            summary = extractive_summary(content, max_sentences=5)
        
        # Add metadata to summary (include original doc_id for lookup)
        summary_with_meta = f"[메타데이터: 문서ID: {doc_id} | 타입: {doc_type} | 요약: true | 원본_문서ID: {doc_id}]\n\n{summary}"
        
        summary_documents.append((f"{doc_id}_summary", summary_with_meta))
    
    return summary_documents, doc_id_to_original


def get_original_content(
    doc_id: str,
    doc_id_to_original: Dict[str, str]
) -> str:
    """
    Get original content for a document ID.
    Handles both summary IDs and original IDs.
    """
    # If it's a summary ID, extract original doc_id
    if doc_id.endswith("_summary"):
        original_doc_id = doc_id[:-8]  # Remove "_summary" suffix
    else:
        original_doc_id = doc_id
    
    return doc_id_to_original.get(original_doc_id, "")

