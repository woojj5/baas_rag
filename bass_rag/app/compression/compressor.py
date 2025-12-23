"""Chunk compression for REFRAG Compress stage."""
from typing import Literal
from app.models.schemas import Chunk


def compress_chunk(chunk: Chunk, mode: Literal["head", "summary"] = "head", max_sentences: int = 1) -> str:
    """
    Compress a chunk while preserving structured data fields.
    
    Args:
        chunk: Chunk to compress
        mode: Compression mode ("head" for first N sentences, "summary" for LLM summary)
        max_sentences: Maximum sentences to keep in "head" mode
        
    Returns:
        Compressed chunk text
    """
    if mode == "head":
        text = chunk.text
        
        # Check if this is structured data (has field: value format)
        structured_fields = ['변수명:', '테이블구분:', '설명:', '단위:', '범위:', '호출주기:', '비고:']
        has_structured_data = any(field in text for field in structured_fields)
        
        if has_structured_data:
            # For structured data, preserve all field: value pairs but limit description
            lines = text.split('\n')
            preserved_lines = []
            description_line = None
            
            for line in lines:
                line_stripped = line.strip()
                # Preserve all structured fields
                if any(line_stripped.startswith(field) for field in structured_fields):
                    preserved_lines.append(line)
                    # Keep description for context but might compress it
                    if line_stripped.startswith('설명:'):
                        description_line = line
                elif line_stripped and not description_line:
                    # Non-structured content (keep first few lines)
                    preserved_lines.append(line)
                    if len(preserved_lines) >= max_sentences * 3:  # More lines for structured data (adjusted for max_sentences=1)
                        break
            
            compressed_text = '\n'.join(preserved_lines)
            if len(lines) > len(preserved_lines):
                compressed_text += "\n..."
        else:
            # Simple head compression for unstructured text
            sentences = text.split('.')
            compressed_text = '.'.join(sentences[:max_sentences])
            if len(sentences) > max_sentences:
                compressed_text += "..."
        
        return f"COMPRESSED_CHUNK[id={chunk.id}, score=N/A]: {compressed_text}"
    
    elif mode == "summary":
        # TODO: LLM-based summarization
        # For now, fall back to head mode
        return compress_chunk(chunk, mode="head", max_sentences=max_sentences)
    
    else:
        raise ValueError(f"Unknown compression mode: {mode}")


def expand_chunk(chunk: Chunk, score: float) -> str:
    """
    Expand a chunk (add header with metadata).
    
    Args:
        chunk: Chunk to expand
        score: Relevance score
        
    Returns:
        Expanded chunk text with header
    """
    return f"EXPANDED_CHUNK[id={chunk.id}, score={score:.3f}]\n{chunk.text}\n"

