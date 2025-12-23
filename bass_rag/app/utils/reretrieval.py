"""Optimized re-retrieval utilities for REFRAG pipeline."""
import re
from typing import List, Set, Tuple
from app.config import Config


def extract_keywords_from_answer(answer: str, max_keywords: int = None) -> List[str]:
    """
    Extract keywords from LLM answer using simple heuristics (no LLM call).
    
    This is optimized for performance - uses regex and pattern matching instead of LLM.
    
    Args:
        answer: LLM-generated answer
        max_keywords: Maximum number of keywords to extract
    
    Returns:
        List of extracted keywords
    """
    if max_keywords is None:
        max_keywords = Config.RERETRIEVAL_MAX_KEYWORDS
    
    keywords = []
    answer_lower = answer.lower()
    
    # Extract variable names (common patterns in Korean domain)
    # Pattern: 변수명, 변수, 필드명 등
    variable_patterns = [
        r'변수명[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'([a-zA-Z_][a-zA-Z0-9_]+)\s*(?:변수|필드)',
        r'([a-z_]+(?:_[a-z_]+)*)',  # snake_case variables
        r'([a-z][a-zA-Z0-9]*)',  # camelCase variables
    ]
    
    for pattern in variable_patterns:
        matches = re.findall(pattern, answer_lower)
        keywords.extend(matches)
    
    # Extract domain-specific terms
    domain_terms = [
        'bms', 'gps', 'soc', 'soh', 'voltage', 'temperature',
        '배터리', '전압', '온도', '수명', '충전', '방전'
    ]
    
    for term in domain_terms:
        if term in answer_lower and term not in keywords:
            keywords.append(term)
    
    # Extract quoted terms or emphasized terms
    quoted = re.findall(r'["\']([^"\']+)["\']', answer)
    keywords.extend(quoted)
    
    # Remove duplicates and common stop words
    stop_words = {'the', 'is', 'are', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'on', 'at'}
    keywords = [kw for kw in keywords if kw.lower() not in stop_words and len(kw) > 2]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw)
            if len(unique_keywords) >= max_keywords:
                break
    
    return unique_keywords


def should_reretrieve(answer: str, original_scores: List[float]) -> bool:
    """
    Determine if re-retrieval should be performed based on answer confidence.
    
    Args:
        answer: LLM-generated answer
        original_scores: Scores from original retrieval
    
    Returns:
        True if re-retrieval should be performed
    """
    if not Config.USE_RERETRIEVAL:
        return False
    
    # Check if answer indicates uncertainty
    uncertainty_phrases = [
        '찾을 수 없습니다', '없습니다', '알 수 없', '확실하지',
        '불확실', '불분명', '모르겠', '확인 필요'
    ]
    
    answer_lower = answer.lower()
    has_uncertainty = any(phrase in answer_lower for phrase in uncertainty_phrases)
    
    # Check if original retrieval scores are low
    avg_score = sum(original_scores) / len(original_scores) if original_scores else 0.0
    low_confidence = avg_score < Config.RERETRIEVAL_MIN_CONFIDENCE
    
    # Re-retrieve if answer is uncertain OR original scores are low
    return has_uncertainty or low_confidence


def build_reretrieval_query(original_query: str, keywords: List[str]) -> str:
    """
    Build a re-retrieval query from original query and extracted keywords.
    
    Args:
        original_query: Original user query
        keywords: Extracted keywords from answer
    
    Returns:
        Combined query for re-retrieval
    """
    if not keywords:
        return original_query
    
    # Combine original query with top keywords
    keyword_str = ' '.join(keywords[:Config.RERETRIEVAL_MAX_KEYWORDS])
    return f"{original_query} {keyword_str}".strip()

