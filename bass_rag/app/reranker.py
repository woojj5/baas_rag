"""Lightweight reranker using Cross-Encoder for passage relevance."""
from typing import List, Tuple, Optional
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None

from app.domain_dict import DomainDictionary


class DomainReranker:
    """Domain-specific reranker with Cross-Encoder and domain bonus scoring."""
    
    def __init__(self, domain_dict: DomainDictionary, use_cross_encoder: bool = True):
        self.domain_dict = domain_dict
        self.use_cross_encoder = use_cross_encoder and CROSS_ENCODER_AVAILABLE
        self.cross_encoder: CrossEncoder | None = None
        
        if self.use_cross_encoder:
            try:
                # Use lightweight multilingual model for Korean support
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                print(f"Warning: Could not load CrossEncoder, falling back to domain-only reranking: {e}")
                self.use_cross_encoder = False
    
    def rerank(
        self,
        query: str,
        passages: List[str],
        passage_indices: List[int],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Rerank passages by relevance.
        Returns: List of (index, reranked_score) tuples, sorted by score.
        """
        if not passage_indices:
            return []
        
        # Get passage texts
        passage_texts = [passages[idx] for idx in passage_indices if 0 <= idx < len(passages)]
        if not passage_texts:
            return []
        
        # Calculate base scores
        base_scores = {}
        
        # Cross-Encoder scoring (if available) - optimized with batch processing
        if self.use_cross_encoder and self.cross_encoder:
            try:
                # Create query-passage pairs
                pairs = [[query, passage] for passage in passage_texts]
                
                # Get Cross-Encoder scores (optimized - use batch processing)
                # Dynamic batch size based on number of candidates for better performance
                # Smaller batches for fewer candidates, larger for more
                if len(pairs) <= 5:
                    batch_size = len(pairs)  # Process all at once for small batches
                elif len(pairs) <= 10:
                    batch_size = 8  # Medium batch size
                else:
                    batch_size = 16  # Larger batch size for many candidates
                
                ce_scores = []
                for i in range(0, len(pairs), batch_size):
                    batch = pairs[i:i + batch_size]
                    batch_scores = self.cross_encoder.predict(batch, show_progress_bar=False)
                    ce_scores.extend(batch_scores)
                ce_scores = np.array(ce_scores)
                
                # Normalize to [0, 1]
                ce_scores = np.array(ce_scores)
                if ce_scores.max() != ce_scores.min():
                    ce_scores = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min())
                else:
                    ce_scores = np.ones_like(ce_scores) * 0.5
                
                for i, idx in enumerate(passage_indices):
                    if i < len(ce_scores):
                        base_scores[idx] = float(ce_scores[i])
            except Exception as e:
                print(f"Warning: CrossEncoder prediction failed: {e}")
                # Fall back to domain-only scoring
                for idx in passage_indices:
                    base_scores[idx] = 0.5
        else:
            # No Cross-Encoder, use uniform base scores
            for idx in passage_indices:
                base_scores[idx] = 0.5
        
        # Apply domain-specific bonuses
        final_scores = {}
        query_lower = query.lower()
        
        for idx in passage_indices:
            if idx not in base_scores:
                continue
            
            score = base_scores[idx]
            passage = passages[idx] if 0 <= idx < len(passages) else ""
            
            # Bonus 1: Rules passages (highest priority)
            if "rules/" in passage or "규칙/필드정의" in passage or "[메타데이터: 타입: 규칙" in passage:
                score *= 1.3
            
            # Bonus 2: Exact variable name match
            exact_matches = self.domain_dict.find_exact_matches(query)
            for match_term, match_type in exact_matches:
                if match_type.startswith("variable"):
                    if f"변수명: {match_term}" in passage or f"변수명:{match_term}" in passage:
                        score *= 1.2
                        break
            
            # Bonus 3: Table match
            for match_term, match_type in exact_matches:
                if match_type == "table":
                    if f"테이블구분: {match_term}" in passage or f"테이블구분:{match_term}" in passage:
                        score *= 1.15
                        break
            
            # Bonus 4: Domain term match (SOC, SOH, etc.)
            domain_terms = ["soc", "soh", "pack_volt", "cell_volt", "mod_temp", "odo_km"]
            for term in domain_terms:
                if term in query_lower:
                    # Check if passage contains related domain terms
                    if term in passage.lower() or self._has_domain_synonym(passage, term):
                        score *= 1.1
                        break
            
            # Bonus 5: Structured format (has 변수명, 테이블구분, etc.)
            if "변수명:" in passage and "테이블구분:" in passage:
                score *= 1.05
            
            final_scores[idx] = score
        
        # Sort by final score
        sorted_results = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top-k
        return sorted_results[:top_k]
    
    def _has_domain_synonym(self, passage: str, term: str) -> bool:
        """Check if passage contains synonyms of domain term."""
        synonyms_map = {
            "soc": ["state of charge", "배터리 충전율", "충전 상태"],
            "soh": ["state of health", "배터리 건강도", "건강 상태"],
            "pack_volt": ["pack voltage", "팩 전압"],
            "cell_volt": ["cell voltage", "셀 전압"],
            "mod_temp": ["module temperature", "모듈 온도"],
            "odo_km": ["odometer", "주행거리", "누적 주행거리"]
        }
        
        passage_lower = passage.lower()
        synonyms = synonyms_map.get(term, [])
        
        for synonym in synonyms:
            if synonym.lower() in passage_lower:
                return True
        
        return False

