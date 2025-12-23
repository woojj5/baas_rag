"""Compression policy for REFRAG Sense stage."""
from typing import List, Dict, Protocol
from app.models.schemas import RetrievedChunk


class CompressionPolicy(Protocol):
    """Protocol for compression policies."""
    
    def decide(self, retrieved: List[RetrievedChunk]) -> Dict[str, str]:
        """
        Decide compression/expansion for each chunk.
        
        Args:
            retrieved: List of retrieved chunks with scores
            
        Returns:
            Dict mapping chunk_id to "COMPRESS" or "EXPAND"
        """
        ...


class HeuristicCompressionPolicy:
    """Heuristic compression policy: REFRAG-style expand fraction or fixed top-k with improved refinement."""
    
    def __init__(self, max_expanded_chunks: int = 5, expand_frac: float = None, score_threshold: float = 0.5):
        """
        Initialize policy.
        
        Args:
            max_expanded_chunks: Maximum number of chunks to expand (fallback)
            expand_frac: Fraction of chunks to expand (REFRAG-style, 0.0-1.0)
                        If provided, uses fraction-based selection instead of fixed count
            score_threshold: Minimum score threshold for expansion (chunks below this are always compressed)
        """
        self.max_expanded_chunks = max_expanded_chunks
        self.expand_frac = expand_frac
        self.score_threshold = score_threshold
    
    def decide(self, retrieved: List[RetrievedChunk]) -> Dict[str, str]:
        """
        Decide compression/expansion based on relevance score with improved refinement.
        
        REFRAG-style: Uses expand_frac if provided, otherwise uses fixed max_expanded_chunks.
        Improved: Uses score threshold and adaptive selection.
        """
        if not retrieved:
            return {}
        
        # Sort by score (descending)
        sorted_chunks = sorted(retrieved, key=lambda x: x.score, reverse=True)
        
        # Calculate number of chunks to expand
        if self.expand_frac is not None and 0.0 < self.expand_frac <= 1.0:
            num_expand = max(1, int(len(sorted_chunks) * self.expand_frac))
        else:
            num_expand = min(self.max_expanded_chunks, len(sorted_chunks))
        
        # Improved refinement: Use score threshold and adaptive selection
        decisions = {}
        high_score_chunks = []
        medium_score_chunks = []
        low_score_chunks = []
        
        # Categorize chunks by score
        for chunk in sorted_chunks:
            if chunk.score >= self.score_threshold:
                high_score_chunks.append(chunk)
            elif chunk.score >= self.score_threshold * 0.7:
                medium_score_chunks.append(chunk)
            else:
                low_score_chunks.append(chunk)
        
        # Priority: Expand high-score chunks first, then medium-score if needed
        expanded_count = 0
        for chunk in high_score_chunks:
            if expanded_count < num_expand:
                decisions[chunk.chunk.id] = "EXPAND"
                expanded_count += 1
            else:
                decisions[chunk.chunk.id] = "COMPRESS"
        
        # Fill remaining expansion slots with medium-score chunks
        for chunk in medium_score_chunks:
            if expanded_count < num_expand:
                decisions[chunk.chunk.id] = "EXPAND"
                expanded_count += 1
            else:
                decisions[chunk.chunk.id] = "COMPRESS"
        
        # Always compress low-score chunks
        for chunk in low_score_chunks:
            decisions[chunk.chunk.id] = "COMPRESS"
        
        return decisions
    
    def apply_aggressive_compression(self, retrieved: List[RetrievedChunk], current_decisions: Dict[str, str]) -> Dict[str, str]:
        """
        Apply more aggressive compression based on current decisions.
        
        Args:
            retrieved: List of retrieved chunks
            current_decisions: Current compression decisions
            
        Returns:
            Updated compression decisions with more aggressive compression
        """
        # Convert more EXPAND decisions to COMPRESS
        updated_decisions = {}
        expanded_count = sum(1 for v in current_decisions.values() if v == "EXPAND")
        
        # Reduce expanded chunks by 50% (round up to keep at least 1)
        target_expanded = max(1, expanded_count // 2)
        
        # Sort chunks by score and keep only top ones expanded
        sorted_chunks = sorted(retrieved, key=lambda x: x.score, reverse=True)
        expanded_so_far = 0
        
        for chunk in sorted_chunks:
            chunk_id = chunk.chunk.id
            if chunk_id in current_decisions:
                if current_decisions[chunk_id] == "EXPAND" and expanded_so_far < target_expanded:
                    updated_decisions[chunk_id] = "EXPAND"
                    expanded_so_far += 1
                else:
                    updated_decisions[chunk_id] = "COMPRESS"
            else:
                updated_decisions[chunk_id] = "COMPRESS"
        
        return updated_decisions
    
    def apply_moderate_compression(self, retrieved: List[RetrievedChunk], current_decisions: Dict[str, str]) -> Dict[str, str]:
        """
        Apply moderate compression based on current decisions.
        
        Args:
            retrieved: List of retrieved chunks
            current_decisions: Current compression decisions
            
        Returns:
            Updated compression decisions with moderate compression
        """
        # Convert some EXPAND decisions to COMPRESS (reduce by 30%)
        updated_decisions = {}
        expanded_count = sum(1 for v in current_decisions.values() if v == "EXPAND")
        
        # Reduce expanded chunks by 30%
        target_expanded = max(1, int(expanded_count * 0.7))
        
        # Sort chunks by score and keep only top ones expanded
        sorted_chunks = sorted(retrieved, key=lambda x: x.score, reverse=True)
        expanded_so_far = 0
        
        for chunk in sorted_chunks:
            chunk_id = chunk.chunk.id
            if chunk_id in current_decisions:
                if current_decisions[chunk_id] == "EXPAND" and expanded_so_far < target_expanded:
                    updated_decisions[chunk_id] = "EXPAND"
                    expanded_so_far += 1
                else:
                    updated_decisions[chunk_id] = "COMPRESS"
            else:
                updated_decisions[chunk_id] = "COMPRESS"
        
        return updated_decisions


class RLCompressionPolicy:
    """RL-based compression policy (skeleton for future implementation)."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize RL policy.
        
        Args:
            model_path: Path to trained RL model (not implemented yet)
        """
        self.model_path = model_path
        # TODO: Load RL model
    
    def decide(self, retrieved: List[RetrievedChunk]) -> Dict[str, str]:
        """
        Decide compression/expansion using RL model.
        
        Currently falls back to heuristic policy.
        """
        # Fallback to heuristic for now
        heuristic = HeuristicCompressionPolicy(max_expanded_chunks=5)
        return heuristic.decide(retrieved)

