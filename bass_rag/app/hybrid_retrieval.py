"""Hybrid retrieval: Dense (FAISS) + Sparse (BM25) + Exact Match."""
from typing import List, Tuple, Dict, Set
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import re

from app.domain_dict import DomainDictionary


class HybridRetriever:
    """Hybrid retrieval combining dense, sparse, and exact matching."""
    
    def __init__(self, passages: List[str], domain_dict: DomainDictionary):
        self.passages = passages
        self.domain_dict = domain_dict
        self.bm25_index: BM25Okapi | None = None
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from passages."""
        if not self.passages:
            return
        
        # Tokenize passages for BM25
        tokenized_passages = []
        for passage in self.passages:
            # Simple tokenization: split by whitespace and punctuation
            tokens = re.findall(r'\b\w+\b', passage.lower())
            tokenized_passages.append(tokens)
        
        if tokenized_passages:
            self.bm25_index = BM25Okapi(tokenized_passages)
    
    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25 sparse retrieval (optimized).
        Returns: List of (index, score) tuples.
        """
        if not self.bm25_index:
            return []
        
        # Tokenize query
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        if not query_tokens:
            return []
        
        # Get BM25 scores (optimized - limit to reasonable number)
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices (optimized - use argpartition for faster partial sort)
        if len(scores) > top_k * 2:
            # Use argpartition for faster partial sorting when we have many results
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            # For small arrays, regular argsort is fine
            top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return (index, score) pairs
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return positive scores
                results.append((int(idx), float(scores[idx])))
        
        return results
    
    def exact_search(self, query: str) -> Set[int]:
        """Exact match search using domain dictionary.
        Returns: Set of passage indices with exact matches.
        """
        exact_indices = set()
        exact_matches = self.domain_dict.find_exact_matches(query)
        
        if not exact_matches:
            return exact_indices
        
        query_lower = query.lower()
        
        for match_term, match_type in exact_matches:
            for idx, passage in enumerate(self.passages):
                if idx in exact_indices:
                    continue
                
                passage_lower = passage.lower()
                
                if match_type == "variable_exact" or match_type == "variable_alias":
                    # Look for variable name in passage
                    if f"변수명: {match_term}" in passage or f"변수명:{match_term}" in passage:
                        exact_indices.add(idx)
                elif match_type == "table":
                    # Look for table name in passage
                    if f"테이블구분: {match_term}" in passage or f"테이블구분:{match_term}" in passage:
                        exact_indices.add(idx)
                    # Also check if passage contains variables from this table
                    table_vars = self.domain_dict.table_to_variables.get(match_term, set())
                    for var_name in table_vars:
                        if var_name in passage or var_name.lower() in passage_lower:
                            exact_indices.add(idx)
                            break
        
        return exact_indices
    
    def hybrid_search(
        self,
        query: str,
        dense_results: List[Tuple[int, float]],  # (index, distance) from FAISS
        top_k: int = 10,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.3,
        exact_boost: float = 1.5
    ) -> List[Tuple[int, float, bool]]:
        """Hybrid search combining dense, sparse, and exact matches.
        Returns: List of (index, combined_score, is_exact) tuples, sorted by score.
        """
        # Get sparse results (optimized - limit to top_k)
        sparse_results = self.sparse_search(query, top_k=min(top_k, 20))  # Limit sparse search
        
        # Get exact matches
        exact_indices = self.exact_search(query)
        
        # Normalize scores
        dense_scores = {}
        sparse_scores = {}
        
        # Normalize dense scores (FAISS returns cosine similarity, higher is better)
        if dense_results:
            max_dense = max(score for _, score in dense_results)
            min_dense = min(score for _, score in dense_results)
            dense_range = max_dense - min_dense if max_dense != min_dense else 1.0
            
            for idx, score in dense_results:
                # Normalize to [0, 1]
                normalized = (score - min_dense) / dense_range if dense_range > 0 else 0.5
                dense_scores[idx] = normalized
        
        # Normalize sparse scores (BM25, higher is better)
        if sparse_results:
            max_sparse = max(score for _, score in sparse_results) if sparse_results else 1.0
            min_sparse = min(score for _, score in sparse_results) if sparse_results else 0.0
            sparse_range = max_sparse - min_sparse if max_sparse != min_sparse else 1.0
            
            for idx, score in sparse_results:
                # Normalize to [0, 1]
                normalized = (score - min_sparse) / sparse_range if sparse_range > 0 else 0.5
                sparse_scores[idx] = normalized
        
        # Combine scores
        combined_scores = {}
        
        # Add dense scores
        for idx, score in dense_scores.items():
            combined_scores[idx] = score * dense_weight
        
        # Add sparse scores
        for idx, score in sparse_scores.items():
            if idx in combined_scores:
                combined_scores[idx] += score * sparse_weight
            else:
                combined_scores[idx] = score * sparse_weight
        
        # Boost exact matches
        for idx in exact_indices:
            if idx in combined_scores:
                combined_scores[idx] *= exact_boost
            else:
                combined_scores[idx] = exact_boost * (dense_weight + sparse_weight)
        
        # Sort by score (descending)
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top-k with is_exact flag
        results = []
        for idx, score in sorted_results[:top_k]:
            is_exact = idx in exact_indices
            results.append((idx, score, is_exact))
        
        return results

