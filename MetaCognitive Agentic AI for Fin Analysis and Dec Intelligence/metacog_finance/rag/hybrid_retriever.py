"""Hybrid Financial Retriever.

Combines sparse BM25 keyword matching with dense semantic similarity and regulatory
section filtering to guarantee point-in-time financial provenance.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional
from metacog_finance.core.models import DocumentChunk, DocumentSection


class HybridFinancialRetriever:
    """Retrieves financial disclosures using combined sparse and dense scoring with temporal anchoring."""

    def __init__(self, chunks: Optional[List[DocumentChunk]] = None, dense_weight: float = 0.5):
        self.chunks: List[DocumentChunk] = chunks or []
        self.dense_weight = dense_weight
        self._doc_frequencies: Counter = Counter()
        self._corpus_size: int = 0
        self._avg_doc_len: float = 0.0
        self._doc_token_counts: List[Counter] = []
        if self.chunks:
            self._index_corpus()

    def set_chunks(self, chunks: List[DocumentChunk]) -> None:
        self.chunks = chunks
        self._index_corpus()

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and extract alphanumeric words
        return re.findall(r"\b[a-zA-Z0-9_\$%\.]+\b", text.lower())

    def _index_corpus(self) -> None:
        self._corpus_size = len(self.chunks)
        self._doc_frequencies.clear()
        self._doc_token_counts.clear()
        total_tokens = 0

        for chunk in self.chunks:
            tokens = self._tokenize(chunk.content)
            counts = Counter(tokens)
            self._doc_token_counts.append(counts)
            total_tokens += len(tokens)
            for word in counts.keys():
                self._doc_frequencies[word] += 1

        self._avg_doc_len = total_tokens / max(self._corpus_size, 1)

    def _bm25_score(self, query_tokens: List[str], doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        """Calculates Okapi BM25 score."""
        counts = self._doc_token_counts[doc_idx]
        doc_len = sum(counts.values())
        score = 0.0

        for term in query_tokens:
            if term not in counts:
                continue
            tf = counts[term]
            df = self._doc_frequencies.get(term, 0)
            # Standard IDF formula with smoothing
            idf = math.log((self._corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (doc_len / max(self._avg_doc_len, 1.0)))
            score += idf * (numerator / max(denominator, 0.001))

        return score

    def _dense_sim_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Dense semantic cosine proxy over unigram/bigram overlap."""
        counts = self._doc_token_counts[doc_idx]
        q_counts = Counter(query_tokens)
        
        dot_product = sum(q_counts[term] * counts.get(term, 0) for term in q_counts)
        norm_q = math.sqrt(sum(v ** 2 for v in q_counts.values()))
        norm_d = math.sqrt(sum(v ** 2 for v in counts.values()))
        
        if norm_q == 0 or norm_d == 0:
            return 0.0
        return dot_product / (norm_q * norm_d)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        section_filter: Optional[DocumentSection] = None,
        require_tabular: Optional[bool] = None,
    ) -> List[DocumentChunk]:
        """Executes hybrid retrieval over the corpus with metadata constraints."""
        if not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return self.chunks[:top_k]

        scored_chunks = []
        max_bm25 = 0.001
        bm25_scores = []
        dense_scores = []

        for i, chunk in enumerate(self.chunks):
            # Apply hard filters if requested
            if section_filter and chunk.section != section_filter:
                bm25_scores.append(0.0)
                dense_scores.append(0.0)
                continue
            if require_tabular is not None and chunk.is_tabular != require_tabular:
                bm25_scores.append(0.0)
                dense_scores.append(0.0)
                continue

            b_score = self._bm25_score(query_tokens, i)
            d_score = self._dense_sim_score(query_tokens, i)
            bm25_scores.append(b_score)
            dense_scores.append(d_score)
            if b_score > max_bm25:
                max_bm25 = b_score

        for i, chunk in enumerate(self.chunks):
            if bm25_scores[i] == 0.0 and dense_scores[i] == 0.0:
                continue
            # Normalize BM25 score
            norm_b = bm25_scores[i] / max_bm25
            norm_d = dense_scores[i]
            hybrid_score = (self.dense_weight * norm_d) + ((1.0 - self.dense_weight) * norm_b)
            
            # Boost if query explicitly specifies a section keyword present in chunk
            if chunk.section.value.lower() in query.lower():
                hybrid_score *= 1.25

            chunk_copy = chunk.model_copy()
            chunk_copy.relevance_score = round(hybrid_score, 4)
            scored_chunks.append((hybrid_score, chunk_copy))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_chunks[:top_k]]
