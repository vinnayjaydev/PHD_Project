"""
Hybrid Financial Retriever: Dense-Sparse Hybrid Search with Reciprocal Rank Fusion (RRF).
Part of the FLAME-AI Financial RAG Subsystem.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HybridRetriever")


class HybridFinancialRetriever:
    """
    Combines BM25 sparse keyword matching with dense semantic embeddings
    using Reciprocal Rank Fusion (RRF) and convex score interpolation.
    """

    def __init__(
        self,
        silver_chunks_path: str = "data/silver/filings_chunks.parquet",
        alpha: float = 0.55,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75
    ):
        self.silver_chunks_path = silver_chunks_path
        self.alpha = alpha  # Weight for dense score (1-alpha for sparse)
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.chunks_df: Optional[pd.DataFrame] = None
        self.bm25: Optional[BM25Okapi] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tokenized_corpus: List[List[str]] = []

        self._initialize_retriever()

    def _initialize_retriever(self):
        """Loads chunks from Silver lakehouse and indexes BM25 and dense representations."""
        if not os.path.exists(self.silver_chunks_path):
            logger.warning(f"Silver chunks file not found: {self.silver_chunks_path}")
            return

        self.chunks_df = pd.read_parquet(self.silver_chunks_path)
        contents = self.chunks_df["content"].tolist()

        # Tokenize for BM25
        self.tokenized_corpus = [self._tokenize(doc) for doc in contents]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)

        # TF-IDF Dense projection model for consistent query embeddings
        self.tfidf_vectorizer = TfidfVectorizer(max_features=256, stop_words="english")
        self.tfidf_vectorizer.fit(contents)
        logger.info(f"Hybrid Retriever initialized with {len(self.chunks_df)} chunks.")

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z0-9_\$%\.]+\b', text.lower())

    def search(
        self,
        query: str,
        as_of_date: Optional[str] = None,
        ticker: Optional[str] = None,
        section_filter: Optional[str] = None,
        top_k: int = 5,
        method: str = "rrf"  # "rrf" or "convex"
    ) -> List[Dict[str, Any]]:
        """
        Executes point-in-time hybrid search.
        Filters out any documents published strictly AFTER `as_of_date` (no lookahead bias).
        """
        if self.chunks_df is None or len(self.chunks_df) == 0:
            return []

        df = self.chunks_df.copy()

        # 1. Point-in-Time Temporal Filtering (Crucial for PhD thesis)
        if as_of_date:
            df = df[df["filing_date"] <= as_of_date]

        # 2. Metadata filtering
        if ticker:
            df = df[df["ticker"] == ticker]
        if section_filter:
            df = df[df["section"].str.contains(section_filter, case=False, na=False)]

        if len(df) == 0:
            logger.info("No documents passed the temporal/metadata filters.")
            return []

        indices = df.index.tolist()
        query_tokens = self._tokenize(query)

        # 3. Sparse BM25 Scoring
        bm25_raw_scores = self.bm25.get_scores(query_tokens)
        sub_bm25_scores = np.array([bm25_raw_scores[i] for i in indices])

        # Min-max normalize BM25
        if np.max(sub_bm25_scores) > np.min(sub_bm25_scores):
            norm_bm25 = (sub_bm25_scores - np.min(sub_bm25_scores)) / (np.max(sub_bm25_scores) - np.min(sub_bm25_scores) + 1e-9)
        else:
            norm_bm25 = np.ones_like(sub_bm25_scores) * 0.5

        # 4. Dense Semantic Embedding Scoring
        query_vec = self.tfidf_vectorizer.transform([query]).toarray()[0]
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        chunk_embeddings = np.array(df["embedding"].tolist())
        # Cosine similarity (vectors are already unit normalized)
        dense_scores = np.dot(chunk_embeddings, query_vec)
        norm_dense = np.clip((dense_scores + 1.0) / 2.0, 0.0, 1.0)

        # 5. Fusion: Reciprocal Rank Fusion (RRF) or Convex Linear
        n = len(indices)
        if method == "rrf":
            # RRF rank calculation
            bm25_rank = np.argsort(-norm_bm25)
            dense_rank = np.argsort(-norm_dense)

            rrf_score = np.zeros(n)
            k_rrf = 60
            for rank_pos, idx in enumerate(bm25_rank):
                rrf_score[idx] += 1.0 / (k_rrf + rank_pos + 1)
            for rank_pos, idx in enumerate(dense_rank):
                rrf_score[idx] += 1.0 / (k_rrf + rank_pos + 1)

            final_scores = rrf_score
        else:
            # Convex combination
            final_scores = self.alpha * norm_dense + (1.0 - self.alpha) * norm_bm25

        # Rank and build top_k results
        top_indices = np.argsort(-final_scores)[:top_k]
        results = []

        for rank_pos, rank_idx in enumerate(top_indices):
            row = df.iloc[rank_idx]
            results.append({
                "rank": rank_pos + 1,
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "ticker": row["ticker"],
                "company_name": row["company_name"],
                "doc_type": row["doc_type"],
                "filing_date": row["filing_date"],
                "section": row["section"],
                "content": row["content"],
                "is_tabular": bool(row["is_tabular"]),
                "dense_score": float(np.round(norm_dense[rank_idx], 4)),
                "bm25_score": float(np.round(norm_bm25[rank_idx], 4)),
                "hybrid_score": float(np.round(final_scores[rank_idx], 4)),
                "citation": f"[{row['ticker']} {row['doc_type']} {row['filing_date']} § {row['section']}]"
            })

        return results


if __name__ == "__main__":
    retriever = HybridFinancialRetriever()
    res = retriever.search("Blackwell AI GPU demand and supply constraints", ticker="NVDA", as_of_date="2024-05-01")
    for r in res:
        print(f"Rank {r['rank']} | Score: {r['hybrid_score']} | Citation: {r['citation']}")
        print(f"Snippet: {r['content'][:150]}...\n")
