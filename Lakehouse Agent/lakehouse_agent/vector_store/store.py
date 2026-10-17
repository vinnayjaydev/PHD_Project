"""
Financial Vector Store: Hybrid Semantic Retrieval Engine
Indexes unstructured financial disclosures, SEC filings, macro statements, and research notes.
Supports cosine similarity search, metadata filtering, and provenance citation tracking.
"""
import numpy as np
from typing import Any, Dict, List, Optional
from lakehouse_agent.vector_store.embeddings import UnifiedEmbeddings
from lakehouse_agent.lakehouse.engine import LakehouseEngine


class FinancialVectorStore:
    """In-memory and Parquet-backed vector store for financial document chunks."""

    def __init__(self, embeddings: Optional[UnifiedEmbeddings] = None):
        self.embeddings = embeddings or UnifiedEmbeddings()
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Chunks and indexes documents with metadata."""
        new_chunks = []
        for doc in documents:
            text = doc.get("content", "")
            # Simple semantic paragraph/sentence chunker
            paragraphs = [p.strip() for p in text.split(". ") if len(p.strip()) > 30]
            if not paragraphs:
                paragraphs = [text]

            for i, p in enumerate(paragraphs):
                chunk_id = f"{doc.get('doc_id', 'DOC')}_p{i}"
                chunk_data = {
                    "chunk_id": chunk_id,
                    "doc_id": doc.get("doc_id"),
                    "ticker": doc.get("ticker", "GENERAL"),
                    "title": doc.get("title", ""),
                    "doc_type": doc.get("doc_type", "GENERAL"),
                    "source": doc.get("source", "Research"),
                    "published_at": doc.get("published_at", ""),
                    "text": p,
                }
                new_chunks.append(chunk_data)

        if not new_chunks:
            return

        texts = [c["text"] for c in new_chunks]
        new_vectors = np.array(self.embeddings.get_embeddings(texts), dtype=np.float32)

        if self.vectors is None or len(self.chunks) == 0:
            self.chunks = new_chunks
            self.vectors = new_vectors
        else:
            self.chunks.extend(new_chunks)
            self.vectors = np.vstack([self.vectors, new_vectors])

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_ticker: Optional[str] = None,
        filter_doc_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic cosine similarity search over indexed chunks,
        applying metadata filters where requested.
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        query_vec = np.array(self.embeddings.get_embedding(query), dtype=np.float32)
        # Compute cosine similarities (both are unit-normalized)
        norm_q = np.linalg.norm(query_vec)
        if norm_q > 0:
            query_vec = query_vec / norm_q

        norms_v = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms_v[norms_v == 0] = 1.0
        normalized_vectors = self.vectors / norms_v

        scores = np.dot(normalized_vectors, query_vec)

        # Apply filtering
        candidates = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            if filter_ticker and chunk["ticker"] != filter_ticker and chunk["ticker"] != "MACRO" and chunk["ticker"] != "MULTI":
                continue
            if filter_doc_type and chunk["doc_type"] != filter_doc_type:
                continue

            candidates.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "ticker": chunk["ticker"],
                "title": chunk["title"],
                "doc_type": chunk["doc_type"],
                "source": chunk["source"],
                "published_at": chunk["published_at"],
                "text": chunk["text"],
                "similarity_score": round(float(score), 4),
                "citation": f"[{chunk['source']} | {chunk['title']} (Ref: {chunk['chunk_id']})]"
            })

        # Sort by similarity score descending
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_k]

    def sync_from_lakehouse(self, engine: LakehouseEngine):
        """Loads and indexes unstructured documents from the Lakehouse Bronze layer."""
        try:
            docs = engine.query_dicts("SELECT * FROM bronze_raw_documents;")
            self.add_documents(docs)
            return len(self.chunks)
        except Exception as e:
            return 0
