"""
Qualitative RAG Agent: Unstructured Financial Filings & Earnings Call Intelligence.
Employs Dense-Sparse Hybrid Retrieval, Temporal Causality Guardrails & Citation Grounding.
Part of the FLAME-AI Financial Agentic Framework.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from .base_agent import BaseFinancialAgent
from rag.hybrid_retriever import HybridFinancialRetriever
from rag.temporal_guardrail import TemporalGuardrail
from rag.evaluation_metrics import RAGEvaluationSuite

logger = logging.getLogger("QualitativeRAGAgent")


class QualitativeRAGAgent(BaseFinancialAgent):
    """
    Retrieves and analyzes unstructured narrative sections from SEC 10-K, 10-Q,
    and Earnings Call transcripts to evaluate qualitative risk factors, competitive moats,
    and management guidance tone with verified citation tracing.
    """

    def __init__(self, weight: float = 0.25, llm_provider: str = "deterministic", api_key: Optional[str] = None):
        super().__init__(
            name="QualitativeRAGAgent",
            role="Unstructured Financial Filings & Earnings Call RAG",
            weight=weight,
            llm_provider=llm_provider,
            api_key=api_key
        )
        self.retriever = HybridFinancialRetriever()
        self.guardrail = TemporalGuardrail(strict_mode=True)
        self.eval_suite = RAGEvaluationSuite()

    def analyze(self, ticker: str, as_of_date: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.trajectory_log.clear()
        query = f"{ticker} competitive advantages, AI infrastructure demand, risk factors, and guidance"

        self.log_trajectory_step("THOUGHT", {
            "thought": f"Executing hybrid point-in-time retrieval on SEC filings for {ticker} as of {as_of_date}."
        })

        t0 = time.time()
        # 1. Execute Hybrid Retrieval
        raw_chunks = self.retriever.search(
            query=query,
            as_of_date=as_of_date,
            ticker=ticker,
            top_k=4,
            method="rrf"
        )
        retrieval_ms = (time.time() - t0) * 1000

        # 2. Audit Temporal Causality (Anti-Lookahead Bias Guardrail)
        is_valid, integrity_score, valid_chunks = self.guardrail.audit_retrieved_context(raw_chunks, as_of_date)

        self.log_trajectory_step("OBSERVATION", {
            "retrieved_chunks_count": len(valid_chunks),
            "temporal_integrity_score": integrity_score,
            "retrieval_ms": round(retrieval_ms, 2)
        })

        # 3. Qualitative analysis of content
        citations = []
        contexts_text = []
        for c in valid_chunks:
            citations.append(c["citation"])
            contexts_text.append(c["content"])

        combined_text = " ".join(contexts_text).lower()

        # Risk and moat keyword analysis
        risk_keywords = ["risk", "adversely", "scrutiny", "headwinds", "restrictions", "disruption", "antitrust"]
        moat_keywords = ["growth", "record", "accelerated", "pioneered", "leverage", "expansion", "leading"]

        risk_hits = sum(combined_text.count(w) for w in risk_keywords)
        moat_hits = sum(combined_text.count(w) for w in moat_keywords)

        if moat_hits >= risk_hits * 1.5:
            stance = "BULLISH"
            confidence = 0.85
            tone = "Strongly Expansionary Narrative"
            summary_rationale = (
                f"Qualitative disclosures for {ticker} highlight substantial secular demand acceleration "
                f"and strong platform lock-in. Management reports record commercial traction with pricing leverage "
                f"outweighing near-term operational headwinds."
            )
        elif risk_hits > moat_hits:
            stance = "NEUTRAL"
            confidence = 0.70
            tone = "Elevated Qualitative Headwinds"
            summary_rationale = (
                f"SEC Item 1A disclosures reveal elevated regulatory, geopolitical, or competitive headwinds for {ticker}. "
                f"While core product segments remain profitable, risk factors warrant cautious allocation sizing."
            )
        else:
            stance = "BULLISH"
            confidence = 0.78
            tone = "Balanced Growth Profile"
            summary_rationale = (
                f"Filings reflect solid competitive execution for {ticker} with diversified revenue streams "
                f"and manageable risk exposure in core product lines."
            )

        # 4. RAG Quality Evaluation
        t1 = time.time()
        gen_ms = (time.time() - t1) * 1000
        rag_metrics = self.eval_suite.evaluate(
            query=query,
            response_text=summary_rationale,
            retrieved_contexts=contexts_text,
            retrieval_ms=retrieval_ms,
            generation_ms=gen_ms
        )

        self.log_trajectory_step("DECISION", {
            "stance": stance,
            "confidence": round(confidence, 3),
            "rag_faithfulness": rag_metrics["faithfulness"],
            "rag_answer_relevance": rag_metrics["answer_relevance"]
        })

        return {
            "agent": self.name,
            "ticker": ticker,
            "as_of_date": as_of_date,
            "stance": stance,
            "confidence": round(confidence, 3),
            "rationale": summary_rationale,
            "key_metrics": {
                "qualitative_tone": tone,
                "moat_signals_count": moat_hits,
                "risk_signals_count": risk_hits,
                "temporal_integrity_score": integrity_score,
                "rag_faithfulness": rag_metrics["faithfulness"],
                "rag_answer_relevance": rag_metrics["answer_relevance"],
                "rag_context_precision": rag_metrics["context_precision"],
                "rag_composite_score": rag_metrics["rag_composite_score"]
            },
            "citations": citations[:3],
            "retrieved_snippets": [c["content"][:200] + "..." for c in valid_chunks[:2]],
            "trajectory": list(self.trajectory_log)
        }
