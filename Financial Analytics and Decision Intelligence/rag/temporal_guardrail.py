"""
Temporal Guardrail: Anti-Lookahead Bias & Information Leakage Auditor.
Strict point-in-time enforcement for academic financial validity.
Part of the FLAME-AI Financial RAG Subsystem.
"""

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TemporalGuardrail")


class TemporalGuardrail:
    """
    Guarantees strict chronological causality:
    Tau_Document <= Tau_Decision <= Tau_Execution
    Prevents lookahead bias and future information leakage in financial backtesting & decision intelligence.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def audit_retrieved_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        decision_date: str
    ) -> Tuple[bool, float, List[Dict[str, Any]]]:
        """
        Audits retrieved chunks against the decision timestamp.
        Returns:
            - is_valid (bool): True if zero future leakage detected
            - temporal_integrity_score (float): 1.0 if perfectly clean, < 1.0 if leakage occurred
            - audited_chunks (list): Cleaned list with any future chunks purged
        """
        if not retrieved_chunks:
            return True, 1.0, []

        decision_dt = self._parse_date(decision_date)
        valid_chunks = []
        violations = []

        for chunk in retrieved_chunks:
            filing_date_str = chunk.get("filing_date") or chunk.get("point_in_time_date")
            chunk_dt = self._parse_date(filing_date_str)

            if chunk_dt <= decision_dt:
                valid_chunks.append(chunk)
            else:
                violations.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "filing_date": filing_date_str,
                    "decision_date": decision_date,
                    "leakage_days": (chunk_dt - decision_dt).days
                })

        total = len(retrieved_chunks)
        leak_count = len(violations)
        integrity_score = (total - leak_count) / float(total) if total > 0 else 1.0

        if leak_count > 0:
            logger.warning(
                f"[TEMPORAL GUARDRAIL ALERT] {leak_count}/{total} chunks violated temporal boundary! "
                f"Integrity Score: {integrity_score:.2f}"
            )
            for v in violations:
                logger.warning(f"  * Future Leakage: {v['chunk_id']} filed {v['filing_date']} > Decision {v['decision_date']}")

            if self.strict_mode:
                return False, integrity_score, valid_chunks

        return True, integrity_score, valid_chunks

    def _parse_date(self, date_str: str) -> datetime:
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d")
        except Exception:
            return datetime.utcnow()


if __name__ == "__main__":
    guard = TemporalGuardrail(strict_mode=True)
    chunks_test = [
        {"chunk_id": "C1", "filing_date": "2023-10-25"},
        {"chunk_id": "C2", "filing_date": "2024-02-15"}
    ]
    is_valid, score, clean = guard.audit_retrieved_context(chunks_test, decision_date="2023-12-31")
    print(f"Audit Result: Valid={is_valid}, Score={score}, Clean Chunks={len(clean)}")
