"""Meta-Cognitive Orchestration Pipeline Engine.

Executes the dual-loop multi-agent architecture with dynamic backtracking,
epistemic discrepancy detection, and deterministic verification.
"""

from __future__ import annotations

from typing import Optional, Tuple
from metacog_finance.agents.data_agent import DataIngestionAgent
from metacog_finance.agents.decision_agent import DecisionIntelligenceAgent
from metacog_finance.agents.forensic_agent import ForensicAuditorAgent
from metacog_finance.agents.fundamental_agent import FundamentalAnalystAgent
from metacog_finance.core.models import DecisionIntelligenceMemo
from metacog_finance.core.telemetry import TelemetryTracer
from metacog_finance.metacognition.calibrator import ConfidenceCalibrator
from metacog_finance.metacognition.monitor import EpistemicDiscrepancyDetector
from metacog_finance.metacognition.reflection_loop import ReflectionLoopController
from metacog_finance.metacognition.verifier import DeterministicFinancialVerifier
from metacog_finance.orchestration.state import PipelineState


class MetaCognitivePipeline:
    """Coordinates multi-agent financial analytics governed by meta-cognitive supervision."""

    def __init__(
        self,
        max_iterations: int = 2,
        tracer: Optional[TelemetryTracer] = None,
    ):
        self.max_iterations = max_iterations
        self.tracer = tracer or TelemetryTracer()

        # Agents
        self.data_agent = DataIngestionAgent()
        self.fundamental_agent = FundamentalAnalystAgent()
        self.forensic_agent = ForensicAuditorAgent()
        self.decision_agent = DecisionIntelligenceAgent()

        # Meta-Cognitive Supervisors
        self.discrepancy_detector = EpistemicDiscrepancyDetector()
        self.verifier = DeterministicFinancialVerifier()
        self.calibrator = ConfidenceCalibrator()
        self.reflection_controller = ReflectionLoopController(max_iterations=max_iterations)

    def run(
        self, ticker: str, fiscal_year: int = 2023, market_cap: Optional[float] = None
    ) -> Tuple[DecisionIntelligenceMemo, PipelineState]:
        """Runs the complete meta-cognitive pipeline for a given entity."""
        state = PipelineState(ticker=ticker, fiscal_year=fiscal_year, market_cap=market_cap)

        self.tracer.record_event(
            phase="ORCHESTRATION_INIT",
            agent_or_component="PipelineSupervisor",
            action=f"Commencing financial decision intelligence evaluation for {ticker.upper()} (FY{fiscal_year})",
        )

        # ---------------------------------------------------------
        # Phase 1: Data Ingestion & Tabular Statement Ingestion
        # ---------------------------------------------------------
        state.status = "INGESTING_DATA"
        ingest_res = self.data_agent.execute(ticker=ticker, fiscal_year=fiscal_year, market_cap=market_cap)

        state.current_stmt = ingest_res["current_stmt"]
        state.prior_stmt = ingest_res["prior_stmt"]
        state.ratios = ingest_res["ratios"]
        state.verification = ingest_res["verification"]
        state.retriever = ingest_res["retriever"]

        self.tracer.record_event(
            phase="DATA_INGESTION",
            agent_or_component="DataIngestionAgent",
            action=f"Loaded audited statements. Computed {len(state.ratios.model_dump())} ratios. Indexed {ingest_res['chunks_count']} chunks.",
            details={"verification_passed": state.verification.passed, "altman_zone": state.ratios.altman_zone},
        )

        # ---------------------------------------------------------
        # Phase 2: Adversarial Forensic Scrutiny
        # ---------------------------------------------------------
        state.status = "FORENSIC_SCRUTINY"
        forensic_res = self.forensic_agent.execute(
            ticker=ticker,
            stmt=state.current_stmt,
            ratios=state.ratios,
            retriever=state.retriever,
        )
        state.forensic_data = forensic_res

        self.tracer.record_event(
            phase="FORENSIC_SCRUTINY",
            agent_or_component="ForensicAuditorAgent",
            action=f"Forensic audit finished with verdict: {forensic_res['verdict']}. Red flags: {len(forensic_res['red_flags'])}.",
            details={"red_flags": forensic_res["red_flags"]},
        )

        # ---------------------------------------------------------
        # Phase 3: Qualitative Analysis & Meta-Cognitive Reflection Loop
        # ---------------------------------------------------------
        state.status = "COGNITIVE_ANALYSIS"
        iteration = 0
        active_critique = ""

        while iteration <= self.max_iterations:
            iteration += 1
            state.iteration_count = iteration

            # 3a. Fundamental qualitative thesis generation
            fund_res = self.fundamental_agent.execute(
                ticker=ticker,
                stmt=state.current_stmt,
                retriever=state.retriever,
                feedback_critique=active_critique,
            )
            state.fundamental_narrative = fund_res["narrative_analysis"]
            state.verbalized_confidence = fund_res["verbalized_confidence"]
            state.citations_used = fund_res["citations_used"]

            self.tracer.record_event(
                phase=f"CYCLE_{iteration}_COGNITIVE_REASONING",
                agent_or_component="FundamentalAnalystAgent",
                action="Produced qualitative evaluation grounded in SEC citations.",
                confidence=state.verbalized_confidence,
            )

            # 3b. Meta-Cognitive Discrepancy Detection (Narrative vs Quantitative)
            discrepancies = self.discrepancy_detector.detect_discrepancies(
                narrative_text=state.fundamental_narrative,
                stmt=state.current_stmt,
                ratios=state.ratios,
                prior_stmt=state.prior_stmt,
            )

            # 3c. Arithmetic verification of textual claims
            text_pass, text_issues = self.verifier.verify_textual_claims(
                state.fundamental_narrative, state.current_stmt
            )
            arithmetic_pass = state.verification.passed and text_pass

            # Grounding faithfulness score (approx based on citations and verification)
            grounding_score = 0.95 if text_pass else 0.70

            # 3d. Formal Confidence Calibration
            audit = self.calibrator.calibrate(
                cycle_index=iteration,
                verbalized_confidence=state.verbalized_confidence,
                grounding_faithfulness_score=grounding_score,
                verification_score=state.verification.score,
                verification_passed=arithmetic_pass,
                discrepancies=discrepancies,
            )
            state.audit_trace.append(audit)

            self.tracer.record_event(
                phase=f"CYCLE_{iteration}_METACOG_AUDIT",
                agent_or_component="ConfidenceCalibrator",
                action=f"Calibrated confidence: {audit.calibrated_confidence:.2f} (Raw: {audit.verbalized_confidence:.2f}). Discrepancies: {len(discrepancies)}.",
                confidence=audit.calibrated_confidence,
                details={
                    "audit_verdict": audit.audit_verdict,
                    "backtracking_triggered": audit.backtracking_triggered,
                    "rationale": audit.backtracking_rationale,
                },
            )

            # 3e. Evaluate Adaptive Backtracking
            backtrack_plan = self.reflection_controller.formulate_backtracking_plan(
                audit=audit, current_iteration=iteration
            )

            if backtrack_plan and backtrack_plan["action"] == "ITERATIVE_REFINEMENT":
                active_critique = backtrack_plan["critique"]
                self.tracer.record_event(
                    phase=f"CYCLE_{iteration}_BACKTRACKING",
                    agent_or_component="ReflectionLoopController",
                    action=f"Triggering adaptive backtracking. Querying: '{backtrack_plan['secondary_query']}'",
                    details={"critique": active_critique},
                )
                continue  # Repeat loop with updated critique
            else:
                # Loop completed or halted
                break

        # ---------------------------------------------------------
        # Phase 4: Calibrated Decision Synthesis
        # ---------------------------------------------------------
        state.status = "DECISION_SYNTHESIS"
        memo = self.decision_agent.execute(
            ticker=ticker,
            stmt=state.current_stmt,
            ratios=state.ratios,
            fundamental_narrative=state.fundamental_narrative,
            forensic_data=state.forensic_data,
            audit_trace=state.audit_trace,
        )
        state.final_memo = memo
        state.status = "COMPLETED"

        self.tracer.record_event(
            phase="DECISION_SYNTHESIS",
            agent_or_component="DecisionIntelligenceAgent",
            action=f"Final Memo synthesized: {memo.recommendation.value} with {memo.confidence_tier} ({memo.calibrated_confidence*100:.1f}%)",
            confidence=memo.calibrated_confidence,
        )

        return memo, state
