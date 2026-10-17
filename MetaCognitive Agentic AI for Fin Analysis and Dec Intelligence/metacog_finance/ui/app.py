"""Interactive Gradio Web Dashboard for the Meta-Cognitive Financial Framework."""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple
import gradio as gr
import pandas as pd

from metacog_finance.core.models import DecisionRecommendation
from metacog_finance.orchestration.engine import MetaCognitivePipeline


def run_pipeline_for_ui(
    ticker: str, fiscal_year: int, max_iterations: int
) -> Tuple[str, str, str, str, pd.DataFrame, pd.DataFrame, str, str]:
    """Invokes the meta-cognitive pipeline and formats output components for Gradio."""
    ticker_clean = ticker.strip().upper() if ticker.strip() else "AAPL"
    pipeline = MetaCognitivePipeline(max_iterations=int(max_iterations))
    memo, state = pipeline.run(ticker=ticker_clean, fiscal_year=int(fiscal_year))

    # 1. Recommendation Badge & Confidence
    rec = memo.recommendation.value
    rec_colors = {
        "STRONG BUY": "#10b981",
        "ACCUMULATE": "#06b6d4",
        "HOLD / NEUTRAL": "#f59e0b",
        "REDUCE / UNDERWEIGHT": "#f97316",
        "CRITICAL AVOID": "#ef4444",
    }
    bg_color = rec_colors.get(rec, "#3b82f6")

    badge_html = f"""
    <div style="background-color: {bg_color}; color: white; padding: 18px 24px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; letter-spacing: 1px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
        VERDICT: {rec}
        <div style="font-size: 15px; font-weight: normal; margin-top: 6px; opacity: 0.95;">
            Calibrated Epistemic Confidence: <b>{memo.calibrated_confidence*100:.1f}%</b> ({memo.confidence_tier}) | Analyzed at {memo.timestamp}
        </div>
    </div>
    """

    # 2. Executive Thesis & Scenarios
    thesis_md = f"""### Executive Decision Thesis
{memo.executive_thesis}

#### Prescriptive Capital Allocation Directives:
{chr(10).join(f"- **{a}**" for a in memo.prescriptive_allocations)}

#### Scenario Stress-Testing:
- 🟢 **Bull Case**: {memo.scenario_analysis.get('Bull_Case', 'N/A')}
- 🔵 **Base Case**: {memo.scenario_analysis.get('Base_Case', 'N/A')}
- 🔴 **Bear Case**: {memo.scenario_analysis.get('Bear_Case', 'N/A')}
"""

    # 3. Discrepancy Matrix
    discrepancies = [d for a in state.audit_trace for d in a.discrepancies]
    if discrepancies:
        disc_rows = []
        for d in discrepancies:
            disc_rows.append({
                "ID": d.discrepancy_id,
                "Category": d.category,
                "Severity": d.severity.value,
                "Claim in Narrative": d.claim_in_narrative,
                "Ground Truth Reality": f"{d.ground_truth_metric} = {d.ground_truth_value}",
                "Explanation": d.explanation,
            })
        disc_df = pd.DataFrame(disc_rows)
    else:
        disc_df = pd.DataFrame([{
            "ID": "CLEAN_AUDIT",
            "Category": "None",
            "Severity": "SAFE",
            "Claim in Narrative": "All claims reconciled",
            "Ground Truth Reality": "Identities verified",
            "Explanation": "No narrative-quantitative dissonance detected.",
        }])

    # 4. Meta-Cognitive Audit Table
    audit_rows = []
    for a in state.audit_trace:
        audit_rows.append({
            "Cycle": a.cycle_index,
            "Verbalized Confidence": f"{a.verbalized_confidence*100:.1f}%",
            "Calibrated Confidence": f"{a.calibrated_confidence*100:.1f}%",
            "Grounding Faithfulness": f"{a.grounding_faithfulness_score*100:.1f}%",
            "Arithmetic Pass": "PASSED" if a.numerical_verification_pass else "FAILED",
            "Discrepancies": len(a.discrepancies),
            "Backtracking": "TRIGGERED" if a.backtracking_triggered else "NONE",
            "Verdict": a.audit_verdict,
        })
    audit_df = pd.DataFrame(audit_rows)

    # 5. Ratios & Forensic
    r = state.ratios
    ratio_md = f"""### Audited Ratios & Forensic Health Assessment
- **Altman Z-Score**: `{r.altman_z_score:.2f}` — **{r.altman_zone}**
- **Beneish M-Score**: `{r.beneish_m_score:.2f}` — **Manipulation Risk: {r.manipulation_risk}**
- **Return on Invested Capital (ROIC)**: `{r.return_on_invested_capital_roic:.1f}%`
- **Return on Equity (ROE)**: `{r.return_on_equity_roe:.1f}%`
- **Operating Margin**: `{r.operating_margin:.1f}%` (Gross: `{r.gross_margin:.1f}%`)
- **FCF Conversion**: `{r.fcf_conversion_ratio:.2f}x` Net Income
- **Current Ratio**: `{r.current_ratio:.2f}` (Quick: `{r.quick_ratio:.2f}`)
- **Debt to Equity**: `{r.debt_to_equity:.2f}x`

#### Forensic Auditor Report:
```
{memo.forensic_health_assessment}
```
"""

    # 6. Audited Financial Statements MD
    stmt = state.current_stmt
    stmt_md = f"""### Audited Consolidated Financial Statement (SEC Form 10-K, FY{stmt.fiscal_year})

| Financial Line Item | Reported ($M) | Provenance Citation |
| :--- | :--- | :--- |
| **Revenue** | ${stmt.revenue:,.1f}M | SEC Form 10-K Item 8 |
| **Cost of Goods Sold** | ${stmt.cost_of_goods_sold:,.1f}M | SEC Form 10-K Item 8 |
| **Gross Profit** | ${stmt.gross_profit:,.1f}M | Calculated Identity |
| **Operating Income (EBIT)** | ${stmt.operating_income:,.1f}M | SEC Form 10-K Item 8 |
| **Net Income** | ${stmt.net_income:,.1f}M | Consolidated Operations |
| **Operating Cash Flow** | ${stmt.operating_cash_flow:,.1f}M | Consolidated Cash Flows |
| **Capital Expenditures** | ${stmt.capital_expenditures:,.1f}M | Capex Schedule |
| **Free Cash Flow** | ${stmt.free_cash_flow:,.1f}M | Non-GAAP Reconciliation |
| **Cash & Equivalents** | ${stmt.cash_and_equivalents:,.1f}M | Consolidated Balance Sheet |
| **Total Current Assets** | ${stmt.total_current_assets:,.1f}M | Consolidated Balance Sheet |
| **Total Assets** | ${stmt.total_assets:,.1f}M | Consolidated Balance Sheet |
| **Total Current Liabilities** | ${stmt.total_current_liabilities:,.1f}M | Consolidated Balance Sheet |
| **Long-Term Debt** | ${stmt.long_term_debt:,.1f}M | Consolidated Balance Sheet |
| **Total Shareholders' Equity** | ${stmt.total_shareholders_equity:,.1f}M | Consolidated Balance Sheet |
"""

    # 7. Telemetry & Introspection Event Log
    events = pipeline.tracer.get_summary()["events"]
    telemetry_log = "\n".join(
        f"[{e['timestamp'][11:19]}] [{e['phase']}] {e['agent']} -> {e['action']} "
        + (f"(Conf: {e['confidence']:.2f})" if e.get('confidence') is not None else "")
        for e in events
    )

    # 8. Qualitative Narrative
    narrative_md = f"""### Fundamental Qualitative Thesis & Citations
{state.fundamental_narrative}

#### Cited Filing Chunks:
{chr(10).join(f"- `{c}`" for c in state.citations_used)}
"""

    return (
        badge_html,
        thesis_md,
        ratio_md,
        narrative_md,
        disc_df,
        audit_df,
        stmt_md,
        telemetry_log,
    )


def create_ui() -> gr.Blocks:
    """Builds the Gradio user interface."""
    with gr.Blocks(title="Meta-Cognitive Financial Intelligence Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧠 Meta-Cognitive Agentic Framework for Financial Analytics & Decision Intelligence
            ### Dual-Loop Introspection, Grounded Hybrid SEC RAG & Deterministic Verification
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                ticker_input = gr.Dropdown(
                    label="Target Entity (Stock Ticker)",
                    choices=["AAPL", "TSLA", "STRESS_CASE", "MSFT", "GOOGL", "NVDA"],
                    value="AAPL",
                    allow_custom_value=True,
                )
                year_input = gr.Number(label="Fiscal Year", value=2023, precision=0)
                iterations_input = gr.Slider(
                    label="Max Meta-Cognitive Backtracking Cycles",
                    minimum=1,
                    maximum=3,
                    value=2,
                    step=1,
                )
                run_btn = gr.Button("⚡ Run Meta-Cognitive Decision Engine", variant="primary")

                gr.Markdown(
                    """
                    **How It Works:**
                    1. **System 1 (Task Agents)**: Ingests 10-K filings, parses tables, drafts fundamental narrative.
                    2. **Deterministic Sandbox**: Reconciles balance sheets, calculates Altman Z & Beneish M scores.
                    3. **System 2 (Meta-Cognitive Monitor)**: Scans for narrative-quantitative dissonance.
                    4. **Confidence Calibrator**: Mathematically penalizes verbalized overconfidence.
                    5. **Adaptive Backtracking**: Re-queries SEC footnotes if confidence < 0.70.
                    """
                )

            with gr.Column(scale=3):
                badge_output = gr.HTML(label="Recommendation & Calibrated Confidence")

                with gr.Tabs():
                    with gr.TabItem("📋 Decision Intelligence Memo"):
                        thesis_output = gr.Markdown()

                    with gr.TabItem("🔍 Epistemic Discrepancy Matrix"):
                        gr.Markdown("#### Dissonance Between Qualitative Rhetoric and Audited Numbers")
                        disc_output = gr.DataFrame(wrap=True)

                    with gr.TabItem("🔄 Meta-Cognitive Audit & Calibration"):
                        gr.Markdown("#### Confidence Calibration Trace Across Iterative Cycles")
                        audit_output = gr.DataFrame()

                    with gr.TabItem("📊 Ratios & Forensic Scores"):
                        ratio_output = gr.Markdown()

                    with gr.TabItem("📑 Audited Financial Statements"):
                        stmt_output = gr.Markdown()

                    with gr.TabItem("📝 Fundamental Analysis"):
                        narrative_output = gr.Markdown()

                    with gr.TabItem("📡 Multi-Agent Telemetry Log"):
                        telemetry_output = gr.Textbox(lines=15, label="Introspection & Agent Transition Trace")

        run_btn.click(
            fn=run_pipeline_for_ui,
            inputs=[ticker_input, year_input, iterations_input],
            outputs=[
                badge_output,
                thesis_output,
                ratio_output,
                narrative_output,
                disc_output,
                audit_output,
                stmt_output,
                telemetry_output,
            ],
        )

    return demo


def main():
    app = create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
