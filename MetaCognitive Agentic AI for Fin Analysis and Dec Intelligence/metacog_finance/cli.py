"""Command-line interface for the Meta-Cognitive Financial Framework."""

from __future__ import annotations

import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from metacog_finance.orchestration.engine import MetaCognitivePipeline

console = Console()


def print_banner():
    banner_text = """[bold cyan]
    __  ___     __            ______                   ______ _
   /  |/  /__  / /_____ _    / ____/___  ____ _       / ____/(_)___
  / /|_/ / _ \/ __/ __ `/   / /   / __ \/ __ `/      / /_   / / __ \\
 / /  / /  __/ /_/ /_/ /   / /___/ /_/ / /_/ /  _   / __/  / / / / /
/_/  /_/\___/\__/\__,_/____\____/\____/\__, /  (_) /_/    /_/_/ /_/
                     /_____/          /____/
[/bold cyan]
[dim]A Meta-Cognitive Agentic Framework for Adaptive Financial Analytics & Decision Intelligence[/dim]
"""
    console.print(banner_text)


def run_cli():
    parser = argparse.ArgumentParser(
        description="Meta-Cognitive Agentic Framework for Adaptive Financial Analytics and Decision Intelligence"
    )
    parser.add_argument("--ticker", "-t", type=str, default="AAPL", help="Stock ticker symbol (e.g. AAPL, TSLA, STRESS)")
    parser.add_argument("--year", "-y", type=int, default=2023, help="Fiscal year to analyze (default: 2023)")
    parser.add_argument("--max-iterations", "-m", type=int, default=2, help="Maximum meta-cognitive backtracking cycles")
    parser.add_argument("--export", "-e", type=str, default="", help="Path to export final decision memo as JSON")

    args = parser.parse_args()
    print_banner()

    console.print(f"[bold green]Initiating evaluation for target:[/bold green] [bold yellow]{args.ticker.upper()}[/bold yellow] (FY{args.year})\n")

    pipeline = MetaCognitivePipeline(max_iterations=args.max_iterations)
    memo, state = pipeline.run(ticker=args.ticker, fiscal_year=args.year)

    # 1. Display Quantitative Ratios Table
    table_ratios = Table(title=f"Audited Quantitative Ratios & Forensic Scores ({args.ticker.upper()} FY{args.year})", show_lines=True)
    table_ratios.add_column("Category", style="cyan", no_wrap=True)
    table_ratios.add_column("Metric", style="magenta")
    table_ratios.add_column("Audited Value", style="green", justify="right")
    table_ratios.add_column("Status / Risk Benchmark", style="yellow")

    r = state.ratios
    table_ratios.add_row("Forensic Health", "Altman Z-Score", f"{r.altman_z_score:.2f}", r.altman_zone)
    table_ratios.add_row("Forensic Health", "Beneish M-Score", f"{r.beneish_m_score:.2f}", f"Manipulation: {r.manipulation_risk}")
    table_ratios.add_row("Earnings Quality", "FCF Conversion", f"{r.fcf_conversion_ratio:.2f}x", "Target >= 0.70x")
    table_ratios.add_row("Earnings Quality", "Accrual Ratio", f"{r.accrual_ratio:.4f}", "Benchmark < 0.05")
    table_ratios.add_row("Profitability", "Gross Margin", f"{r.gross_margin:.1f}%", "-")
    table_ratios.add_row("Profitability", "Operating Margin", f"{r.operating_margin:.1f}%", "-")
    table_ratios.add_row("Returns", "ROIC", f"{r.return_on_invested_capital_roic:.1f}%", "Economic Value Add")
    table_ratios.add_row("Liquidity", "Current Ratio", f"{r.current_ratio:.2f}", "Benchmark >= 1.00")
    table_ratios.add_row("Solvency", "Debt-to-Equity", f"{r.debt_to_equity:.2f}x", "-")

    console.print(table_ratios)
    console.print("")

    # 2. Display Meta-Cognitive Audit & Discrepancies Table
    table_audit = Table(title="Meta-Cognitive Introspection & Discrepancy Audits", show_lines=True)
    table_audit.add_column("Cycle", style="cyan", justify="center")
    table_audit.add_column("Verbalized Conf", justify="right")
    table_audit.add_column("Calibrated Conf", style="bold green", justify="right")
    table_audit.add_column("Discrepancies", justify="center")
    table_audit.add_column("Backtracking?", justify="center")
    table_audit.add_column("Audit Verdict", style="bold magenta")

    for audit in state.audit_trace:
        backtrack_str = "[bold red]YES[/bold red]" if audit.backtracking_triggered else "[green]NO[/green]"
        table_audit.add_row(
            str(audit.cycle_index),
            f"{audit.verbalized_confidence*100:.1f}%",
            f"{audit.calibrated_confidence*100:.1f}%",
            str(len(audit.discrepancies)),
            backtrack_str,
            audit.audit_verdict,
        )

    console.print(table_audit)
    console.print("")

    # 3. Display Discrepancies if any
    all_discrepancies = [d for a in state.audit_trace for d in a.discrepancies]
    if all_discrepancies:
        table_disc = Table(title="Detected Epistemic Discrepancies (Narrative vs Quantitative Reality)", show_lines=True)
        table_disc.add_column("ID", style="cyan")
        table_disc.add_column("Category", style="magenta")
        table_disc.add_column("Severity", style="bold red")
        table_disc.add_column("Contradiction Explanation", style="white")

        for d in all_discrepancies:
            table_disc.add_row(d.discrepancy_id, d.category, d.severity.value, d.explanation)

        console.print(table_disc)
        console.print("")

    # 4. Final Decision Panel
    rec_colors = {
        "STRONG BUY": "bold green",
        "ACCUMULATE": "bold cyan",
        "HOLD / NEUTRAL": "bold yellow",
        "REDUCE / UNDERWEIGHT": "bold orange3",
        "CRITICAL AVOID": "bold red",
    }
    rec_color = rec_colors.get(memo.recommendation.value, "bold white")

    panel_content = f"""
[{rec_color}]RECOMMENDATION: {memo.recommendation.value}[/{rec_color}]
[bold]Calibrated Epistemic Confidence:[/bold] {memo.calibrated_confidence*100:.1f}% ({memo.confidence_tier})
[bold]Evaluation Timestamp:[/bold] {memo.timestamp}

[bold cyan]Executive Thesis:[/bold cyan]
{memo.executive_thesis}

[bold magenta]Prescriptive Capital Allocations:[/bold magenta]
{chr(10).join(f"- {a}" for a in memo.prescriptive_allocations)}

[bold yellow]Scenario Stress Testing:[/bold yellow]
- [bold green]Bull Case:[/bold green] {memo.scenario_analysis.get('Bull_Case', '')}
- [bold blue]Base Case:[/bold blue] {memo.scenario_analysis.get('Base_Case', '')}
- [bold red]Bear Case:[/bold red] {memo.scenario_analysis.get('Bear_Case', '')}
"""
    console.print(Panel(panel_content, title=f"Verified Decision Intelligence Memo: {args.ticker.upper()}", expand=False))

    if args.export:
        import json
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(memo.model_dump(), f, indent=2)
        console.print(f"\n[green]Saved calibrated decision memo to: {args.export}[/green]")


def main():
    run_cli()


if __name__ == "__main__":
    main()
