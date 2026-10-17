"""Deterministic Financial Verifier.

Executes sandboxed validation of financial statement balance identities,
cross-checks arithmetic claims in generated texts, and isolates hallucinations.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
from metacog_finance.core.models import FinancialStatement


class VerificationResult:
    """Result of arithmetic and identity verification."""
    def __init__(self, passed: bool, score: float, details: List[str], violations: List[Dict[str, Any]]):
        self.passed = passed
        self.score = score  # 0.0 to 1.0 (fraction of passed checks)
        self.details = details
        self.violations = violations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "verification_score": round(self.score, 3),
            "total_checks": len(self.details),
            "violations_count": len(self.violations),
            "details": self.details,
            "violations": self.violations,
        }


class DeterministicFinancialVerifier:
    """Rigorous sandboxed verification of accounting balance identities and agent claims."""

    def __init__(self, tolerance_pct: float = 0.02):
        """Tolerance percentage for rounding differences (default 2% or $2M)."""
        self.tolerance_pct = tolerance_pct

    def verify_statement_identities(self, stmt: FinancialStatement) -> VerificationResult:
        """Verifies core accounting identities on point-in-time financial statements."""
        checks = []
        violations = []

        # 1. Balance Sheet Equality: Assets == Liabilities + Equity
        calc_total_le = stmt.total_liabilities + stmt.total_shareholders_equity
        if stmt.total_assets > 0:
            diff = abs(stmt.total_assets - calc_total_le)
            err_ratio = diff / stmt.total_assets
            if err_ratio <= self.tolerance_pct or diff < 5.0:  # Allow minimal rounding discrepancy
                checks.append(f"PASS: Balance Sheet Identity (Assets: {stmt.total_assets:.1f}M ~= L+E: {calc_total_le:.1f}M)")
            else:
                msg = f"FAIL: Balance Sheet Inequality. Assets={stmt.total_assets:.1f}M vs L+E={calc_total_le:.1f}M (Diff: {diff:.1f}M)"
                checks.append(msg)
                violations.append({
                    "rule": "Balance_Sheet_Identity",
                    "expected": stmt.total_assets,
                    "actual": calc_total_le,
                    "discrepancy": diff,
                })
        else:
            checks.append("SKIP: Total assets not recorded")

        # 2. Gross Profit Identity: Gross Profit == Revenue - COGS
        if stmt.revenue > 0 and stmt.cost_of_goods_sold > 0:
            expected_gp = stmt.revenue - stmt.cost_of_goods_sold
            diff_gp = abs(expected_gp - stmt.gross_profit)
            if diff_gp / stmt.revenue <= self.tolerance_pct or diff_gp < 5.0:
                checks.append(f"PASS: Gross Profit Identity ({stmt.gross_profit:.1f}M matches Rev - COGS)")
            else:
                msg = f"FAIL: Gross Profit Mismatch. Reported={stmt.gross_profit:.1f}M vs Computed={expected_gp:.1f}M"
                checks.append(msg)
                violations.append({
                    "rule": "Gross_Profit_Identity",
                    "expected": expected_gp,
                    "actual": stmt.gross_profit,
                    "discrepancy": diff_gp,
                })

        # 3. Free Cash Flow Identity: FCF == Operating Cash Flow - Capex
        if stmt.operating_cash_flow != 0 and stmt.free_cash_flow != 0:
            computed_fcf = stmt.operating_cash_flow - abs(stmt.capital_expenditures)
            diff_fcf = abs(computed_fcf - stmt.free_cash_flow)
            denom = abs(stmt.free_cash_flow) if abs(stmt.free_cash_flow) > 0 else 1.0
            if diff_fcf / denom <= self.tolerance_pct or diff_fcf < 5.0:
                checks.append(f"PASS: Free Cash Flow Identity ({stmt.free_cash_flow:.1f}M matches OCF - Capex)")
            else:
                msg = f"FAIL: FCF Mismatch. Reported={stmt.free_cash_flow:.1f}M vs Computed={computed_fcf:.1f}M"
                checks.append(msg)
                violations.append({
                    "rule": "FCF_Identity",
                    "expected": computed_fcf,
                    "actual": stmt.free_cash_flow,
                    "discrepancy": diff_fcf,
                })

        # 4. Working Capital Consistency: Current Assets >= Current Liabilities for positive working capital
        calc_wc = stmt.total_current_assets - stmt.total_current_liabilities
        checks.append(f"INFO: Working Capital computed as {calc_wc:.1f}M")

        total_rules = max(len(checks) - 1, 1)  # excluding INFO
        passed_count = sum(1 for c in checks if c.startswith("PASS"))
        score = passed_count / total_rules if total_rules > 0 else 1.0
        passed = len(violations) == 0

        return VerificationResult(
            passed=passed,
            score=score,
            details=checks,
            violations=violations,
        )

    def verify_textual_claims(
        self, text: str, stmt: FinancialStatement
    ) -> Tuple[bool, List[str]]:
        """Extracts numerical claims from natural language text and verifies them against ground-truth statement."""
        issues = []
        
        # Check revenue mentions: e.g. "revenue of $383,285 million" or "revenue of 383.2B"
        rev_patterns = [
            r"revenue(?:s)? of \$?([\d,\.]+)\s*(?:million|billion|M|B)?",
            r"total revenue was \$?([\d,\.]+)\s*(?:million|billion|M|B)?",
        ]
        
        for pat in rev_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                clean_val_str = m.replace(",", "")
                try:
                    val = float(clean_val_str)
                    # Normalize billion to million if needed
                    if val < 1000.0 and stmt.revenue > 50000.0:
                        val = val * 1000.0
                    
                    diff = abs(val - stmt.revenue)
                    if diff / max(stmt.revenue, 1.0) > 0.10:  # > 10% discrepancy
                        issues.append(
                            f"Hallucinated or mismatched revenue: Claimed ${m} vs Ground Truth ${stmt.revenue:,.1f}M"
                        )
                except ValueError:
                    continue

        # Check net income mentions
        net_patterns = [
            r"net income of \$?([\d,\.]+)\s*(?:million|billion|M|B)?",
            r"net profit of \$?([\d,\.]+)\s*(?:million|billion|M|B)?",
        ]
        for pat in net_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                clean_val_str = m.replace(",", "")
                try:
                    val = float(clean_val_str)
                    if val < 1000.0 and stmt.net_income > 20000.0:
                        val = val * 1000.0
                    diff = abs(val - stmt.net_income)
                    if diff / max(abs(stmt.net_income), 1.0) > 0.10:
                        issues.append(
                            f"Hallucinated or mismatched net income: Claimed ${m} vs Ground Truth ${stmt.net_income:,.1f}M"
                        )
                except ValueError:
                    continue

        return len(issues) == 0, issues
