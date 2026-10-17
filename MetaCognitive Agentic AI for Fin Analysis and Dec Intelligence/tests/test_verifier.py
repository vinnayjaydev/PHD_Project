"""Unit tests for the Deterministic Financial Verifier."""

from metacog_finance.core.models import FinancialStatement
from metacog_finance.metacognition.verifier import DeterministicFinancialVerifier


def test_balance_sheet_identity_pass():
    stmt = FinancialStatement(
        ticker="TEST",
        fiscal_year=2023,
        total_assets=1000.0,
        total_liabilities=600.0,
        total_shareholders_equity=400.0,
    )
    verifier = DeterministicFinancialVerifier()
    res = verifier.verify_statement_identities(stmt)
    assert res.passed
    assert res.score == 1.0
    assert len(res.violations) == 0


def test_balance_sheet_identity_violation():
    stmt = FinancialStatement(
        ticker="BROKEN",
        fiscal_year=2023,
        total_assets=1000.0,
        total_liabilities=600.0,
        total_shareholders_equity=200.0,  # 600 + 200 = 800 != 1000
    )
    verifier = DeterministicFinancialVerifier()
    res = verifier.verify_statement_identities(stmt)
    assert not res.passed
    assert len(res.violations) == 1
    assert res.violations[0]["rule"] == "Balance_Sheet_Identity"


def test_textual_claim_verification():
    stmt = FinancialStatement(
        ticker="TEST",
        fiscal_year=2023,
        revenue=383285.0,
        net_income=96995.0,
    )
    verifier = DeterministicFinancialVerifier()

    # Accurate text
    accurate_text = "The company generated revenue of $383,285 million and net income of $96,995 million."
    passed, issues = verifier.verify_textual_claims(accurate_text, stmt)
    assert passed
    assert len(issues) == 0

    # Hallucinated text
    hallucinated_text = "The enterprise delivered revenue of $550,000 million and net income of $20,000 million."
    passed_h, issues_h = verifier.verify_textual_claims(hallucinated_text, stmt)
    assert not passed_h
    assert len(issues_h) >= 1
