"""Unit tests for deterministic financial metrics and forensic scoring models."""

import pytest
from metacog_finance.core.models import FinancialStatement
from metacog_finance.metacognition.financial_metrics import (
    compute_altman_z_score,
    compute_beneish_m_score,
    compute_financial_ratios,
)


@pytest.fixture
def sample_healthy_stmt():
    return FinancialStatement(
        ticker="HEALTHY",
        fiscal_year=2023,
        revenue=10000.0,
        cost_of_goods_sold=6000.0,
        gross_profit=4000.0,
        operating_income=2500.0,
        net_income=2000.0,
        cash_and_equivalents=3000.0,
        marketable_securities=1000.0,
        accounts_receivable=1500.0,
        total_current_assets=6000.0,
        total_assets=12000.0,
        total_current_liabilities=3000.0,
        short_term_debt=500.0,
        long_term_debt=1500.0,
        total_liabilities=5000.0,
        retained_earnings=4000.0,
        total_shareholders_equity=7000.0,
        operating_cash_flow=2400.0,
        capital_expenditures=400.0,
        free_cash_flow=2000.0,
    )


@pytest.fixture
def sample_prior_stmt():
    return FinancialStatement(
        ticker="HEALTHY",
        fiscal_year=2022,
        revenue=9000.0,
        cost_of_goods_sold=5500.0,
        gross_profit=3500.0,
        operating_income=2200.0,
        net_income=1800.0,
        accounts_receivable=1400.0,
        total_current_assets=5500.0,
        total_assets=11000.0,
        total_current_liabilities=2800.0,
        short_term_debt=500.0,
        long_term_debt=1500.0,
        total_liabilities=4500.0,
        total_shareholders_equity=6500.0,
        operating_cash_flow=2100.0,
        capital_expenditures=350.0,
        free_cash_flow=1750.0,
    )


def test_liquidity_and_margin_ratios(sample_healthy_stmt):
    ratios = compute_financial_ratios(sample_healthy_stmt)
    assert ratios.current_ratio == 2.0  # 6000 / 3000
    assert ratios.quick_ratio == pytest.approx(1.833, 0.01)  # (3000 + 1000 + 1500) / 3000 = 1.833
    assert ratios.gross_margin == 40.0  # 4000 / 10000 * 100
    assert ratios.operating_margin == 25.0  # 2500 / 10000 * 100
    assert ratios.fcf_conversion_ratio == 1.0  # 2000 / 2000


def test_altman_z_score_calculation(sample_healthy_stmt):
    z, zone = compute_altman_z_score(sample_healthy_stmt, market_cap=15000.0)
    assert z > 2.99
    assert "Safe" in zone


def test_beneish_m_score_healthy(sample_healthy_stmt, sample_prior_stmt):
    m, risk = compute_beneish_m_score(sample_healthy_stmt, sample_prior_stmt)
    assert m < -1.78
    assert risk == "Low (<-1.78)"


def test_beneish_m_score_stressed():
    curr = FinancialStatement(
        ticker="DISTRESSED",
        fiscal_year=2023,
        revenue=5000.0,
        gross_profit=500.0,
        operating_income=100.0,
        net_income=300.0,
        accounts_receivable=3000.0,  # Sudden huge receivables spike
        total_current_assets=3500.0,
        total_assets=6000.0,
        operating_cash_flow=-500.0,  # Massive negative accrual divergence
    )
    prior = FinancialStatement(
        ticker="DISTRESSED",
        fiscal_year=2022,
        revenue=5000.0,
        gross_profit=1500.0,
        operating_income=600.0,
        net_income=450.0,
        accounts_receivable=800.0,
        total_current_assets=2000.0,
        total_assets=5000.0,
        operating_cash_flow=400.0,
    )
    m, risk = compute_beneish_m_score(curr, prior)
    assert m > -1.78
    assert risk == "High (>-1.78)"
