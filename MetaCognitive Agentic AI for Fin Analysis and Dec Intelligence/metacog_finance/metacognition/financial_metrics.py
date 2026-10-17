"""Deterministic Financial Metrics, Ratios, and Forensic Distress/Manipulation Scores.

These functions provide mathematically rigorous financial computations executed deterministically,
guaranteeing zero arithmetic hallucination from language models.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
from metacog_finance.core.models import FinancialRatios, FinancialStatement


def compute_financial_ratios(
    current: FinancialStatement,
    prior: Optional[FinancialStatement] = None,
    market_cap: Optional[float] = None,
) -> FinancialRatios:
    """Computes full suite of liquidity, solvency, profitability, and forensic metrics.
    
    Args:
        current: Point-in-time financial statement for period T.
        prior: Optional point-in-time financial statement for period T-1 (needed for Beneish M-Score).
        market_cap: Optional market capitalization in millions USD (defaults to total equity if absent).
    """
    # 1. Liquidity
    curr_assets = current.total_current_assets
    curr_liab = current.total_current_liabilities if current.total_current_liabilities > 0 else 1.0
    current_ratio = round(curr_assets / curr_liab, 3)
    
    quick_assets = current.cash_and_equivalents + current.marketable_securities + current.accounts_receivable
    quick_ratio = round(quick_assets / curr_liab, 3)
    
    cash_ratio = round((current.cash_and_equivalents + current.marketable_securities) / curr_liab, 3)
    
    # 2. Solvency & Leverage
    equity = current.total_shareholders_equity if current.total_shareholders_equity != 0 else 1.0
    total_debt = current.short_term_debt + current.long_term_debt
    total_assets = current.total_assets if current.total_assets > 0 else 1.0
    
    debt_to_equity = round(total_debt / equity, 3)
    debt_to_assets = round(total_debt / total_assets, 3)
    
    interest_exp = current.interest_expense if current.interest_expense > 0 else 0.001
    interest_coverage = round(current.operating_income / interest_exp, 2)
    
    # 3. Profitability & Margins
    rev = current.revenue if current.revenue > 0 else 1.0
    gross_margin = round((current.gross_profit / rev) * 100, 2)
    operating_margin = round((current.operating_income / rev) * 100, 2)
    net_profit_margin = round((current.net_income / rev) * 100, 2)
    
    roa = round((current.net_income / total_assets) * 100, 2)
    roe = round((current.net_income / equity) * 100, 2)
    
    # Invested Capital = Equity + Total Debt - Cash
    invested_capital = equity + total_debt - current.cash_and_equivalents
    invested_capital = invested_capital if invested_capital > 0 else equity
    # Effective tax rate bounded between 0% and 35%
    tax_rate = (current.tax_expense / current.operating_income) if current.operating_income > 0 else 0.21
    tax_rate = max(0.0, min(0.35, tax_rate))
    nopat = current.operating_income * (1 - tax_rate)
    roic = round((nopat / invested_capital) * 100, 2)
    
    # 4. Cash Flow & Earnings Quality
    net_inc = current.net_income if current.net_income != 0 else 0.001
    fcf = current.free_cash_flow if current.free_cash_flow != 0 else (current.operating_cash_flow - abs(current.capital_expenditures))
    fcf_conversion = round(fcf / net_inc, 3)
    
    # Accrual Ratio = (Net Income - Operating Cash Flow) / Total Assets
    accrual_ratio = round((current.net_income - current.operating_cash_flow) / total_assets, 4)
    
    # 5. Altman Z-Score
    z_score, z_zone = compute_altman_z_score(current, market_cap)
    
    # 6. Beneish M-Score
    m_score, m_risk = compute_beneish_m_score(current, prior)
    
    return FinancialRatios(
        current_ratio=current_ratio,
        quick_ratio=quick_ratio,
        cash_ratio=cash_ratio,
        debt_to_equity=debt_to_equity,
        debt_to_assets=debt_to_assets,
        interest_coverage_ratio=interest_coverage,
        gross_margin=gross_margin,
        operating_margin=operating_margin,
        net_profit_margin=net_profit_margin,
        return_on_assets_roa=roa,
        return_on_equity_roe=roe,
        return_on_invested_capital_roic=roic,
        fcf_conversion_ratio=fcf_conversion,
        accrual_ratio=accrual_ratio,
        altman_z_score=z_score,
        altman_zone=z_zone,
        beneish_m_score=m_score,
        manipulation_risk=m_risk,
    )


def compute_altman_z_score(stmt: FinancialStatement, market_cap: Optional[float] = None) -> Tuple[float, str]:
    """Computes the Altman Z-Score for public companies:
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.999*X5
    
    Where:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Value of Equity / Total Liabilities
    X5 = Sales / Total Assets
    """
    total_assets = stmt.total_assets if stmt.total_assets > 0 else 1.0
    total_liab = stmt.total_liabilities if stmt.total_liabilities > 0 else 1.0
    
    working_capital = stmt.total_current_assets - stmt.total_current_liabilities
    equity_val = market_cap if (market_cap and market_cap > 0) else max(stmt.total_shareholders_equity, 1.0)
    
    x1 = working_capital / total_assets
    x2 = stmt.retained_earnings / total_assets
    x3 = stmt.operating_income / total_assets
    x4 = equity_val / total_liab
    x5 = stmt.revenue / total_assets
    
    z = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (0.999 * x5)
    z = round(z, 2)
    
    if z >= 2.99:
        zone = "Safe (>2.99)"
    elif z >= 1.81:
        zone = "Grey / Caution (1.81-2.99)"
    else:
        zone = "Distress / High Default Risk (<1.81)"
        
    return z, zone


def compute_beneish_m_score(
    curr: FinancialStatement,
    prior: Optional[FinancialStatement] = None,
) -> Tuple[float, str]:
    """Computes the 8-variable Beneish M-Score for earnings manipulation detection.
    
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.037*TATA + 0.0327*LVGI
    Threshold: M > -1.78 indicates high probability of manipulation.
    """
    if prior is None or prior.revenue <= 0 or prior.total_assets <= 0:
        # Default baseline if single period is available
        total_assets = curr.total_assets if curr.total_assets > 0 else 1.0
        tata = (curr.net_income - curr.operating_cash_flow) / total_assets
        # Approximated single-period heuristic
        m_est = round(-4.84 + (4.037 * tata) + 1.8, 2)
        risk = "High (>-1.78)" if m_est > -1.78 else "Low (<-1.78)"
        return m_est, risk

    # 1. DSRI: Days Sales in Receivables Index
    dsri_curr = curr.accounts_receivable / curr.revenue if curr.revenue > 0 else 1.0
    dsri_prior = prior.accounts_receivable / prior.revenue if prior.revenue > 0 else 1.0
    dsri = dsri_curr / dsri_prior if dsri_prior > 0 else 1.0

    # 2. GMI: Gross Margin Index (prior / current)
    gm_curr = curr.gross_profit / curr.revenue if curr.revenue > 0 else 0.001
    gm_prior = prior.gross_profit / prior.revenue if prior.revenue > 0 else 0.001
    gmi = gm_prior / gm_curr if gm_curr > 0 else 1.0

    # 3. AQI: Asset Quality Index
    non_curr_curr = 1.0 - ((curr.total_current_assets + curr.property_plant_equipment) / curr.total_assets) if curr.total_assets > 0 else 0.0
    non_curr_prior = 1.0 - ((prior.total_current_assets + prior.property_plant_equipment) / prior.total_assets) if prior.total_assets > 0 else 0.0
    aqi = (non_curr_curr / non_curr_prior) if non_curr_prior > 0.001 else 1.0

    # 4. SGI: Sales Growth Index
    sgi = curr.revenue / prior.revenue if prior.revenue > 0 else 1.0

    # 5. DEPI: Depreciation Index (prior rate / curr rate)
    depr_curr = max(curr.operating_income - (curr.gross_profit - curr.sga_expenses), 0.0)
    depr_rate_curr = (depr_curr / (depr_curr + curr.property_plant_equipment)) if (depr_curr + curr.property_plant_equipment) > 0 else 0.1
    depr_prior = max(prior.operating_income - (prior.gross_profit - prior.sga_expenses), 0.0)
    depr_rate_prior = (depr_prior / (depr_prior + prior.property_plant_equipment)) if (depr_prior + prior.property_plant_equipment) > 0 else 0.1
    depi = depr_rate_prior / depr_rate_curr if depr_rate_curr > 0 else 1.0

    # 6. SGAI: Sales, General and Administrative expenses Index
    sgai_curr = curr.sga_expenses / curr.revenue if curr.revenue > 0 else 0.0
    sgai_prior = prior.sga_expenses / prior.revenue if prior.revenue > 0 else 0.0
    sgai = sgai_curr / sgai_prior if sgai_prior > 0 else 1.0

    # 7. LVGI: Leverage Index (curr total debt/assets / prior total debt/assets)
    lvg_curr = (curr.short_term_debt + curr.long_term_debt) / curr.total_assets if curr.total_assets > 0 else 0.0
    lvg_prior = (prior.short_term_debt + prior.long_term_debt) / prior.total_assets if prior.total_assets > 0 else 0.0
    lvgi = lvg_curr / lvg_prior if lvg_prior > 0 else 1.0

    # 8. TATA: Total Accruals to Total Assets
    tata = (curr.net_income - curr.operating_cash_flow) / curr.total_assets if curr.total_assets > 0 else 0.0

    m = (
        -4.84
        + (0.920 * dsri)
        + (0.528 * gmi)
        + (0.404 * aqi)
        + (0.892 * sgi)
        + (0.115 * depi)
        - (0.172 * sgai)
        + (4.037 * tata)
        + (0.0327 * lvgi)
    )
    m = round(m, 2)
    risk = "High (>-1.78)" if m > -1.78 else "Low (<-1.78)"
    return m, risk
