"""
Adaptive Portfolio Optimizer: Hierarchical Risk Parity (HRP), Black-Litterman with Agent Views,
and Regime-Adaptive Risk Envelopes.
Part of the FLAME-AI Decision Intelligence Subsystem.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PortfolioOptimizer")


class AdaptivePortfolioOptimizer:
    """
    Quantitative Portfolio Allocation Engine:
    - Hierarchical Risk Parity (HRP)
    - Equal Risk Contribution (ERC / Risk Parity)
    - Black-Litterman with Agentic Views
    - Regime-Conditioned Adaptive Switching
    """

    def __init__(self, risk_free_rate: float = 0.045, risk_aversion_lambda: float = 2.5):
        self.risk_free_rate = risk_free_rate
        self.lambda_param = risk_aversion_lambda

    def optimize(
        self,
        cov_matrix: pd.DataFrame,
        regime_id: int = 0,
        agent_views: Optional[Dict[str, float]] = None,
        method: str = "regime_adaptive",
        max_weight: float = 0.35,
        min_weight: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculates optimal asset allocations conditioned on method, covariance, and active regime.
        """
        tickers = list(cov_matrix.columns)
        n = len(tickers)
        if n == 0:
            return {}
        if n == 1:
            return {tickers[0]: 1.0}

        # Check and sanitize cov_matrix
        cov_vals = cov_matrix.values
        if np.any(np.isnan(cov_vals)) or np.any(np.isinf(cov_vals)) or cov_vals.sum() == 0:
            return {t: 1.0 / n for t in tickers}

        # Ridge stabilization to prevent singular / ill-conditioned matrices
        trace_mean = float(np.trace(cov_vals) / n)
        ridge = max(trace_mean * 1e-4, 1e-5)
        cov_clean = cov_matrix.copy() + np.eye(n) * ridge

        # 1. Dispatch optimization by method
        try:
            if method == "hrp":
                raw_weights = self._hierarchical_risk_parity(cov_clean)
            elif method == "risk_parity":
                raw_weights = self._inverse_variance_weights(cov_clean)
            elif method == "black_litterman":
                raw_weights = self._black_litterman(cov_clean, agent_views or {})
            elif method == "regime_adaptive":
                raw_weights = self._regime_adaptive_allocation(cov_clean, regime_id, agent_views or {})
            else:
                raw_weights = {t: 1.0 / n for t in tickers}
        except Exception as e:
            logger.warning(f"Optimization {method} encountered issue ({e}). Reverting to risk parity.")
            raw_weights = self._inverse_variance_weights(cov_clean)

        # 2. Apply institutional concentration constraints [min_weight, max_weight]
        constrained_weights = self._apply_weight_constraints(raw_weights, min_weight, max_weight)
        return constrained_weights

    def _hierarchical_risk_parity(self, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        López de Prado Hierarchical Risk Parity (HRP) algorithm:
        1. Tree correlation clustering distance
        2. Quasi-diagonalization
        3. Recursive bisection
        """
        cov = cov_matrix.values
        tickers = list(cov_matrix.columns)
        n = len(tickers)

        # Correlation matrix
        vols = np.sqrt(np.diag(cov))
        vols[vols == 0] = 1e-6
        corr = cov / np.outer(vols, vols)
        corr = np.clip(corr, -1.0, 1.0)

        # Distance matrix d_ij = sqrt(0.5 * (1 - rho_ij))
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        # Hierarchical Linkage Clustering
        condensed_dist = squareform(dist, checks=False)
        link = linkage(condensed_dist, method="single")

        # Quasi-diagonalization (get sorted cluster indices)
        sorted_indices = self._get_quasi_diag_indices(link, n)
        sorted_tickers = [tickers[i] for i in sorted_indices]

        # Recursive Bisection
        weights = pd.Series(1.0, index=sorted_tickers)
        cluster_items = [sorted_tickers]

        while len(cluster_items) > 0:
            cluster_items = [
                i[j:k] for i in cluster_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            for i in range(0, len(cluster_items), 2):
                c1 = cluster_items[i]
                c2 = cluster_items[i + 1]

                cov1 = cov_matrix.loc[c1, c1]
                cov2 = cov_matrix.loc[c2, c2]

                v1 = self._cluster_variance(cov1)
                v2 = self._cluster_variance(cov2)

                alpha = 1.0 - v1 / (v1 + v2 + 1e-9)
                weights[c1] *= alpha
                weights[c2] *= (1.0 - alpha)

        return weights.to_dict()

    def _get_quasi_diag_indices(self, link: np.ndarray, n: int) -> List[int]:
        """Extracts sorted leaves order from hierarchical linkage matrix."""
        order = [int(link[-1, 0]), int(link[-1, 1])]
        num_clusters = 2 * n - 1
        while max(order) >= n:
            new_order = []
            for item in order:
                if item >= n:
                    sub_idx = item - n
                    new_order.extend([int(link[sub_idx, 0]), int(link[sub_idx, 1])])
                else:
                    new_order.append(item)
            order = new_order
        return order

    def _cluster_variance(self, cluster_cov: pd.DataFrame) -> float:
        """Inverse-variance portfolio variance for a cluster with ridge stabilization."""
        diag = np.diag(cluster_cov.values)
        diag = np.where(np.isnan(diag) | (diag <= 1e-7), 1e-4, diag)
        ivp = 1.0 / diag
        s = np.sum(ivp)
        if s == 0 or np.isnan(s):
            ivp = np.ones_like(diag) / len(diag)
        else:
            ivp /= s
        var = float(np.dot(np.dot(ivp, cluster_cov.values), ivp))
        return var if (var > 0 and not np.isnan(var)) else 1e-4

    def _inverse_variance_weights(self, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Risk Parity / Equal Risk Contribution proxy."""
        vols = np.sqrt(np.diag(cov_matrix.values))
        inv_vols = 1.0 / (vols + 1e-6)
        weights = inv_vols / np.sum(inv_vols)
        return dict(zip(cov_matrix.columns, weights))

    def _black_litterman(
        self,
        cov_matrix: pd.DataFrame,
        agent_views: Dict[str, float],
        tau: float = 0.05
    ) -> Dict[str, float]:
        """
        Black-Litterman model integrating agent consensus views with market equilibrium prior.
        """
        tickers = list(cov_matrix.columns)
        n = len(tickers)
        sigma = cov_matrix.values

        # Market equilibrium prior (equal-weighted benchmark proxy)
        w_mkt = np.ones(n) / n
        pi = self.lambda_param * np.dot(sigma, w_mkt)

        # Build view matrix P and vector Q
        views = []
        p_rows = []
        for i, t in enumerate(tickers):
            if t in agent_views:
                views.append(agent_views[t])
                p_row = np.zeros(n)
                p_row[i] = 1.0
                p_rows.append(p_row)

        if not views:
            return self._inverse_variance_weights(cov_matrix)

        P = np.array(p_rows)
        Q = np.array(views)
        k = len(views)

        # View uncertainty covariance Omega = diag(P (tau Sigma) P^T)
        omega = np.diag(np.diag(np.dot(np.dot(P, tau * sigma), P.T)))

        # Posterior mean E[R]
        inv_tau_sigma = np.linalg.pinv(tau * sigma)
        inv_omega = np.linalg.pinv(omega)

        m1 = inv_tau_sigma + np.dot(np.dot(P.T, inv_omega), P)
        inv_m1 = np.linalg.pinv(m1)
        m2 = np.dot(inv_tau_sigma, pi) + np.dot(np.dot(P.T, inv_omega), Q)
        er = np.dot(inv_m1, m2)

        # Unconstrained optimal weights w = 1/lambda * Sigma^-1 * er
        raw_w = np.dot(np.linalg.pinv(sigma), er) / self.lambda_param
        # Long-only projection
        raw_w = np.maximum(raw_w, 0.0)
        total = np.sum(raw_w)
        if total > 0:
            raw_w /= total
        else:
            raw_w = np.ones(n) / n

        return dict(zip(tickers, raw_w))

    def _regime_adaptive_allocation(
        self,
        cov_matrix: pd.DataFrame,
        regime_id: int,
        agent_views: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Dynamic Regime-Adaptive Switching Logic:
        - Bull (0): Black-Litterman with Agent Views
        - Bear (1): Inverse Volatility Risk Parity
        - Crisis (2): Hierarchical Risk Parity (HRP)
        - Sideways (3): Hybrid HRP + Agent Views
        """
        if regime_id == 0:  # Bull
            return self._black_litterman(cov_matrix, agent_views)
        elif regime_id == 1:  # Bear
            return self._inverse_variance_weights(cov_matrix)
        elif regime_id == 2:  # High Volatility Crisis
            return self._hierarchical_risk_parity(cov_matrix)
        else:  # Sideways RangeBound
            w_hrp = self._hierarchical_risk_parity(cov_matrix)
            w_bl = self._black_litterman(cov_matrix, agent_views)
            combined = {t: 0.5 * w_hrp[t] + 0.5 * w_bl[t] for t in cov_matrix.columns}
            tot = sum(combined.values())
            return {t: v / tot for t, v in combined.items()}

    def _apply_weight_constraints(
        self,
        weights: Dict[str, float],
        min_weight: float,
        max_weight: float
    ) -> Dict[str, float]:
        """Clips and iteratively normalizes weights to strictly respect concentration limits."""
        w = pd.Series(weights).fillna(1.0 / len(weights))
        if w.sum() == 0 or np.isnan(w.sum()):
            w = pd.Series(1.0 / len(weights), index=weights.keys())

        for _ in range(10):
            w = w.clip(lower=min_weight, upper=max_weight)
            total = w.sum()
            if abs(total - 1.0) < 1e-4:
                break
            w = w / (total + 1e-9)

        return {k: float(np.round(v, 4)) for k, v in w.items()}


if __name__ == "__main__":
    opt = AdaptivePortfolioOptimizer()
    sample_cov = pd.DataFrame(
        np.array([
            [0.04, 0.015, 0.01],
            [0.015, 0.06, 0.02],
            [0.01, 0.02, 0.03]
        ]),
        index=["AAPL", "NVDA", "MSFT"],
        columns=["AAPL", "NVDA", "MSFT"]
    )
    print("HRP Weights:", opt.optimize(sample_cov, method="hrp"))
    print("Regime Adaptive (Crisis):", opt.optimize(sample_cov, regime_id=2))
