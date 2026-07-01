"""
HEDGE: Heteroscedastic Entropy-Driven Group Estimation of Credit.

Implements all variants from the paper:
- Base HEDGE (Eq. 5): James-Stein shrinkage with empirical Bayes
- Bias-Tolerant HEDGE (Eq. 6): handles floating-point non-determinism
- Correlated HEDGE: matrix shrinkage for multi-round systems
- Stochastic HEDGE: separates policy variance from environmental noise
- Adaptive HEDGE: EMA-based running statistics
- HEDGE-inv-H: entropy proxy (zero extra cost)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from credit.entropy import compute_entropy_from_logprobs, entropy_to_variance_proxy

LOGGER = logging.getLogger(__name__)


class HEDGEEstimator:
    """Base HEDGE estimator (Eq. 5).

    ĉ_t^HEDGE = c̄ + λ_t * (ĉ_t^L1O - c̄)
    λ_t = w_t / (w_t + w_0)
    w_t = m / σ̂²_t  (precision)
    w_0 estimated via empirical Bayes
    """

    def __init__(
        self,
        positive_part: bool = True,
        min_variance: float = 1e-8,
    ):
        self.positive_part = positive_part
        self.min_variance = min_variance

    def compute_credits(
        self,
        l1o_credits: np.ndarray,
        variances: np.ndarray,
        m: int = 16,
        entropies: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply HEDGE shrinkage to L1O credits.

        Args:
            l1o_credits: (n_steps,) raw L1O credit estimates
            variances: (n_steps,) empirical variances from counterfactuals
            m: number of counterfactual samples
            entropies: (n_steps,) optional entropy values (unused in base)

        Returns:
            hedge_credits: (n_steps,) shrunk credit estimates
        """
        n = len(l1o_credits)
        if n < 3:
            return l1o_credits.copy()

        variances = np.maximum(variances, self.min_variance)
        precisions = m / variances

        c_bar = np.average(l1o_credits, weights=precisions)

        w_0 = self._estimate_w0(l1o_credits, variances, m)

        lambdas = precisions / (precisions + w_0)

        if self.positive_part:
            lambdas = np.clip(lambdas, 0.0, 1.0)

        hedge_credits = c_bar + lambdas * (l1o_credits - c_bar)
        return hedge_credits

    def _estimate_w0(
        self,
        credits: np.ndarray,
        variances: np.ndarray,
        m: int,
    ) -> float:
        """Empirical Bayes estimation of pooling strength w_0.

        Uses marginal maximum likelihood under Gaussian model:
        w_0 = (n - 2) / sum((c_t - c̄)² - σ²_t/m)
        """
        n = len(credits)
        precisions = m / variances
        c_bar = np.average(credits, weights=precisions)
        deviations_sq = (credits - c_bar) ** 2
        excess = deviations_sq - variances / m
        total_excess = np.sum(np.maximum(excess, 0))

        if total_excess < 1e-10:
            return 1e6

        w_0 = (n - 2) / total_excess
        return max(w_0, 1e-10)

    def compute_improvement(
        self,
        l1o_credits: np.ndarray,
        variances: np.ndarray,
        m: int = 16,
    ) -> float:
        """Theoretical MSE improvement (Theorem 3)."""
        n = len(l1o_credits)
        if n < 3:
            return 0.0
        leading = ((n - 2) ** 2) / (n * (n - 2) + 2)
        total_var = np.sum(variances / m)
        return leading * total_var


class BiasTolerantHEDGE(HEDGEEstimator):
    """Bias-Tolerant HEDGE (Eq. 6).

    ĉ_t^BT = c̄ + λ_t * (ĉ_t^L1O - c̄ - b̂_t)
    where b̂_t is estimated via k deterministic replays.
    """

    def __init__(self, k_replays: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.k_replays = k_replays

    def compute_credits(
        self,
        l1o_credits: np.ndarray,
        variances: np.ndarray,
        m: int = 16,
        bias_estimates: Optional[np.ndarray] = None,
        entropies: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply Bias-Tolerant HEDGE.

        Args:
            bias_estimates: (n_steps,) estimated bias from deterministic replays.
                If None, defaults to zero (falls back to base HEDGE).
        """
        n = len(l1o_credits)
        if n < 3:
            return l1o_credits.copy()

        if bias_estimates is None:
            bias_estimates = np.zeros(n)

        variances = np.maximum(variances, self.min_variance)
        precisions = m / variances
        c_bar = np.average(l1o_credits, weights=precisions)
        w_0 = self._estimate_w0(l1o_credits, variances, m)
        lambdas = precisions / (precisions + w_0)

        if self.positive_part:
            lambdas = np.clip(lambdas, 0.0, 1.0)

        hedge_credits = c_bar + lambdas * (l1o_credits - c_bar - bias_estimates)
        return hedge_credits


class CorrelatedHEDGE(HEDGEEstimator):
    """Correlated HEDGE for multi-round systems.

    Replaces scalar shrinkage with matrix shrinkage incorporating
    the estimated cross-step correlation structure.

    ĉ^C-HEDGE = c̄·1 + Λ(ĉ^L1O - c̄·1)
    """

    def compute_credits(
        self,
        l1o_credits: np.ndarray,
        variances: np.ndarray,
        m: int = 16,
        correlation_matrix: Optional[np.ndarray] = None,
        entropies: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply Correlated HEDGE with matrix shrinkage."""
        n = len(l1o_credits)
        if n < 3:
            return l1o_credits.copy()

        variances = np.maximum(variances, self.min_variance)

        if correlation_matrix is None:
            return super().compute_credits(l1o_credits, variances, m, entropies)

        sigma_diag = np.sqrt(variances / m)
        Sigma = np.outer(sigma_diag, sigma_diag) * correlation_matrix

        precisions = m / variances
        c_bar = np.average(l1o_credits, weights=precisions)

        try:
            Sigma_inv = np.linalg.inv(Sigma + np.eye(n) * self.min_variance)
        except np.linalg.LinAlgError:
            return super().compute_credits(l1o_credits, variances, m, entropies)

        trace_Sigma = np.trace(Sigma)
        deviation = l1o_credits - c_bar
        quad_form = deviation @ Sigma_inv @ deviation

        shrinkage_factor = max(0, 1 - (n - 2) * trace_Sigma / (quad_form + 1e-10))
        shrinkage_factor = min(shrinkage_factor, 1.0)

        hedge_credits = c_bar + shrinkage_factor * deviation
        return hedge_credits

    @staticmethod
    def estimate_correlation(
        all_counterfactual_outcomes: List[List[float]],
    ) -> np.ndarray:
        """Estimate cross-step correlation from counterfactual outcomes."""
        n_steps = len(all_counterfactual_outcomes)
        m = len(all_counterfactual_outcomes[0])
        outcomes_matrix = np.array(all_counterfactual_outcomes)
        corr = np.corrcoef(outcomes_matrix)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        return corr


class StochasticHEDGE(HEDGEEstimator):
    """Stochastic HEDGE for environments with aleatoric uncertainty.

    Separates policy variance from environmental noise:
    σ̂²_t = σ²_{t,policy} + σ²_{t,env}

    Uses k deterministic replays to estimate σ²_{t,env}.
    """

    def __init__(self, k_replays: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.k_replays = k_replays

    def compute_credits(
        self,
        l1o_credits: np.ndarray,
        variances: np.ndarray,
        m: int = 16,
        env_variances: Optional[np.ndarray] = None,
        entropies: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply Stochastic HEDGE with variance separation.

        Args:
            env_variances: (n_steps,) environmental variance from deterministic replays.
        """
        if env_variances is None:
            return super().compute_credits(l1o_credits, variances, m, entropies)

        policy_variances = np.maximum(variances - env_variances, self.min_variance)

        return super().compute_credits(l1o_credits, policy_variances, m, entropies)


class AdaptiveHEDGE(HEDGEEstimator):
    """Adaptive HEDGE with exponential moving average statistics.

    Maintains running estimates of variance and credit mean,
    reducing latency from 42ms to 8ms with only 0.1 points accuracy loss.
    """

    def __init__(self, gamma: float = 0.10, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self._ema_variances: Optional[np.ndarray] = None
        self._ema_mean: Optional[float] = None
        self._initialized = False

    def compute_credits(
        self,
        l1o_credits: np.ndarray,
        variances: np.ndarray,
        m: int = 16,
        entropies: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply Adaptive HEDGE using EMA statistics."""
        n = len(l1o_credits)

        if not self._initialized or self._ema_variances is None:
            self._ema_variances = variances.copy()
            self._ema_mean = float(np.mean(l1o_credits))
            self._initialized = True
            return super().compute_credits(l1o_credits, variances, m, entropies)

        if len(self._ema_variances) != n:
            self._ema_variances = variances.copy()
            self._ema_mean = float(np.mean(l1o_credits))

        self._ema_variances = (1 - self.gamma) * self._ema_variances + self.gamma * variances
        self._ema_mean = (1 - self.gamma) * self._ema_mean + self.gamma * np.mean(l1o_credits)

        return super().compute_credits(l1o_credits, self._ema_variances, m, entropies)

    def reset(self):
        self._initialized = False
        self._ema_variances = None
        self._ema_mean = None


class HEDGEInvEntropy(HEDGEEstimator):
    """HEDGE-inv-H: uses entropy proxy instead of empirical variance.

    w_t = 1/H_t (zero extra cost, no counterfactuals needed for variance).
    Matches full HEDGE at m < 4 where empirical variance is unreliable.
    """

    def compute_credits(
        self,
        l1o_credits: np.ndarray,
        variances: np.ndarray,
        m: int = 16,
        entropies: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply HEDGE with entropy-proxy precision."""
        if entropies is None:
            return super().compute_credits(l1o_credits, variances, m)

        n = len(l1o_credits)
        if n < 3:
            return l1o_credits.copy()

        entropy_variances = np.array([
            entropy_to_variance_proxy(h) for h in entropies
        ])
        entropy_variances = np.maximum(entropy_variances, self.min_variance)

        return super().compute_credits(l1o_credits, entropy_variances, m)
