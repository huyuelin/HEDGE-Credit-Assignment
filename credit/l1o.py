"""
Leave-One-Out (L1O) credit estimator.

Implements Eq. 1 from the paper:
  ĉ_t^L1O = R(τ) - (1/m) Σ R(τ_{-t}^{(j)})

where τ_{-t}^{(j)} resamples the action at step t from the agent's policy.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from agents.transcript import Transcript

LOGGER = logging.getLogger(__name__)


class L1OEstimator:
    """Leave-One-Out counterfactual credit estimator."""

    def __init__(self, m: int = 16):
        self.m = m

    def compute_credit(
        self,
        original_outcome: float,
        counterfactual_outcomes: List[float],
    ) -> Tuple[float, float]:
        """Compute L1O credit for a single step.

        Args:
            original_outcome: R(τ) — outcome of the original transcript
            counterfactual_outcomes: [R(τ_{-t}^{(1)}), ..., R(τ_{-t}^{(m)})]

        Returns:
            (credit, variance): estimated credit and empirical variance
        """
        cf = np.array(counterfactual_outcomes)
        mean_cf = cf.mean()
        credit = original_outcome - mean_cf
        variance = cf.var(ddof=1) if len(cf) > 1 else 0.0
        return float(credit), float(variance)

    def compute_all_credits(
        self,
        original_outcome: float,
        all_counterfactual_outcomes: List[List[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute L1O credits for all steps in a transcript.

        Args:
            original_outcome: R(τ)
            all_counterfactual_outcomes: List of m outcomes per step
                Shape: [n_steps][m]

        Returns:
            (credits, variances): arrays of shape (n_steps,)
        """
        n_steps = len(all_counterfactual_outcomes)
        credits = np.zeros(n_steps)
        variances = np.zeros(n_steps)

        for t in range(n_steps):
            c, v = self.compute_credit(
                original_outcome,
                all_counterfactual_outcomes[t],
            )
            credits[t] = c
            variances[t] = v

        return credits, variances

    def compute_snr(
        self,
        credits: np.ndarray,
        variances: np.ndarray,
        m: int = None,
    ) -> float:
        """Compute signal-to-noise ratio (Theorem 2).

        SNR = |mean(c)| / (mean(σ) * sqrt(n/m))
        """
        m = m or self.m
        n = len(credits)
        signal = np.abs(credits.mean())
        noise = np.sqrt(variances.mean() * n / m)
        return signal / noise if noise > 1e-10 else float('inf')
