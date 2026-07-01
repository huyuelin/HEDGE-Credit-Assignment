"""
Entropy computation utilities for credit assignment.
"""

import math
from typing import List

import numpy as np


def compute_entropy_from_logprobs(logprobs: List[float]) -> float:
    """Compute normalized entropy from token-level log-probabilities.

    H_t = -mean(logprob) / log(V), normalized to [0, 1].
    This is the average surprise per token.
    """
    if not logprobs:
        return 0.5
    neg_lp = [-lp for lp in logprobs]
    mean_neg_lp = sum(neg_lp) / len(neg_lp)
    max_entropy = math.log(32000)
    return min(mean_neg_lp / max_entropy, 1.0)


def compute_gini_simpson(logprobs: List[float]) -> float:
    """Compute Gini-Simpson diversity index D_eff = 1 - sum(p_i^2).

    Used in Assumption 2 of the paper as the variance-diversity link.
    """
    if not logprobs:
        return 0.5
    probs = [math.exp(lp) for lp in logprobs]
    sum_sq = sum(p * p for p in probs)
    return 1.0 - sum_sq


def entropy_to_variance_proxy(entropy: float, kappa: float = 1.0, gamma: float = 1.0) -> float:
    """Map entropy to expected variance under Assumption 2.

    σ²_t ≈ κ * D_eff^γ ≈ κ * (1 - e^{-H_t})^γ
    """
    diversity = 1.0 - math.exp(-entropy)
    return kappa * (diversity ** gamma)


def batch_entropies(logprobs_list: List[List[float]]) -> np.ndarray:
    """Compute entropies for a batch of steps."""
    return np.array([compute_entropy_from_logprobs(lp) for lp in logprobs_list])
