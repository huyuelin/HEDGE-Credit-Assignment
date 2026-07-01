"""
End-task evaluator for credit-optimized multi-agent systems.

Runs the full pipeline: generate transcripts → compute credits → report accuracy.
"""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from agents.vllm_backend import VLLMBackend
from credit.l1o import L1OEstimator
from credit.hedge import (
    HEDGEEstimator,
    BiasTolerantHEDGE,
    CorrelatedHEDGE,
    StochasticHEDGE,
    AdaptiveHEDGE,
    HEDGEInvEntropy,
)
from credit.baselines import UniformCredit, VRRLCredit, CCPOCredit

LOGGER = logging.getLogger(__name__)


class CreditEvaluator:
    """Evaluates multiple credit methods on a set of problems."""

    def __init__(
        self,
        m: int = 16,
        methods: Optional[List[str]] = None,
    ):
        self.m = m
        self.methods = methods or [
            "uniform", "l1o", "hedge", "hedge_bt", "hedge_corr",
            "hedge_stoch", "hedge_adaptive", "hedge_inv_h",
            "vrrl", "ccpo",
        ]
        self.l1o = L1OEstimator(m=m)
        self.hedge = HEDGEEstimator()
        self.hedge_bt = BiasTolerantHEDGE()
        self.hedge_corr = CorrelatedHEDGE()
        self.hedge_stoch = StochasticHEDGE()
        self.hedge_adaptive = AdaptiveHEDGE()
        self.hedge_inv_h = HEDGEInvEntropy()
        self.vrrl = VRRLCredit()
        self.ccpo = CCPOCredit()

    def evaluate_episode(
        self,
        original_outcome: float,
        counterfactual_outcomes: List[List[float]],
        entropies: List[float],
    ) -> Dict[str, np.ndarray]:
        """Compute all credit methods for one episode.

        Returns dict mapping method_name → credit array.
        """
        n = len(counterfactual_outcomes)
        l1o_credits, variances = self.l1o.compute_all_credits(
            original_outcome, counterfactual_outcomes
        )
        entropies_arr = np.array(entropies)

        results = {}
        results["uniform"] = np.ones(n) / n
        results["l1o"] = l1o_credits
        results["hedge"] = self.hedge.compute_credits(l1o_credits, variances, self.m, entropies_arr)
        results["hedge_bt"] = self.hedge_bt.compute_credits(l1o_credits, variances, self.m)
        results["hedge_corr"] = self.hedge_corr.compute_credits(l1o_credits, variances, self.m)
        results["hedge_stoch"] = self.hedge_stoch.compute_credits(l1o_credits, variances, self.m)
        results["hedge_adaptive"] = self.hedge_adaptive.compute_credits(l1o_credits, variances, self.m, entropies_arr)
        results["hedge_inv_h"] = self.hedge_inv_h.compute_credits(l1o_credits, variances, self.m, entropies_arr)
        results["vrrl"] = self.vrrl.compute_credits(l1o_credits, variances, self.m)
        results["ccpo"] = self.ccpo.compute_credits(l1o_credits, variances, self.m)
        results["variances"] = variances

        return results

    def run_evaluation(
        self,
        all_episodes: List[Dict],
    ) -> Dict[str, float]:
        """Aggregate evaluation across multiple episodes.

        Args:
            all_episodes: List of dicts from env.run_with_credits()

        Returns:
            Dict mapping method → mean credit quality metric
        """
        method_scores = {m: [] for m in self.methods}

        for ep in all_episodes:
            credits = self.evaluate_episode(
                ep["original_outcome"],
                ep["counterfactual_outcomes"],
                ep["entropies"],
            )
            for method in self.methods:
                if method in credits:
                    score = self._credit_quality(credits[method], ep["original_outcome"])
                    method_scores[method].append(score)

        return {m: np.mean(scores) if scores else 0.0 for m, scores in method_scores.items()}

    def _credit_quality(self, credits: np.ndarray, outcome: float) -> float:
        """Simple credit quality metric: correlation with outcome signal."""
        if outcome > 0.5:
            return float(credits.max())
        else:
            return float(1.0 - credits.max())
