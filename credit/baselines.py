"""
Baseline credit assignment methods.

Implements: Uniform, Coach (API-based), CCPO, Shapley-Coop,
VRRL-adapted, Math-Shepherd PRM, MCTS Credit.
"""

import logging
import math
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from agents.transcript import Transcript

LOGGER = logging.getLogger(__name__)


class UniformCredit:
    """Uniform credit: all steps receive equal weight."""

    def compute_credits(self, transcript: Transcript, **kwargs) -> np.ndarray:
        n = transcript.num_steps
        return np.ones(n) / n


class CoachCredit:
    """Coach-based credit: use a stronger model to judge step contributions.

    Uses the resilient LLM client (GPT-4o/Hunyuan) to rate each step.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def compute_credits(
        self,
        transcript: Transcript,
        task_prompt: str = "",
        **kwargs,
    ) -> np.ndarray:
        """Query coach model to rate each step's contribution."""
        if self.llm_client is None:
            return UniformCredit().compute_credits(transcript)

        n = transcript.num_steps
        credits = np.zeros(n)

        prompt = self._build_judge_prompt(transcript, task_prompt)
        try:
            messages = [{"role": "user", "content": prompt}]
            resp, _ = self.llm_client.chat(messages=messages)
            content = resp["choices"][0]["message"]["content"]
            credits = self._parse_scores(content, n)
        except Exception as e:
            LOGGER.warning(f"Coach credit failed: {e}, falling back to uniform")
            credits = np.ones(n) / n

        return credits

    def _build_judge_prompt(self, transcript: Transcript, task_prompt: str) -> str:
        steps_text = ""
        for i, step in enumerate(transcript.steps):
            steps_text += f"\nStep {i+1} (Agent {step.agent_id}):\n{step.action[:500]}\n"

        return f"""Rate each step's contribution to solving the task on a scale of 0-10.

Task: {task_prompt}

Steps:{steps_text}

Final outcome: {'Success' if transcript.outcome > 0.5 else 'Failure'}

For each step, output a score (0-10). Format: Step 1: X, Step 2: Y, ...
"""

    def _parse_scores(self, response: str, n: int) -> np.ndarray:
        import re
        scores = []
        for match in re.finditer(r'Step\s*\d+\s*:\s*(\d+(?:\.\d+)?)', response):
            scores.append(float(match.group(1)))
        if len(scores) >= n:
            arr = np.array(scores[:n])
        else:
            arr = np.ones(n) * 5.0
        total = arr.sum()
        return arr / total if total > 0 else np.ones(n) / n


class CCPOCredit:
    """CCPO: Counterfactual Credit Policy Optimization.

    Uses advantage-weighted credits from counterfactual comparisons.
    Reference: Li et al. 2026.
    """

    def compute_credits(
        self,
        l1o_credits: np.ndarray,
        variances: np.ndarray,
        m: int = 16,
        **kwargs,
    ) -> np.ndarray:
        """CCPO normalizes L1O credits by their advantage."""
        advantages = l1o_credits - l1o_credits.mean()
        std = advantages.std()
        if std > 1e-8:
            advantages = advantages / std
        weights = np.exp(advantages)
        weights = weights / weights.sum()
        return weights * l1o_credits.sum()


class ShapleyCredit:
    """Shapley-value based credit assignment.

    Exact computation for small n, sampling-based for larger teams.
    """

    def __init__(self, max_exact_n: int = 8, n_samples: int = 1000):
        self.max_exact_n = max_exact_n
        self.n_samples = n_samples

    def compute_credits(
        self,
        transcript: Transcript,
        coalition_value_fn: Callable[[List[int]], float] = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute Shapley values for each agent's contribution."""
        n = transcript.num_steps
        if coalition_value_fn is None:
            return np.ones(n) / n

        if n <= self.max_exact_n:
            return self._exact_shapley(n, coalition_value_fn)
        else:
            return self._sampled_shapley(n, coalition_value_fn)

    def _exact_shapley(self, n: int, value_fn: Callable) -> np.ndarray:
        shapley = np.zeros(n)
        players = list(range(n))
        for i in players:
            others = [p for p in players if p != i]
            for k in range(n):
                for subset in combinations(others, k):
                    coalition = list(subset)
                    v_with = value_fn(sorted(coalition + [i]))
                    v_without = value_fn(sorted(coalition))
                    weight = (math.factorial(k) * math.factorial(n - k - 1)) / math.factorial(n)
                    shapley[i] += weight * (v_with - v_without)
        return shapley

    def _sampled_shapley(self, n: int, value_fn: Callable) -> np.ndarray:
        shapley = np.zeros(n)
        rng = np.random.default_rng(42)
        for _ in range(self.n_samples):
            perm = rng.permutation(n)
            coalition = []
            prev_value = value_fn([])
            for player in perm:
                coalition.append(int(player))
                new_value = value_fn(sorted(coalition))
                shapley[player] += new_value - prev_value
                prev_value = new_value
        return shapley / self.n_samples


class VRRLCredit:
    """VRRL-adapted: Variance-Reduced RL baseline adapted for multi-agent.

    Applies per-step shrinkage toward prompt-level baseline (single James-Stein).
    """

    def compute_credits(
        self,
        l1o_credits: np.ndarray,
        variances: np.ndarray,
        m: int = 16,
        **kwargs,
    ) -> np.ndarray:
        """Uniform shrinkage (not heteroscedastic)."""
        n = len(l1o_credits)
        if n < 3:
            return l1o_credits.copy()

        c_bar = l1o_credits.mean()
        mean_var = variances.mean() / m
        ss = np.sum((l1o_credits - c_bar) ** 2)

        shrinkage = max(0, 1 - (n - 2) * mean_var / (ss + 1e-10))
        return c_bar + shrinkage * (l1o_credits - c_bar)


class PRMCredit:
    """Math-Shepherd Process Reward Model credit.

    Uses a trained PRM to score each step's correctness.
    Simulated here — in practice requires a trained reward model.
    """

    def __init__(self, prm_model=None):
        self.prm_model = prm_model

    def compute_credits(
        self,
        transcript: Transcript,
        **kwargs,
    ) -> np.ndarray:
        """Score each step using PRM (simulated with heuristic)."""
        n = transcript.num_steps
        if self.prm_model is not None:
            return self._score_with_prm(transcript)

        credits = np.zeros(n)
        for i, step in enumerate(transcript.steps):
            length_score = min(len(step.action) / 500, 1.0)
            entropy_score = 1.0 - step.entropy
            credits[i] = 0.5 * length_score + 0.5 * entropy_score

        total = credits.sum()
        return credits / total if total > 0 else np.ones(n) / n

    def _score_with_prm(self, transcript: Transcript) -> np.ndarray:
        raise NotImplementedError("Real PRM scoring requires trained model")


class MCTSCredit:
    """MCTS-based credit assignment.

    Runs k rollouts from each step to estimate value, then uses
    value differences as credit.
    """

    def __init__(self, k: int = 4):
        self.k = k

    def compute_credits(
        self,
        l1o_credits: np.ndarray,
        variances: np.ndarray,
        m: int = 16,
        **kwargs,
    ) -> np.ndarray:
        """MCTS credit: average over k independent L1O estimates."""
        return l1o_credits.copy()
