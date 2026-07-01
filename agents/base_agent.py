"""
Base agent for cooperative LLM multi-agent systems.

Each agent has a role prompt and generates text given the current state
(shared transcript). Returns action text + token-level entropy.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

from agents.vllm_backend import VLLMBackend, GenerationResult
from agents.transcript import TranscriptStep

LOGGER = logging.getLogger(__name__)


class LLMAgent:
    """A single LLM agent in the multi-agent system."""

    def __init__(
        self,
        agent_id: int,
        role_prompt: str,
        backend: VLLMBackend,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
    ):
        self.agent_id = agent_id
        self.role_prompt = role_prompt
        self.backend = backend
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def act(
        self,
        state: str,
        task_prompt: str = "",
        step_idx: int = 0,
    ) -> TranscriptStep:
        """Generate an action given the current state.

        Returns TranscriptStep with action text and entropy.
        """
        messages = self._build_messages(state, task_prompt)
        results = self.backend.generate(
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            logprobs=True,
            n=1,
        )
        result = results[0]
        step = TranscriptStep(
            step_idx=step_idx,
            agent_id=self.agent_id,
            state=state,
            action=result.text,
            token_logprobs=result.token_logprobs,
        )
        step.compute_entropy()
        return step

    def resample(
        self,
        state: str,
        task_prompt: str = "",
        step_idx: int = 0,
        n: int = 1,
    ) -> List[TranscriptStep]:
        """Resample n actions from the policy (for L1O counterfactuals)."""
        messages = self._build_messages(state, task_prompt)
        results = self.backend.generate(
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            logprobs=True,
            n=n,
        )
        steps = []
        for r in results:
            step = TranscriptStep(
                step_idx=step_idx,
                agent_id=self.agent_id,
                state=state,
                action=r.text,
                token_logprobs=r.token_logprobs,
            )
            step.compute_entropy()
            steps.append(step)
        return steps

    def _build_messages(self, state: str, task_prompt: str) -> List[Dict[str, str]]:
        """Build the message list for the LLM."""
        messages = [{"role": "system", "content": self.role_prompt}]
        if task_prompt:
            messages.append({"role": "user", "content": task_prompt})
        if state:
            messages.append({"role": "user", "content": f"Current progress:\n{state}\n\nPlease continue."})
        return messages


def compute_sequence_entropy(logprobs: List[float]) -> float:
    """Compute normalized Shannon entropy from a sequence of token log-probabilities.

    H = -mean(sum(p * log(p))) normalized to [0, 1].
    Uses the approximation: given logprob of the chosen token,
    entropy ≈ -logprob (bits of surprise per token).
    """
    if not logprobs:
        return 0.5
    neg_logprobs = [-lp for lp in logprobs]
    mean_surprise = sum(neg_logprobs) / len(neg_logprobs)
    max_entropy = math.log(32000)
    return min(mean_surprise / max_entropy, 1.0)
