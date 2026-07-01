"""
PIST-MAS: Parallel-Independent-Subtask Multi-Agent System.

The key causal decoupling experiment from Section 3.
n agents solve n completely independent math problems in parallel.
Coordination difficulty is ZERO by construction — agents never interact.

Purpose: Demonstrate that credit collapse is an estimator phenomenon,
not a coordination phenomenon. L1O still collapses at n≈8 despite
zero inter-agent dependency.
"""

import logging
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from agents.base_agent import LLMAgent
from agents.transcript import Transcript, TranscriptStep
from agents.vllm_backend import VLLMBackend
from agents.workflow import ParallelWorkflow

LOGGER = logging.getLogger(__name__)


def extract_number_answer(text: str) -> Optional[str]:
    """Extract numerical answer from generated text."""
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    answer_match = re.search(r'(?:answer|Answer|ANSWER)\s*(?:is|:)\s*([^\n.,]+)', text)
    if answer_match:
        return answer_match.group(1).strip()
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]
    return None


def make_gsm8k_outcome_fn(gold_answer: str) -> Callable[[str], float]:
    """Create an outcome function that checks if the answer matches gold."""
    def outcome_fn(action_text: str) -> float:
        predicted = extract_number_answer(action_text)
        if predicted is None:
            return 0.0
        try:
            pred_num = float(predicted.replace(",", ""))
            gold_num = float(gold_answer.replace(",", ""))
            return 1.0 if abs(pred_num - gold_num) < 1e-6 else 0.0
        except ValueError:
            return 1.0 if predicted.strip() == gold_answer.strip() else 0.0
    return outcome_fn


class PISTMASEnvironment:
    """PIST-MAS environment for causal decoupling experiments."""

    def __init__(
        self,
        backend: VLLMBackend,
        n_agents: int = 8,
        temperature: float = 0.7,
    ):
        self.backend = backend
        self.n_agents = n_agents
        self.temperature = temperature

    def create_agents(self) -> List[LLMAgent]:
        """Create n independent math-solving agents."""
        agents = []
        for i in range(self.n_agents):
            agent = LLMAgent(
                agent_id=i,
                role_prompt=(
                    "You are a mathematical problem solver. "
                    "Solve the given math problem step by step. "
                    "Put your final numerical answer in \\boxed{}."
                ),
                backend=self.backend,
                temperature=self.temperature,
                max_tokens=512,
            )
            agents.append(agent)
        return agents

    def run_episode(
        self,
        problems: List[Dict[str, str]],
    ) -> Tuple[Transcript, ParallelWorkflow]:
        """Run one PIST-MAS episode with n independent problems.

        Args:
            problems: List of n dicts with 'problem' and 'answer' keys.

        Returns:
            (transcript, workflow) for credit computation.
        """
        assert len(problems) == self.n_agents

        agents = self.create_agents()
        prompts = [p["problem"] for p in problems]
        outcome_fns = [make_gsm8k_outcome_fn(p["answer"]) for p in problems]

        workflow = ParallelWorkflow(
            agents=agents,
            subtask_prompts=prompts,
            subtask_outcome_fns=outcome_fns,
            aggregate_fn=lambda scores: sum(scores) / len(scores),
        )

        transcript = workflow.run()
        return transcript, workflow

    def run_with_credits(
        self,
        problems: List[Dict[str, str]],
        m: int = 16,
    ) -> Dict:
        """Run episode and compute L1O counterfactuals for credit estimation.

        Returns dict with transcript, counterfactual_outcomes, and metadata.
        """
        transcript, workflow = self.run_episode(problems)

        all_cf_outcomes = []
        for step_idx in range(self.n_agents):
            cfs = workflow.generate_counterfactuals(transcript, step_idx, m=m)
            cf_outcomes = [cf.outcome for cf in cfs]
            all_cf_outcomes.append(cf_outcomes)

        return {
            "transcript": transcript,
            "original_outcome": transcript.outcome,
            "counterfactual_outcomes": all_cf_outcomes,
            "entropies": transcript.get_entropies(),
            "sub_outcomes": transcript.metadata.get("sub_outcomes", []),
        }
