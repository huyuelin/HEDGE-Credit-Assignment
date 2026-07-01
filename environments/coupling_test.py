"""
Coupling stress test environment (Chain-Dependency MAS).

Implements Table 6 from the paper. A coupling parameter ρ∈[0,1] controls
how strongly each agent's action space is constrained by the previous
agent's output.

ρ=0: fully independent (Assumption 3 holds exactly)
ρ=1: fully coupled (maximum violation)
"""

import logging
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from agents.base_agent import LLMAgent
from agents.transcript import Transcript, TranscriptStep
from agents.vllm_backend import VLLMBackend
from agents.workflow import SequentialWorkflow

LOGGER = logging.getLogger(__name__)


class CouplingTestEnvironment:
    """Chain-dependency MAS with controllable coupling ρ."""

    def __init__(
        self,
        backend: VLLMBackend,
        n_agents: int = 8,
        coupling: float = 0.0,
        temperature: float = 0.7,
    ):
        self.backend = backend
        self.n_agents = n_agents
        self.coupling = coupling
        self.temperature = temperature

    def create_agents(self) -> List[LLMAgent]:
        """Create agents with coupling-dependent role prompts."""
        agents = []
        for i in range(self.n_agents):
            if self.coupling < 0.3:
                role = (
                    f"You are Agent {i+1}. Solve your part of the problem independently. "
                    "Show your work step by step and put your answer in \\boxed{{}}."
                )
            elif self.coupling < 0.7:
                role = (
                    f"You are Agent {i+1}. Build upon the previous agent's work. "
                    "You should extend their approach and refine the answer. "
                    "Put your answer in \\boxed{{}}."
                )
            else:
                role = (
                    f"You are Agent {i+1}. You MUST use the exact approach and "
                    "intermediate results from the previous agent. Continue their "
                    "specific method without deviation. Put your answer in \\boxed{{}}."
                )
            agent = LLMAgent(
                agent_id=i,
                role_prompt=role,
                backend=self.backend,
                temperature=self.temperature * (1 - 0.3 * self.coupling),
                max_tokens=512,
            )
            agents.append(agent)
        return agents

    def make_outcome_fn(self, gold_answer: str) -> Callable[[Transcript], float]:
        """Outcome function with coupling-dependent noise."""
        def outcome_fn(transcript: Transcript) -> float:
            full_text = transcript.to_text()
            boxed = re.findall(r'\\boxed\{([^}]+)\}', full_text)
            if not boxed:
                return 0.0
            predicted = boxed[-1].strip()
            try:
                pred_f = float(predicted.replace(",", ""))
                gold_f = float(gold_answer.replace(",", ""))
                return 1.0 if abs(pred_f - gold_f) < 1e-6 else 0.0
            except ValueError:
                return 1.0 if predicted.strip() == gold_answer.strip() else 0.0
        return outcome_fn

    def run_episode(
        self,
        problem: str,
        gold_answer: str,
    ) -> Tuple[Transcript, SequentialWorkflow]:
        agents = self.create_agents()
        outcome_fn = self.make_outcome_fn(gold_answer)

        task_prompt = f"Solve this math problem collaboratively:\n{problem}"
        workflow = SequentialWorkflow(
            agents=agents,
            outcome_fn=outcome_fn,
            task_prompt=task_prompt,
        )

        transcript = workflow.run()
        return transcript, workflow

    def run_with_credits(
        self,
        problem: str,
        gold_answer: str,
        m: int = 16,
    ) -> Dict:
        transcript, workflow = self.run_episode(problem, gold_answer)

        all_cf_outcomes = []
        for step_idx in range(transcript.num_steps):
            cfs = workflow.generate_counterfactuals(transcript, step_idx, m=m)
            cf_outcomes = [cf.outcome for cf in cfs]
            all_cf_outcomes.append(cf_outcomes)

        return {
            "transcript": transcript,
            "original_outcome": transcript.outcome,
            "counterfactual_outcomes": all_cf_outcomes,
            "entropies": transcript.get_entropies(),
            "coupling": self.coupling,
        }

    def estimate_correlation(
        self,
        all_cf_outcomes: List[List[float]],
    ) -> float:
        """Estimate max cross-step correlation from counterfactual data."""
        n = len(all_cf_outcomes)
        if n < 2:
            return 0.0
        outcomes_matrix = np.array(all_cf_outcomes)
        corr = np.corrcoef(outcomes_matrix)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)
        return float(np.max(np.abs(corr)))
