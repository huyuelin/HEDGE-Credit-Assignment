"""
Self-Refinement Loop environment.

A single agent refines its own output repeatedly over n iterations.
Each iteration is treated as a "step" for credit assignment.
"""

import logging
from typing import Callable, Dict, List, Tuple

from agents.base_agent import LLMAgent
from agents.transcript import Transcript
from agents.vllm_backend import VLLMBackend
from agents.workflow import SequentialWorkflow

LOGGER = logging.getLogger(__name__)


class RefinementEnvironment:
    """Self-refinement loop with n iterations."""

    def __init__(
        self,
        backend: VLLMBackend,
        n_iterations: int = 9,
        temperature: float = 0.7,
    ):
        self.backend = backend
        self.n_iterations = n_iterations
        self.temperature = temperature

    def create_agents(self) -> List[LLMAgent]:
        agents = []
        for i in range(self.n_iterations):
            if i == 0:
                role = "You are a problem solver. Provide an initial solution to the given problem."
            elif i == self.n_iterations - 1:
                role = "You are the final reviewer. Produce the definitive answer based on all refinements above. Put answer in \\boxed{}."
            else:
                role = f"You are a refinement agent (iteration {i+1}). Review the solution above, identify weaknesses, and improve it."
            agent = LLMAgent(
                agent_id=0,
                role_prompt=role,
                backend=self.backend,
                temperature=self.temperature * (0.9 ** i),
                max_tokens=512,
            )
            agents.append(agent)
        return agents

    def make_outcome_fn(self, gold_answer: str) -> Callable[[Transcript], float]:
        from environments.mathchat import extract_math_answer
        def outcome_fn(transcript: Transcript) -> float:
            full_text = transcript.to_text()
            predicted = extract_math_answer(full_text)
            if predicted is None:
                return 0.0
            try:
                pred_f = float(predicted.replace(",", "").strip())
                gold_f = float(gold_answer.replace(",", "").strip())
                return 1.0 if abs(pred_f - gold_f) < 1e-6 else 0.0
            except ValueError:
                return 1.0 if predicted.strip() == gold_answer.strip() else 0.0
        return outcome_fn

    def run_with_credits(self, problem: str, gold_answer: str, m: int = 16) -> Dict:
        agents = self.create_agents()
        outcome_fn = self.make_outcome_fn(gold_answer)
        workflow = SequentialWorkflow(agents=agents, outcome_fn=outcome_fn, task_prompt=problem)
        transcript = workflow.run()

        all_cf_outcomes = []
        for step_idx in range(transcript.num_steps):
            cfs = workflow.generate_counterfactuals(transcript, step_idx, m=m)
            all_cf_outcomes.append([cf.outcome for cf in cfs])

        return {
            "transcript": transcript,
            "original_outcome": transcript.outcome,
            "counterfactual_outcomes": all_cf_outcomes,
            "entropies": transcript.get_entropies(),
        }
