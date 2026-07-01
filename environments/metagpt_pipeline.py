"""
MetaGPT Pipeline environment.

Simulates a software development pipeline with specialized agents:
Product Manager → Architect → Engineer → QA → Reviewer.
"""

import logging
from typing import Callable, Dict, List, Tuple

from agents.base_agent import LLMAgent
from agents.transcript import Transcript
from agents.vllm_backend import VLLMBackend
from agents.workflow import SequentialWorkflow

LOGGER = logging.getLogger(__name__)


METAGPT_ROLES = [
    "You are the Product Manager. Analyze the requirements and write clear specifications for the coding task.",
    "You are the Software Architect. Design the solution architecture: classes, functions, data flow.",
    "You are the Senior Engineer. Implement the solution based on the architecture above. Write clean, correct code.",
    "You are the Code Reviewer. Review the implementation for bugs, edge cases, and improvements.",
    "You are the QA Engineer. Write test cases and verify the implementation handles all requirements.",
    "You are the Technical Writer. Document the solution and ensure the final answer is clear.",
    "You are the Integration Engineer. Ensure all components work together correctly.",
    "You are the Performance Optimizer. Optimize the solution for efficiency.",
    "You are the Security Reviewer. Check for potential security issues in the implementation.",
    "You are the Final Approver. Make the final decision on the solution quality and correctness.",
]


class MetaGPTEnvironment:
    """MetaGPT-style pipeline environment."""

    def __init__(
        self,
        backend: VLLMBackend,
        n_agents: int = 10,
        temperature: float = 0.7,
    ):
        self.backend = backend
        self.n_agents = n_agents
        self.temperature = temperature

    def create_agents(self) -> List[LLMAgent]:
        agents = []
        for i in range(self.n_agents):
            role = METAGPT_ROLES[i % len(METAGPT_ROLES)]
            agent = LLMAgent(
                agent_id=i,
                role_prompt=role,
                backend=self.backend,
                temperature=self.temperature,
                max_tokens=768,
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
