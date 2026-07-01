"""
DSBench multi-agent environment for data science problems.
"""

import logging
from typing import Callable, Dict, List, Tuple

from agents.base_agent import LLMAgent
from agents.transcript import Transcript
from agents.vllm_backend import VLLMBackend
from agents.workflow import SequentialWorkflow

LOGGER = logging.getLogger(__name__)


DSBENCH_ROLES = [
    "You are the Data Analyst. Understand the dataset and the question being asked. Identify relevant features.",
    "You are the Statistician. Choose appropriate statistical methods or models for the analysis.",
    "You are the Python Coder. Write clean Python/pandas code to perform the analysis.",
    "You are the Results Interpreter. Interpret the output and provide a clear answer.",
    "You are the Validation Expert. Verify the analysis methodology and check for errors.",
    "You are the Report Writer. Summarize findings clearly with the final numerical answer.",
    "You are the Data Cleaning Specialist. Ensure data preprocessing is correct.",
    "You are the Visualization Expert. Suggest appropriate visualizations and verify trends.",
]


class DSBenchEnvironment:
    """DSBench multi-agent environment."""

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
        agents = []
        for i in range(self.n_agents):
            role = DSBENCH_ROLES[i % len(DSBENCH_ROLES)]
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
                if gold_f == 0:
                    return 1.0 if abs(pred_f) < 1e-6 else 0.0
                rel_error = abs(pred_f - gold_f) / abs(gold_f)
                return 1.0 if rel_error < 0.01 else 0.0
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
