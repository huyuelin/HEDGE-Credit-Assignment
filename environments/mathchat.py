"""
MathChat multi-agent environment.

n agents collaborate sequentially to solve math problems.
Each agent sees the full transcript of previous agents' work.
The last agent's answer determines the outcome.
"""

import logging
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from agents.base_agent import LLMAgent
from agents.transcript import Transcript
from agents.vllm_backend import VLLMBackend
from agents.workflow import SequentialWorkflow

LOGGER = logging.getLogger(__name__)


MATH_AGENT_ROLES = [
    "You are the Problem Analyzer. Read the math problem carefully and identify the key information, constraints, and what needs to be solved. Outline a solution strategy.",
    "You are the Solution Developer. Based on the analysis above, work through the solution step by step with detailed calculations.",
    "You are the Computation Verifier. Check the calculations above for errors. If you find mistakes, correct them. If correct, confirm and simplify.",
    "You are the Answer Formatter. Review the solution above, verify the final answer, and present it clearly in \\boxed{} format.",
    "You are a Mathematical Reasoner. Continue developing the solution. Add alternative approaches or verify steps.",
    "You are a Problem Decomposer. Break complex steps into simpler sub-problems if needed.",
    "You are a Logical Validator. Check logical consistency of each step in the solution.",
    "You are a Solution Synthesizer. Combine insights from all previous steps into a coherent answer.",
    "You are an Error Detector. Look for common mathematical errors (sign errors, arithmetic mistakes, etc.).",
    "You are a Final Reviewer. Give the definitive final answer after reviewing all work above.",
    "You are an Alternative Approach Expert. Try a different method to verify the answer.",
    "You are a Precision Specialist. Ensure all numerical computations are exact.",
]


def extract_math_answer(text: str) -> Optional[str]:
    """Extract answer from text, looking for \\boxed{}."""
    boxed = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if boxed:
        return boxed[-1].strip()
    numbers = re.findall(r'(?:answer|Answer|=)\s*[-]?\d+(?:\.\d+)?(?:/\d+)?', text)
    if numbers:
        return numbers[-1].split()[-1]
    return None


class MathChatEnvironment:
    """MathChat multi-agent sequential environment."""

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
        """Create n sequential math agents with different roles."""
        agents = []
        for i in range(self.n_agents):
            role = MATH_AGENT_ROLES[i % len(MATH_AGENT_ROLES)]
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
        """Create outcome function that checks correctness."""
        def outcome_fn(transcript: Transcript) -> float:
            full_text = transcript.to_text()
            predicted = extract_math_answer(full_text)
            if predicted is None:
                return 0.0
            try:
                pred_clean = predicted.replace(",", "").strip()
                gold_clean = gold_answer.replace(",", "").strip()
                if pred_clean == gold_clean:
                    return 1.0
                pred_f = float(pred_clean)
                gold_f = float(gold_clean)
                return 1.0 if abs(pred_f - gold_f) < 1e-6 else 0.0
            except ValueError:
                return 1.0 if predicted.strip() == gold_answer.strip() else 0.0
        return outcome_fn

    def run_episode(
        self,
        problem: str,
        gold_answer: str,
    ) -> Tuple[Transcript, SequentialWorkflow]:
        """Run one MathChat episode."""
        agents = self.create_agents()
        outcome_fn = self.make_outcome_fn(gold_answer)

        workflow = SequentialWorkflow(
            agents=agents,
            outcome_fn=outcome_fn,
            task_prompt=f"Solve this math problem:\n{problem}",
        )

        transcript = workflow.run()
        return transcript, workflow

    def run_with_credits(
        self,
        problem: str,
        gold_answer: str,
        m: int = 16,
    ) -> Dict:
        """Run episode and compute counterfactuals."""
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
        }
