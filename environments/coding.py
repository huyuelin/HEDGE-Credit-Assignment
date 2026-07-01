"""
Coding environments: APPS, SWE-Bench, BFCL.

Multi-agent coding with sequential collaboration.
"""

import logging
import re
from typing import Callable, Dict, List, Tuple

from agents.base_agent import LLMAgent
from agents.transcript import Transcript
from agents.vllm_backend import VLLMBackend
from agents.workflow import SequentialWorkflow

LOGGER = logging.getLogger(__name__)


CODING_ROLES = [
    "You are the Problem Analyzer. Read the coding problem and identify input/output format, constraints, and edge cases.",
    "You are the Algorithm Designer. Design an efficient algorithm to solve the problem. Describe the approach clearly.",
    "You are the Implementation Expert. Write clean, correct code implementing the algorithm above.",
    "You are the Test Writer. Write test cases including edge cases to verify the solution.",
    "You are the Code Debugger. Review the code for bugs and fix any issues found.",
    "You are the Optimizer. Improve time/space complexity if possible while maintaining correctness.",
    "You are the Final Coder. Produce the final, clean implementation ready for submission.",
    "You are a Code Reviewer. Check for correctness, edge cases, and formatting.",
]


class CodingEnvironment:
    """Multi-agent coding environment for APPS/SWE-Bench/BFCL."""

    def __init__(
        self,
        backend: VLLMBackend,
        n_agents: int = 8,
        temperature: float = 0.7,
        benchmark: str = "apps",
    ):
        self.backend = backend
        self.n_agents = n_agents
        self.temperature = temperature
        self.benchmark = benchmark

    def create_agents(self) -> List[LLMAgent]:
        agents = []
        for i in range(self.n_agents):
            role = CODING_ROLES[i % len(CODING_ROLES)]
            agent = LLMAgent(
                agent_id=i,
                role_prompt=role,
                backend=self.backend,
                temperature=self.temperature,
                max_tokens=1024,
            )
            agents.append(agent)
        return agents

    def make_outcome_fn(self, test_cases: List[Dict] = None) -> Callable[[Transcript], float]:
        """Create outcome function based on test case execution."""
        def outcome_fn(transcript: Transcript) -> float:
            full_text = transcript.to_text()
            code_blocks = re.findall(r'```(?:python)?\n(.*?)```', full_text, re.DOTALL)
            if not code_blocks:
                return 0.0
            if test_cases:
                passed = 0
                for tc in test_cases:
                    try:
                        # Simplified: check if code contains expected patterns
                        if tc.get("expected_output", "") in full_text:
                            passed += 1
                    except Exception:
                        pass
                return passed / len(test_cases) if test_cases else 0.0
            return 0.5 if code_blocks else 0.0
        return outcome_fn

    def run_with_credits(
        self,
        problem: str,
        test_cases: List[Dict] = None,
        m: int = 16,
    ) -> Dict:
        agents = self.create_agents()
        outcome_fn = self.make_outcome_fn(test_cases)
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
