"""
Multi-Round Debate environment.

n agents engage in multi-round debate to answer questions.
Correlation arises from agents responding to each other's arguments.
"""

import logging
from typing import Callable, Dict, List, Tuple

from agents.base_agent import LLMAgent
from agents.transcript import Transcript
from agents.vllm_backend import VLLMBackend
from agents.workflow import SequentialWorkflow

LOGGER = logging.getLogger(__name__)


DEBATE_ROLES = [
    "You are Debater A. Present your argument for the correct answer. Be concise and logical.",
    "You are Debater B. Challenge the previous argument if wrong, or support it if correct. Present your reasoning.",
    "You are Debater C. Synthesize the arguments above and present a balanced view.",
    "You are the Moderator. Based on the debate above, determine the correct answer.",
]


class DebateEnvironment:
    """Multi-round debate with n agents."""

    def __init__(
        self,
        backend: VLLMBackend,
        n_agents: int = 7,
        temperature: float = 0.7,
    ):
        self.backend = backend
        self.n_agents = n_agents
        self.temperature = temperature

    def create_agents(self) -> List[LLMAgent]:
        agents = []
        for i in range(self.n_agents):
            role = DEBATE_ROLES[i % len(DEBATE_ROLES)]
            agent = LLMAgent(
                agent_id=i,
                role_prompt=f"Round {i // len(DEBATE_ROLES) + 1}. {role}",
                backend=self.backend,
                temperature=self.temperature,
                max_tokens=512,
            )
            agents.append(agent)
        return agents

    def make_outcome_fn(self, gold_answer: str) -> Callable[[Transcript], float]:
        from environments.hotpotqa import f1_score, extract_answer
        def outcome_fn(transcript: Transcript) -> float:
            full_text = transcript.to_text()
            predicted = extract_answer(full_text)
            return f1_score(predicted, gold_answer)
        return outcome_fn

    def run_with_credits(self, question: str, gold_answer: str, m: int = 16) -> Dict:
        agents = self.create_agents()
        outcome_fn = self.make_outcome_fn(gold_answer)
        workflow = SequentialWorkflow(agents=agents, outcome_fn=outcome_fn, task_prompt=question)
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
