"""
HotpotQA multi-agent environment.

n agents collaborate to answer multi-hop questions.
Uses F1 score as outcome metric.
"""

import logging
import re
import string
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from agents.base_agent import LLMAgent
from agents.transcript import Transcript
from agents.vllm_backend import VLLMBackend
from agents.workflow import SequentialWorkflow

LOGGER = logging.getLogger(__name__)


HOTPOTQA_ROLES = [
    "You are the Question Decomposer. Break down this multi-hop question into simpler sub-questions that can be answered independently.",
    "You are the Information Gatherer. Based on the decomposed questions above, identify what facts are needed and provide relevant information from your knowledge.",
    "You are the Reasoning Agent. Use the gathered information to reason through the sub-questions and build toward the final answer.",
    "You are the Answer Synthesizer. Combine all reasoning above into a clear, concise final answer.",
    "You are a Fact Verifier. Check if the reasoning above is consistent and well-supported.",
    "You are a Bridge Entity Identifier. Find the connecting entities between different parts of the question.",
    "You are an Alternative Reasoning Agent. Try a different reasoning path to verify the answer.",
    "You are a Final Answer Agent. Provide the definitive answer based on all reasoning above.",
    "You are a Consistency Checker. Verify logical consistency across all reasoning steps.",
    "You are a Confidence Assessor. Rate confidence in the answer and flag uncertainties.",
    "You are a Supporting Evidence Agent. Provide additional evidence for the answer.",
    "You are a Summary Agent. Provide a brief summary of the answer with key supporting facts.",
]


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def extract_answer(text: str) -> str:
    """Extract final answer from generated text."""
    patterns = [
        r'(?:final answer|answer)\s*(?:is|:)\s*(.+?)(?:\.|$)',
        r'(?:therefore|thus|so)\s*,?\s*(.+?)(?:\.|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    lines = text.strip().split('\n')
    return lines[-1].strip()


class HotpotQAEnvironment:
    """HotpotQA multi-agent sequential environment."""

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
            role = HOTPOTQA_ROLES[i % len(HOTPOTQA_ROLES)]
            agent = LLMAgent(
                agent_id=i,
                role_prompt=role,
                backend=self.backend,
                temperature=self.temperature,
                max_tokens=512,
            )
            agents.append(agent)
        return agents

    def make_outcome_fn(self, gold_answer: str) -> Callable[[Transcript], float]:
        def outcome_fn(transcript: Transcript) -> float:
            full_text = transcript.to_text()
            predicted = extract_answer(full_text)
            return f1_score(predicted, gold_answer)
        return outcome_fn

    def run_episode(
        self,
        question: str,
        gold_answer: str,
    ) -> Tuple[Transcript, SequentialWorkflow]:
        agents = self.create_agents()
        outcome_fn = self.make_outcome_fn(gold_answer)

        workflow = SequentialWorkflow(
            agents=agents,
            outcome_fn=outcome_fn,
            task_prompt=f"Answer this question:\n{question}",
        )

        transcript = workflow.run()
        return transcript, workflow

    def run_with_credits(
        self,
        question: str,
        gold_answer: str,
        m: int = 16,
    ) -> Dict:
        transcript, workflow = self.run_episode(question, gold_answer)

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
