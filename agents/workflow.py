"""
Workflow engine for cooperative multi-agent systems.

Supports:
- Sequential workflow: agents take turns in order
- Parallel workflow: agents work independently (PIST-MAS)
- Resampling at any step for counterfactual evaluation
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

from agents.base_agent import LLMAgent
from agents.transcript import Transcript, TranscriptStep
from agents.vllm_backend import VLLMBackend

LOGGER = logging.getLogger(__name__)


class SequentialWorkflow:
    """Sequential multi-agent workflow.

    Agents take turns in a fixed order. Each agent sees the full
    transcript of previous actions.
    """

    def __init__(
        self,
        agents: List[LLMAgent],
        outcome_fn: Callable[[Transcript], float],
        task_prompt: str = "",
    ):
        self.agents = agents
        self.outcome_fn = outcome_fn
        self.task_prompt = task_prompt

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    def run(self) -> Transcript:
        """Execute the full workflow, returning the transcript."""
        transcript = Transcript()
        for step_idx, agent in enumerate(self.agents):
            state = transcript.get_state_at(step_idx) if step_idx > 0 else ""
            step = agent.act(
                state=state,
                task_prompt=self.task_prompt,
                step_idx=step_idx,
            )
            transcript.add_step(step)

        transcript.outcome = self.outcome_fn(transcript)
        return transcript

    def run_counterfactual(
        self,
        original_transcript: Transcript,
        replace_step: int,
        replacement_action: str,
        replacement_logprobs: List[float] = None,
    ) -> Transcript:
        """Run a counterfactual: replace action at step t and re-execute from t+1.

        This is the key operation for L1O credit estimation.
        """
        cf_transcript = Transcript()

        for step_idx in range(original_transcript.num_steps):
            if step_idx < replace_step:
                cf_transcript.add_step(original_transcript.steps[step_idx])
            elif step_idx == replace_step:
                replaced = TranscriptStep(
                    step_idx=step_idx,
                    agent_id=original_transcript.steps[step_idx].agent_id,
                    state=original_transcript.steps[step_idx].state,
                    action=replacement_action,
                    token_logprobs=replacement_logprobs or [],
                )
                replaced.compute_entropy()
                cf_transcript.add_step(replaced)
            else:
                state = cf_transcript.get_state_at(step_idx)
                agent = self.agents[step_idx % self.n_agents]
                step = agent.act(
                    state=state,
                    task_prompt=self.task_prompt,
                    step_idx=step_idx,
                )
                cf_transcript.add_step(step)

        cf_transcript.outcome = self.outcome_fn(cf_transcript)
        return cf_transcript

    def generate_counterfactuals(
        self,
        original_transcript: Transcript,
        step_idx: int,
        m: int = 16,
    ) -> List[Transcript]:
        """Generate m counterfactual transcripts by resampling at step_idx."""
        agent = self.agents[step_idx % self.n_agents]
        state = original_transcript.steps[step_idx].state

        resampled_steps = agent.resample(
            state=state,
            task_prompt=self.task_prompt,
            step_idx=step_idx,
            n=m,
        )

        counterfactuals = []
        for rs in resampled_steps:
            cf = self.run_counterfactual(
                original_transcript,
                step_idx,
                rs.action,
                rs.token_logprobs,
            )
            counterfactuals.append(cf)

        return counterfactuals


class ParallelWorkflow:
    """Parallel independent workflow (for PIST-MAS).

    Each agent solves an independent subtask. No inter-agent communication.
    """

    def __init__(
        self,
        agents: List[LLMAgent],
        subtask_prompts: List[str],
        subtask_outcome_fns: List[Callable[[str], float]],
        aggregate_fn: Callable[[List[float]], float] = None,
    ):
        self.agents = agents
        self.subtask_prompts = subtask_prompts
        self.subtask_outcome_fns = subtask_outcome_fns
        self.aggregate_fn = aggregate_fn or (lambda scores: sum(scores) / len(scores))

    def run(self) -> Transcript:
        """Execute parallel independent subtasks."""
        transcript = Transcript()

        for step_idx, (agent, prompt, outcome_fn) in enumerate(
            zip(self.agents, self.subtask_prompts, self.subtask_outcome_fns)
        ):
            step = agent.act(state="", task_prompt=prompt, step_idx=step_idx)
            transcript.add_step(step)

        sub_outcomes = []
        for step, outcome_fn in zip(transcript.steps, self.subtask_outcome_fns):
            sub_outcomes.append(outcome_fn(step.action))

        transcript.outcome = self.aggregate_fn(sub_outcomes)
        transcript.metadata["sub_outcomes"] = sub_outcomes
        return transcript

    def run_counterfactual(
        self,
        original_transcript: Transcript,
        replace_step: int,
        replacement_action: str,
        replacement_logprobs: List[float] = None,
    ) -> Transcript:
        """Counterfactual for parallel: only the replaced step changes outcome."""
        cf_transcript = Transcript()

        for step_idx in range(original_transcript.num_steps):
            if step_idx == replace_step:
                replaced = TranscriptStep(
                    step_idx=step_idx,
                    agent_id=original_transcript.steps[step_idx].agent_id,
                    state="",
                    action=replacement_action,
                    token_logprobs=replacement_logprobs or [],
                )
                replaced.compute_entropy()
                cf_transcript.add_step(replaced)
            else:
                cf_transcript.add_step(original_transcript.steps[step_idx])

        sub_outcomes = []
        for step, outcome_fn in zip(cf_transcript.steps, self.subtask_outcome_fns):
            sub_outcomes.append(outcome_fn(step.action))

        cf_transcript.outcome = self.aggregate_fn(sub_outcomes)
        cf_transcript.metadata["sub_outcomes"] = sub_outcomes
        return cf_transcript

    def generate_counterfactuals(
        self,
        original_transcript: Transcript,
        step_idx: int,
        m: int = 16,
    ) -> List[Transcript]:
        """Generate m counterfactuals by resampling the agent at step_idx."""
        agent = self.agents[step_idx]
        prompt = self.subtask_prompts[step_idx]

        resampled_steps = agent.resample(
            state="",
            task_prompt=prompt,
            step_idx=step_idx,
            n=m,
        )

        counterfactuals = []
        for rs in resampled_steps:
            cf = self.run_counterfactual(
                original_transcript,
                step_idx,
                rs.action,
                rs.token_logprobs,
            )
            counterfactuals.append(cf)

        return counterfactuals
