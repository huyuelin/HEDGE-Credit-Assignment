"""
Transcript management for multi-agent systems.

A transcript records the sequence of (state, action, agent_id, entropy)
tuples produced during a multi-agent workflow execution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import copy
import math


@dataclass
class TranscriptStep:
    """A single step in the multi-agent transcript."""
    step_idx: int
    agent_id: int
    state: str
    action: str
    token_logprobs: List[float] = field(default_factory=list)
    entropy: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_entropy(self) -> float:
        """Compute Shannon entropy from token logprobs (nats, normalized to [0,1])."""
        if not self.token_logprobs:
            return 0.5
        probs = [math.exp(lp) for lp in self.token_logprobs]
        H = -sum(p * math.log(p + 1e-10) for p in probs if p > 0) / len(probs)
        max_H = math.log(2)
        self.entropy = min(H / max_H, 1.0) if max_H > 0 else 0.0
        return self.entropy


@dataclass
class Transcript:
    """Full transcript of a multi-agent workflow execution."""
    steps: List[TranscriptStep] = field(default_factory=list)
    outcome: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TranscriptStep):
        self.steps.append(step)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def num_agents(self) -> int:
        if not self.steps:
            return 0
        return max(s.agent_id for s in self.steps) + 1

    def get_state_at(self, step_idx: int) -> str:
        """Get the textual state just before step_idx."""
        if step_idx == 0:
            return self.steps[0].state
        parts = []
        for s in self.steps[:step_idx]:
            parts.append(s.action)
        return "\n".join(parts)

    def get_entropies(self) -> List[float]:
        return [s.entropy for s in self.steps]

    def replace_step(self, step_idx: int, new_action: str, new_logprobs: List[float] = None) -> "Transcript":
        """Create a new transcript with step_idx replaced by new_action.

        Returns a NEW transcript — subsequent steps are re-executed by the caller.
        For L1O counterfactuals, only replace the single step and rerun from there.
        """
        new_transcript = Transcript(
            steps=copy.deepcopy(self.steps[:step_idx]),
            metadata=copy.deepcopy(self.metadata),
        )
        replaced = copy.deepcopy(self.steps[step_idx])
        replaced.action = new_action
        if new_logprobs is not None:
            replaced.token_logprobs = new_logprobs
            replaced.compute_entropy()
        new_transcript.add_step(replaced)
        return new_transcript

    def to_text(self) -> str:
        """Convert transcript to full text for evaluation."""
        return "\n".join(s.action for s in self.steps)
