#!/usr/bin/env python3
"""
Quick pipeline test — runs a minimal PIST-MAS experiment
using transformers backend (no vLLM needed).

Usage: python scripts/test_pipeline.py
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def main():
    from agents.transformers_backend import TransformersBackend
    from agents.base_agent import LLMAgent
    from agents.workflow import ParallelWorkflow
    from agents.transcript import Transcript
    from credit.l1o import L1OEstimator
    from credit.hedge import HEDGEEstimator
    from environments.pist_mas import make_gsm8k_outcome_fn
    from data.download_data import load_jsonl

    MODEL_PATH = "/data/jackey_workspace/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/" + \
        os.listdir("/data/jackey_workspace/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/")[0]

    LOGGER.info(f"Loading model from: {MODEL_PATH}")
    backend = TransformersBackend(model_name_or_path=MODEL_PATH)

    problems = load_jsonl("/data/jackey_workspace/hedge_data/gsm8k.jsonl")[:4]
    LOGGER.info(f"Using {len(problems)} problems for quick test")

    # Create 2 agents (smallest PIST-MAS)
    n = 2
    m = 4  # fewer counterfactuals for quick test

    agents = []
    for i in range(n):
        agent = LLMAgent(
            agent_id=i,
            role_prompt="You are a math problem solver. Solve step by step. Put answer in \\boxed{}.",
            backend=backend,
            temperature=0.7,
            max_tokens=256,
        )
        agents.append(agent)

    subtask_prompts = [p["problem"] for p in problems[:n]]
    outcome_fns = [make_gsm8k_outcome_fn(p["answer"]) for p in problems[:n]]

    workflow = ParallelWorkflow(
        agents=agents,
        subtask_prompts=subtask_prompts,
        subtask_outcome_fns=outcome_fns,
    )

    LOGGER.info("Running original episode...")
    transcript = workflow.run()
    LOGGER.info(f"Original outcome: {transcript.outcome:.2f}")
    LOGGER.info(f"Sub-outcomes: {transcript.metadata.get('sub_outcomes', [])}")
    LOGGER.info(f"Entropies: {[f'{e:.3f}' for e in transcript.get_entropies()]}")

    # Generate counterfactuals
    LOGGER.info(f"\nGenerating {m} counterfactuals per step...")
    all_cf_outcomes = []
    for step_idx in range(n):
        cfs = workflow.generate_counterfactuals(transcript, step_idx, m=m)
        cf_outcomes = [cf.outcome for cf in cfs]
        all_cf_outcomes.append(cf_outcomes)
        LOGGER.info(f"  Step {step_idx}: CF outcomes = {[f'{o:.2f}' for o in cf_outcomes]}")

    # Compute credits
    l1o_est = L1OEstimator(m=m)
    hedge_est = HEDGEEstimator()

    l1o_credits, variances = l1o_est.compute_all_credits(transcript.outcome, all_cf_outcomes)
    entropies = np.array(transcript.get_entropies())
    hedge_credits = hedge_est.compute_credits(l1o_credits, variances, m, entropies)

    LOGGER.info(f"\n=== Credit Results ===")
    LOGGER.info(f"L1O credits:   {l1o_credits}")
    LOGGER.info(f"Variances:     {variances}")
    LOGGER.info(f"HEDGE credits: {hedge_credits}")
    LOGGER.info(f"Entropies:     {entropies}")

    LOGGER.info(f"\n[PASS] Pipeline test complete!")
    return True


if __name__ == "__main__":
    main()
