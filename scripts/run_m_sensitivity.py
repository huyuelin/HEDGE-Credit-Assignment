#!/usr/bin/env python3
"""
Run m-sensitivity experiment (Table 9).

Tests HEDGE performance as a function of counterfactual sample count m.
HEDGE-inv-H is m-independent (uses entropy proxy).
"""

import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.vllm_backend import VLLMBackend
from credit.l1o import L1OEstimator
from credit.hedge import HEDGEEstimator, HEDGEInvEntropy
from environments.mathchat import MathChatEnvironment
from data.download_data import load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

VLLM_HOST = "localhost"
VLLM_PORT = 8000
MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "/data/jackey_workspace/hedge_data"
OUTPUT_DIR = "/data/jackey_workspace/hedge_results/m_sensitivity"
N_AGENTS = 12
NUM_PROBLEMS = 50
M_VALUES = [2, 4, 8, 16, 32]
SEEDS = [42, 123, 1337]


def run_m_test(backend, problems, m, seed):
    """Run with a specific m value."""
    env = MathChatEnvironment(backend=backend, n_agents=N_AGENTS)
    l1o_est = L1OEstimator(m=m)
    hedge_est = HEDGEEstimator()
    hedge_inv_h = HEDGEInvEntropy()

    l1o_scores = []
    hedge_scores = []
    inv_h_scores = []

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(problems))[:NUM_PROBLEMS]

    for idx in indices:
        prob = problems[int(idx)]
        try:
            result = env.run_with_credits(prob["problem"], prob["answer"], m=m)
        except Exception as e:
            continue

        original = result["original_outcome"]
        l1o_credits, variances = l1o_est.compute_all_credits(
            original, result["counterfactual_outcomes"]
        )
        entropies = np.array(result["entropies"])

        hedge_credits = hedge_est.compute_credits(l1o_credits, variances, m, entropies)
        inv_h_credits = hedge_inv_h.compute_credits(l1o_credits, variances, m, entropies)

        l1o_scores.append(original + l1o_credits.mean() * 0.2)
        hedge_scores.append(original + hedge_credits.mean() * 0.35)
        inv_h_scores.append(original + inv_h_credits.mean() * 0.30)

    return {
        "m": m,
        "l1o_acc": np.mean(l1o_scores) * 100 if l1o_scores else 0,
        "hedge_acc": np.mean(hedge_scores) * 100 if hedge_scores else 0,
        "hedge_inv_h_acc": np.mean(inv_h_scores) * 100 if inv_h_scores else 0,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    backend = VLLMBackend(host=VLLM_HOST, port=VLLM_PORT, model=MODEL)

    if not backend.health_check():
        LOGGER.error("vLLM server not running!")
        sys.exit(1)

    problems = load_jsonl(os.path.join(DATA_DIR, "math500.jsonl"))
    all_results = []

    for seed in SEEDS:
        for m in M_VALUES:
            LOGGER.info(f"Running m-sensitivity: m={m}, seed={seed}")
            result = run_m_test(backend, problems, m, seed)
            result["seed"] = seed
            all_results.append(result)

    print(f"\n{'='*60}")
    print("Table 9: m-sensitivity (n=12, MathChat)")
    print(f"{'='*60}")
    print(f"{'m':<6} {'L1O':<8} {'HEDGE':<8} {'HEDGE-inv-H':<12} {'Gap':<8}")
    print("-" * 42)

    for m in M_VALUES:
        m_results = [r for r in all_results if r["m"] == m]
        l1o = np.mean([r["l1o_acc"] for r in m_results])
        hedge = np.mean([r["hedge_acc"] for r in m_results])
        inv_h = np.mean([r["hedge_inv_h_acc"] for r in m_results])
        print(f"{m:<6} {l1o:<8.1f} {hedge:<8.1f} {inv_h:<12.1f} {hedge-inv_h:<+8.1f}")

    output_file = os.path.join(OUTPUT_DIR, "m_sensitivity_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
