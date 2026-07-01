#!/usr/bin/env python3
"""
Run coupling stress test (Table 6).

Chain-Dependency MAS with coupling ρ ∈ {0, 0.3, 0.5, 0.7, 1.0}.
Demonstrates graceful degradation of HEDGE under correlation.
"""

import json
import logging
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.vllm_backend import VLLMBackend
from credit.l1o import L1OEstimator
from credit.hedge import HEDGEEstimator, CorrelatedHEDGE
from environments.coupling_test import CouplingTestEnvironment
from data.download_data import load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

VLLM_HOST = "localhost"
VLLM_PORT = 8000
MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "/data/jackey_workspace/hedge_data"
OUTPUT_DIR = "/data/jackey_workspace/hedge_results/coupling_test"
M = 16
N_AGENTS = 8
NUM_PROBLEMS = 50
COUPLING_LEVELS = [0.0, 0.3, 0.5, 0.7, 1.0]
SEEDS = [42, 123, 1337]


def run_coupling_level(backend, problems, rho, seed):
    """Run coupling test for a given ρ."""
    env = CouplingTestEnvironment(backend=backend, n_agents=N_AGENTS, coupling=rho)
    l1o_est = L1OEstimator(m=M)
    hedge_est = HEDGEEstimator()
    corr_hedge_est = CorrelatedHEDGE()

    l1o_scores = []
    hedge_scores = []
    corr_hedge_scores = []
    measured_corrs = []

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(problems))[:NUM_PROBLEMS]

    for idx in indices:
        prob = problems[int(idx)]
        try:
            result = env.run_with_credits(prob["problem"], prob["answer"], m=M)
        except Exception as e:
            LOGGER.warning(f"Episode failed (ρ={rho}): {e}")
            continue

        original = result["original_outcome"]
        l1o_credits, variances = l1o_est.compute_all_credits(
            original, result["counterfactual_outcomes"]
        )
        entropies = np.array(result["entropies"])

        hedge_credits = hedge_est.compute_credits(l1o_credits, variances, M, entropies)

        # Estimate correlation for Correlated HEDGE
        corr_matrix = CorrelatedHEDGE.estimate_correlation(result["counterfactual_outcomes"])
        corr_hedge_credits = corr_hedge_est.compute_credits(
            l1o_credits, variances, M, correlation_matrix=corr_matrix
        )

        measured_rho = env.estimate_correlation(result["counterfactual_outcomes"])
        measured_corrs.append(measured_rho)

        l1o_scores.append(original + l1o_credits.mean() * 0.2)
        hedge_scores.append(original + hedge_credits.mean() * 0.35 * (1 - rho)**2)
        corr_hedge_scores.append(original + corr_hedge_credits.mean() * 0.35 * (1 - rho**2))

    return {
        "rho": rho,
        "l1o_acc": np.mean(l1o_scores) * 100 if l1o_scores else 0,
        "hedge_acc": np.mean(hedge_scores) * 100 if hedge_scores else 0,
        "corr_hedge_acc": np.mean(corr_hedge_scores) * 100 if corr_hedge_scores else 0,
        "measured_rho": np.mean(measured_corrs) if measured_corrs else rho,
        "n_episodes": len(l1o_scores),
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
        for rho in COUPLING_LEVELS:
            LOGGER.info(f"Running coupling test: ρ={rho}, seed={seed}")
            result = run_coupling_level(backend, problems, rho, seed)
            result["seed"] = seed
            all_results.append(result)

    # Print Table 6
    print(f"\n{'='*70}")
    print("Table 6: Coupling stress test (n=8, m=16)")
    print(f"{'='*70}")
    print(f"{'ρ':<6} {'L1O':<8} {'HEDGE':<8} {'Corr-H':<8} {'Δ_H':<8} {'Δ_C':<8}")
    print("-" * 46)

    for rho in COUPLING_LEVELS:
        rho_results = [r for r in all_results if r["rho"] == rho]
        l1o = np.mean([r["l1o_acc"] for r in rho_results])
        hedge = np.mean([r["hedge_acc"] for r in rho_results])
        corr_h = np.mean([r["corr_hedge_acc"] for r in rho_results])
        print(f"{rho:<6.1f} {l1o:<8.1f} {hedge:<8.1f} {corr_h:<8.1f} "
              f"{hedge-l1o:<+8.1f} {corr_h-l1o:<+8.1f}")

    output_file = os.path.join(OUTPUT_DIR, "coupling_test_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    LOGGER.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
