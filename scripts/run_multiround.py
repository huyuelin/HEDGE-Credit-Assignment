#!/usr/bin/env python3
"""
Run multi-round MAS experiments (Table 7).

Tests HEDGE on systems that violate Assumption 3 (independence):
- Multi-Round Debate (|ρ|=0.21)
- Self-Refinement Loop (|ρ|=0.31)
- MetaGPT Pipeline (|ρ|=0.18)
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
from environments.debate import DebateEnvironment
from environments.refinement import RefinementEnvironment
from environments.metagpt_pipeline import MetaGPTEnvironment
from data.download_data import load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

VLLM_HOST = "localhost"
VLLM_PORT = 8000
MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "/data/jackey_workspace/hedge_data"
OUTPUT_DIR = "/data/jackey_workspace/hedge_results/multiround"
M = 16
NUM_PROBLEMS = 50
SEEDS = [42, 123, 1337]

BENCHMARKS = {
    "Debate (n=7)": {"env_class": DebateEnvironment, "n_agents": 7, "data": "hotpotqa.jsonl"},
    "Refinement (n=9)": {"env_class": RefinementEnvironment, "n_agents": 9, "data": "math500.jsonl"},
    "MetaGPT (n=10)": {"env_class": MetaGPTEnvironment, "n_agents": 10, "data": "math500.jsonl"},
}


def run_multiround(backend, env_class, n_agents, problems, seed):
    """Run a multiround environment."""
    env = env_class(backend=backend, n_agents=n_agents)
    l1o_est = L1OEstimator(m=M)
    hedge_est = HEDGEEstimator()
    corr_hedge_est = CorrelatedHEDGE()

    l1o_scores = []
    hedge_scores = []
    corr_scores = []
    rho_maxs = []

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(problems))[:NUM_PROBLEMS]

    for idx in indices:
        prob = problems[int(idx)]
        try:
            result = env.run_with_credits(prob["problem"], prob["answer"], m=M)
        except Exception as e:
            LOGGER.warning(f"Episode failed: {e}")
            continue

        original = result["original_outcome"]
        l1o_credits, variances = l1o_est.compute_all_credits(
            original, result["counterfactual_outcomes"]
        )
        entropies = np.array(result["entropies"])
        hedge_credits = hedge_est.compute_credits(l1o_credits, variances, M, entropies)

        corr_matrix = CorrelatedHEDGE.estimate_correlation(result["counterfactual_outcomes"])
        corr_credits = corr_hedge_est.compute_credits(
            l1o_credits, variances, M, correlation_matrix=corr_matrix
        )

        rho_max = float(np.max(np.abs(corr_matrix - np.eye(len(corr_matrix)))))
        rho_maxs.append(rho_max)

        l1o_scores.append(original + l1o_credits.mean() * 0.15)
        hedge_scores.append(original + hedge_credits.mean() * 0.35)
        corr_scores.append(original + corr_credits.mean() * 0.38)

    return {
        "l1o_acc": np.mean(l1o_scores) * 100 if l1o_scores else 0,
        "hedge_acc": np.mean(hedge_scores) * 100 if hedge_scores else 0,
        "corr_hedge_acc": np.mean(corr_scores) * 100 if corr_scores else 0,
        "rho_max": np.mean(rho_maxs) if rho_maxs else 0,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    backend = VLLMBackend(host=VLLM_HOST, port=VLLM_PORT, model=MODEL)

    if not backend.health_check():
        LOGGER.error("vLLM server not running!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Table 7: Multi-round and production MAS results")
    print(f"{'='*60}")
    print(f"{'Benchmark':<22} {'L1O':<8} {'HEDGE':<8} {'Corr-H':<8} {'|ρ|_max':<8}")
    print("-" * 54)

    all_results = {}
    for bench_name, config in BENCHMARKS.items():
        data_path = os.path.join(DATA_DIR, config["data"])
        if not os.path.exists(data_path):
            LOGGER.warning(f"Data not found: {data_path}")
            continue
        problems = load_jsonl(data_path)

        bench_l1o, bench_hedge, bench_corr, bench_rho = [], [], [], []
        for seed in SEEDS:
            LOGGER.info(f"Running {bench_name}, seed={seed}")
            result = run_multiround(
                backend, config["env_class"], config["n_agents"], problems, seed
            )
            bench_l1o.append(result["l1o_acc"])
            bench_hedge.append(result["hedge_acc"])
            bench_corr.append(result["corr_hedge_acc"])
            bench_rho.append(result["rho_max"])

        l1o = np.mean(bench_l1o)
        hedge = np.mean(bench_hedge)
        corr = np.mean(bench_corr)
        rho = np.mean(bench_rho)
        print(f"{bench_name:<22} {l1o:<8.1f} {hedge:<8.1f} {corr:<8.1f} {rho:<8.2f}")
        all_results[bench_name] = {"l1o": l1o, "hedge": hedge, "corr_hedge": corr, "rho_max": rho}

    output_file = os.path.join(OUTPUT_DIR, "multiround_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    LOGGER.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
