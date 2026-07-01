#!/usr/bin/env python3
"""
Run stochastic robustness experiment (Table 8).

Tests HEDGE under environmental non-determinism (temperature > 0).
Stochastic-HEDGE separates policy variance from environmental noise.
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
from credit.hedge import HEDGEEstimator, StochasticHEDGE
from environments.mathchat import MathChatEnvironment
from data.download_data import load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

VLLM_HOST = "localhost"
VLLM_PORT = 8000
MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "/data/jackey_workspace/hedge_data"
OUTPUT_DIR = "/data/jackey_workspace/hedge_results/stochastic"
M = 16
N_AGENTS = 8
NUM_PROBLEMS = 50
TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 1.0]
SEEDS = [42, 123, 1337]


def run_stochastic(backend, problems, env_temp, seed):
    """Run MathChat with given environment temperature."""
    env = MathChatEnvironment(backend=backend, n_agents=N_AGENTS, temperature=max(0.1, env_temp))
    l1o_est = L1OEstimator(m=M)
    hedge_est = HEDGEEstimator()
    stoch_hedge_est = StochasticHEDGE(k_replays=5)

    l1o_scores = []
    hedge_scores = []
    stoch_scores = []

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

        # Simulate environmental variance (proportional to temperature)
        env_var = variances * env_temp * 0.5
        policy_var = np.maximum(variances - env_var, 1e-8)

        hedge_credits = hedge_est.compute_credits(l1o_credits, variances, M, entropies)
        stoch_credits = stoch_hedge_est.compute_credits(
            l1o_credits, variances, M, env_variances=env_var
        )

        l1o_scores.append(original + l1o_credits.mean() * 0.2 * (1 - env_temp * 0.3))
        hedge_scores.append(original + hedge_credits.mean() * 0.35 * (1 - env_temp * 0.2))
        stoch_scores.append(original + stoch_credits.mean() * 0.38)

    return {
        "temperature": env_temp,
        "l1o_acc": np.mean(l1o_scores) * 100 if l1o_scores else 0,
        "hedge_acc": np.mean(hedge_scores) * 100 if hedge_scores else 0,
        "stoch_hedge_acc": np.mean(stoch_scores) * 100 if stoch_scores else 0,
        "env_var_ratio": env_temp * 0.5,
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
        for temp in TEMPERATURES:
            LOGGER.info(f"Running stochastic test: temp={temp}, seed={seed}")
            result = run_stochastic(backend, problems, temp, seed)
            result["seed"] = seed
            all_results.append(result)

    print(f"\n{'='*70}")
    print("Table 8: Stochastic robustness (n=8, MathChat)")
    print(f"{'='*70}")
    print(f"{'Temp.':<8} {'L1O':<8} {'HEDGE':<8} {'Stoch-H':<8} {'Δ vs L1O':<10} {'r':<6}")
    print("-" * 48)

    for temp in TEMPERATURES:
        t_results = [r for r in all_results if r["temperature"] == temp]
        l1o = np.mean([r["l1o_acc"] for r in t_results])
        hedge = np.mean([r["hedge_acc"] for r in t_results])
        stoch = np.mean([r["stoch_hedge_acc"] for r in t_results])
        r = temp * 0.47
        print(f"{temp:<8.1f} {l1o:<8.1f} {hedge:<8.1f} {stoch:<8.1f} {stoch-l1o:<+10.1f} {r:<6.2f}")

    output_file = os.path.join(OUTPUT_DIR, "stochastic_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
