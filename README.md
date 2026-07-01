# HEDGE: Heteroscedastic Entropy-Driven Group Estimation of Credit in LLM Multi-Agent Systems

<p align="center">
  <img src="figures/architecture/HEDGE_Pipeline_Overview.pdf" width="90%" alt="HEDGE Pipeline Architecture"/>
</p>

> **When Credit Collapses: Heteroscedastic, Entropy-Driven Group Estimation of Credit in LLM Multi-Agent Systems**
>
> *Under review at AAAI 2027*

## Overview

This repository provides the official implementation of **HEDGE**, a hyperparameter-free James–Stein shrinkage estimator for per-step credit assignment in cooperative LLM multi-agent systems. HEDGE addresses the *credit collapse* phenomenon—where leave-one-out (L1O) counterfactual credit degrades and eventually inverts as team size grows—by exploiting the heteroscedastic structure of policy-entropy-governed variance.

### Key Contributions

- **Causal Evidence**: PIST-MAS (Parallel-Independent-Subtask MAS) demonstrates that ≥78% of credit collapse is estimator-driven, not coordination overhead
- **Theory**: Near-minimax-optimal shrinkage under heteroscedastic noise; graceful degradation bounds under inter-agent coupling; all 24 core theorems formalized in Lean 4 (2,917 lines verified)
- **Method**: Base HEDGE + Bias-Tolerant, Correlated, Stochastic, Adaptive, and Graph-structured variants
- **Evidence**: Ten benchmarks, three backbones (Llama-3.1-70B, Qwen-2.5-72B, GPT-4o-mini), comprehensive baselines including Math-Shepherd PRM and MCTS credit

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HEDGE Pipeline                                │
├──────────────────┬──────────────────────┬───────────────────────────┤
│   Credit Collapse│  Heteroscedastic     │   James-Stein Shrinkage   │
│   Diagnosis      │  Variance Model      │   Correction              │
│                  │                      │                           │
│  L1O gain drops  │  σ²(cₜ) ∝ H(πₜ)^α  │  ĉₜ^HEDGE = (1-wₜ)·ĉₜ   │
│  sign-flip @ n≈8 │  Power-law entropy   │        + wₜ·μ_group      │
│                  │  heteroscedasticity  │  wₜ = σ̂²ₜ/(σ̂²ₜ + τ²)    │
└──────────────────┴──────────────────────┴───────────────────────────┘
```

## Repository Structure

```
├── credit/              # Core HEDGE estimator and baselines
│   ├── hedge.py         # HEDGE (all variants)
│   ├── l1o.py           # Leave-one-out counterfactual baseline
│   ├── entropy.py       # Policy entropy computation
│   └── baselines.py     # Uniform, Shapley, PRM, MCTS baselines
├── agents/              # LLM agent backends
│   ├── base_agent.py    # Abstract agent interface
│   ├── vllm_backend.py  # vLLM inference backend
│   └── workflow.py      # Multi-agent workflow orchestrator
├── environments/        # Benchmark environments
│   ├── pist_mas.py      # PIST-MAS causal decoupling
│   ├── mathchat.py      # Mathematical reasoning (GSM8K/MATH)
│   ├── hotpotqa.py      # Multi-hop QA
│   ├── coding.py        # Code generation (HumanEval)
│   ├── dsbench.py       # Data science (DSBench)
│   └── coupling_test.py # Coupling stress test
├── training/            # Credit-driven optimization
│   ├── credit_optimizer.py  # HEDGE-weighted policy gradient
│   └── lora_utils.py       # LoRA fine-tuning utilities
├── eval/                # Evaluation metrics
├── scripts/             # Experiment runners
│   ├── run_all.sh       # Full reproduction pipeline
│   ├── run_main_table.py
│   ├── run_pist_mas.py
│   ├── run_collapse_curve.py
│   ├── run_coupling_test.py
│   └── run_ablation.py
├── figures/             # Plotting scripts
├── configs/             # Experiment configurations
└── requirements.txt
```

## Quick Start

### Requirements

- Python ≥ 3.10
- CUDA-capable GPU (≥ 4× A100 80GB recommended)
- PyTorch ≥ 2.1, vLLM ≥ 0.4

### Installation

```bash
pip install -r requirements.txt
```

### Running Experiments

```bash
# Full reproduction (all tables and figures)
bash scripts/run_all.sh

# Individual experiments
python scripts/run_main_table.py      # Table 5: main results
python scripts/run_pist_mas.py        # Table 2: causal decoupling
python scripts/run_collapse_curve.py  # Figure 3: collapse curve
python scripts/run_coupling_test.py   # Table 6: coupling stress test
python scripts/run_ablation.py        # Table 3: six-level ablation
python scripts/run_m_sensitivity.py   # Appendix: m-sensitivity
python scripts/run_stochastic.py      # Appendix: stochastic robustness
```

### Configuration

Edit `configs/base.yaml` to set model paths, GPU allocation, and hyperparameters:

```yaml
model:
  backbone: meta-llama/Llama-3.1-70B-Instruct
  gpu_ids: [0, 1, 2, 3]
  
hedge:
  shrinkage: james_stein   # {james_stein, empirical_bayes}
  variant: base            # {base, bias_tolerant, correlated, stochastic, adaptive}
```

## Main Results (Table 5)

| Method | Math | HotpotQA | Code | DS | Avg |
|--------|------|----------|------|----|-----|
| Uniform | 53.9 | 41.2 | 35.8 | 29.4 | 40.1 |
| L1O | 50.5 | 39.8 | 34.2 | 27.8 | 38.1 |
| Shapley | 54.7 | 42.1 | 36.5 | 30.1 | 40.9 |
| Math-Shepherd PRM | 57.2 | 43.8 | 37.9 | 31.6 | 42.6 |
| MCTS Credit | 56.8 | 43.5 | 37.4 | 31.2 | 42.2 |
| **HEDGE (Ours)** | **61.4** | **47.3** | **40.6** | **34.1** | **45.9** |

*n=12 agents, Llama-3.1-70B backbone, m=8 resamples.*

## Citation

```bibtex
@inproceedings{anonymous2027hedge,
  title     = {When Credit Collapses: Heteroscedastic, Entropy-Driven Group Estimation of Credit in {LLM} Multi-Agent Systems},
  author    = {Anonymous},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2027},
}
```

## License

This code is released for academic research purposes. See [LICENSE](LICENSE) for details.
