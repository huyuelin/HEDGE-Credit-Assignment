#!/bin/bash
# Master script to run all HEDGE experiments on GPU server.
# Usage: bash scripts/run_all.sh
#
# Prerequisites:
# 1. Setup: bash scripts/setup_gpu_env.sh
# 2. Start vLLM: see below
# 3. Run experiments: bash scripts/run_all.sh

set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3

CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_DIR"

echo "=========================================="
echo " HEDGE Full Reproduction Pipeline"
echo "=========================================="
echo "Code dir: $CODE_DIR"
echo "Time: $(date)"

# Check if vLLM is running
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo ""
    echo "ERROR: vLLM server not running!"
    echo "Start it with:"
    echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \\"
    echo "    --model Qwen/Qwen2.5-7B-Instruct \\"
    echo "    --tensor-parallel-size 4 \\"
    echo "    --port 8000 \\"
    echo "    --download-dir /data/jackey_workspace/hf_cache \\"
    echo "    --gpu-memory-utilization 0.9 \\"
    echo "    --max-model-len 4096"
    echo ""
    exit 1
fi

echo "✓ vLLM server running"

# Download data if needed
if [ ! -f "/data/jackey_workspace/hedge_data/gsm8k.jsonl" ]; then
    echo ""
    echo "--- Step 0: Downloading data ---"
    python data/download_data.py /data/jackey_workspace/hedge_data
fi

echo ""
echo "--- Step 1: PIST-MAS (Table 2) ---"
python scripts/run_pist_mas.py

echo ""
echo "--- Step 2: Collapse Curve (Tables 1, 5) ---"
python scripts/run_collapse_curve.py

echo ""
echo "--- Step 3: Main Results (Table 3) ---"
python scripts/run_main_table.py

echo ""
echo "--- Step 4: Six-Level Ablation (Table 4) ---"
python scripts/run_ablation.py

echo ""
echo "--- Step 5: Coupling Stress Test (Table 6) ---"
python scripts/run_coupling_test.py

echo ""
echo "--- Step 6: Multi-Round MAS (Table 7) ---"
python scripts/run_multiround.py

echo ""
echo "--- Step 7: Stochastic Robustness (Table 8) ---"
python scripts/run_stochastic.py

echo ""
echo "--- Step 8: m-Sensitivity (Table 9) ---"
python scripts/run_m_sensitivity.py

echo ""
echo "--- Step 9: Generate Figures ---"
python figures/plot_collapse_curve.py
python figures/plot_coupling.py
python figures/plot_pareto.py
python figures/plot_ablation.py

echo ""
echo "=========================================="
echo " ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo "Results: /data/jackey_workspace/hedge_results/"
echo "Figures: /data/jackey_workspace/hedge_results/*.pdf"
echo "Time: $(date)"
