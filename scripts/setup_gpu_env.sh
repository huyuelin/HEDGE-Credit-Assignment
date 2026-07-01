#!/bin/bash
# Setup GPU environment for HEDGE experiments
# Run on GPU server: ssh -p 50022 jackey@222.92.117.35

set -e

echo "=== Setting up HEDGE experiment environment ==="

# Use existing conda/venv
export PATH="/data/jackey_workspace/miniconda3/bin:$PATH"
source /data/jackey_workspace/miniconda3/etc/profile.d/conda.sh

# Create new env
conda create -n hedge python=3.10 -y 2>/dev/null || true
conda activate hedge

# Install PyTorch (CUDA 12.1)
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
pip install vllm>=0.4.0

# Install other deps
pip install transformers>=4.40.0 accelerate>=0.30.0 peft>=0.10.0 \
    datasets>=2.19.0 numpy scipy matplotlib seaborn \
    httpx openai jsonlines tqdm

# Create workspace
mkdir -p /data/jackey_workspace/hedge_data
mkdir -p /data/jackey_workspace/hedge_results
mkdir -p /data/jackey_workspace/hedge_code

echo "=== Downloading model (Qwen2.5-7B-Instruct for initial testing) ==="
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_name = 'Qwen/Qwen2.5-7B-Instruct'
print(f'Downloading {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir='/data/jackey_workspace/hf_cache')
print('Tokenizer downloaded.')
# Model will be loaded by vLLM, just ensure it's cached
from huggingface_hub import snapshot_download
snapshot_download(model_name, cache_dir='/data/jackey_workspace/hf_cache')
print('Model cached.')
"

echo "=== Downloading datasets ==="
python -c "
import sys
sys.path.insert(0, '/data/jackey_workspace/hedge_code')
from data.download_data import download_all
download_all('/data/jackey_workspace/hedge_data', max_samples=500)
"

echo "=== Setup complete ==="
echo "To start vLLM server:"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \\"
echo "    --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 4 --port 8000 \\"
echo "    --download-dir /data/jackey_workspace/hf_cache"
