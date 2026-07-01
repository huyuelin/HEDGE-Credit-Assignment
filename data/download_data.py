"""
Data loaders for GSM8K, MATH-500, and HotpotQA.
Downloads datasets from HuggingFace and saves as JSONL.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """Save data as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def download_gsm8k(output_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Download GSM8K test set."""
    from datasets import load_dataset

    LOGGER.info("Downloading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    data = []
    for item in dataset:
        question = item["question"]
        answer_text = item["answer"]
        numbers = re.findall(r'####\s*(.+)', answer_text)
        answer = numbers[0].strip().replace(",", "") if numbers else ""
        data.append({"problem": question, "answer": answer, "solution": answer_text})

    if max_samples:
        data = data[:max_samples]

    save_jsonl(data, output_path)
    LOGGER.info(f"Saved {len(data)} GSM8K problems to {output_path}")
    return data


def download_math500(output_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Download MATH-500 test set."""
    from datasets import load_dataset

    LOGGER.info("Downloading MATH-500...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    data = []
    for item in dataset:
        problem = item.get("problem", "")
        solution = item.get("solution", "")
        boxed = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', solution)
        answer = boxed[-1] if boxed else ""
        data.append({"problem": problem, "answer": answer, "solution": solution})

    if max_samples:
        data = data[:max_samples]

    save_jsonl(data, output_path)
    LOGGER.info(f"Saved {len(data)} MATH-500 problems to {output_path}")
    return data


def download_hotpotqa(output_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Download HotpotQA validation set."""
    from datasets import load_dataset

    LOGGER.info("Downloading HotpotQA...")
    dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
    data = []
    for item in dataset:
        question = item.get("question", "")
        answer = item.get("answer", "")
        data.append({"problem": question, "answer": answer})
        if max_samples and len(data) >= max_samples:
            break

    save_jsonl(data, output_path)
    LOGGER.info(f"Saved {len(data)} HotpotQA questions to {output_path}")
    return data


def download_all(data_dir: str, max_samples: Optional[int] = 500):
    """Download all datasets."""
    os.makedirs(data_dir, exist_ok=True)
    download_gsm8k(os.path.join(data_dir, "gsm8k.jsonl"), max_samples)
    download_math500(os.path.join(data_dir, "math500.jsonl"), max_samples)
    download_hotpotqa(os.path.join(data_dir, "hotpotqa.jsonl"), max_samples)


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/jackey_workspace/hedge_data"
    download_all(data_dir)
