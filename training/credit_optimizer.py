"""
Credit-weighted LoRA optimizer.

Uses per-step credit scores to weight the training loss.
High-credit steps receive more gradient; low-credit steps receive less.

This implements the credit-driven optimization described in Section 5.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from training.lora_utils import create_lora_model, load_tokenizer

LOGGER = logging.getLogger(__name__)


class CreditWeightedDataset(Dataset):
    """Dataset of (prompt, response, credit_weight) tuples."""

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_length: int = 1024,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex["prompt"]
        response = ex["response"]
        credit = ex["credit_weight"]

        full_text = f"{prompt}\n{response}"
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = encoding["input_ids"].clone()
        labels[0, :prompt_len] = -100

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "credit_weight": torch.tensor(credit, dtype=torch.float32),
        }


class CreditOptimizer:
    """Credit-weighted LoRA training pipeline."""

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str,
        lora_rank: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.05,
    ):
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio

        os.makedirs(output_dir, exist_ok=True)

    def train(
        self,
        train_examples: List[Dict],
        eval_examples: Optional[List[Dict]] = None,
    ) -> Dict:
        """Run credit-weighted LoRA training.

        Args:
            train_examples: List of dicts with keys:
                - prompt: str
                - response: str
                - credit_weight: float (normalized credit score)
            eval_examples: Optional eval set

        Returns:
            Training metrics dict
        """
        tokenizer = load_tokenizer(self.model_name_or_path)
        model = create_lora_model(self.model_name_or_path, lora_rank=self.lora_rank)

        train_dataset = CreditWeightedDataset(train_examples, tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        model.train()
        global_step = 0
        total_loss = 0.0
        metrics = {"train_losses": [], "eval_losses": []}

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)
                credit_weights = batch["credit_weight"].to(model.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                per_sample_loss = outputs.loss
                weighted_loss = (per_sample_loss * credit_weights.mean())
                loss = weighted_loss / self.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * self.gradient_accumulation_steps

            avg_epoch_loss = epoch_loss / len(train_loader)
            metrics["train_losses"].append(avg_epoch_loss)
            LOGGER.info(f"Epoch {epoch+1}/{self.num_epochs}: loss={avg_epoch_loss:.4f}")

        save_path = os.path.join(self.output_dir, "lora_adapter")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        metrics_path = os.path.join(self.output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        LOGGER.info(f"Model saved to {save_path}")
        return metrics


def prepare_training_data(
    transcripts_and_credits: List[Tuple],
    task_prompts: List[str],
    credit_method: str = "hedge",
) -> List[Dict]:
    """Convert transcripts + credits into training examples.

    Args:
        transcripts_and_credits: List of (transcript, credits_array) tuples
        task_prompts: Corresponding task prompts
        credit_method: Which credit scores to use

    Returns:
        List of training examples with credit weights
    """
    examples = []
    for (transcript, credits), task_prompt in zip(transcripts_and_credits, task_prompts):
        credit_min = credits.min()
        credit_max = credits.max()
        if credit_max - credit_min > 1e-8:
            normalized = (credits - credit_min) / (credit_max - credit_min)
        else:
            normalized = np.ones_like(credits) / len(credits)

        for step, weight in zip(transcript.steps, normalized):
            examples.append({
                "prompt": task_prompt + "\n" + step.state if step.state else task_prompt,
                "response": step.action,
                "credit_weight": float(max(weight, 0.01)),
            })

    return examples
