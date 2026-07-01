"""
Transformers-based inference backend (fallback when vLLM is unavailable).

Uses HuggingFace transformers for text generation with logprobs.
Slower than vLLM but doesn't require a separate server.
"""

import logging
import math
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


class TransformersBackend:
    """Direct transformers inference backend with logprob support."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        torch_dtype=torch.bfloat16,
        max_batch_size: int = 4,
    ):
        LOGGER.info(f"Loading model: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()
        self.device = device
        self.max_batch_size = max_batch_size
        self.model_name = model_name_or_path

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        logprobs: bool = True,
        top_logprobs: int = 5,
        n: int = 1,
        stop: Optional[List[str]] = None,
    ) -> List:
        """Generate completions with logprobs."""
        from agents.vllm_backend import GenerationResult

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        results = []
        for _ in range(n):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    top_p=top_p,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            generated_ids = outputs.sequences[0][prompt_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            token_logprobs = []
            if logprobs and outputs.scores:
                for i, score in enumerate(outputs.scores):
                    probs = torch.softmax(score[0], dim=-1)
                    token_id = generated_ids[i].item()
                    if token_id < len(probs):
                        lp = math.log(probs[token_id].item() + 1e-10)
                    else:
                        lp = -5.0
                    token_logprobs.append(lp)

            tokens = [self.tokenizer.decode([tid]) for tid in generated_ids.tolist()]

            results.append(GenerationResult(
                text=text,
                token_logprobs=token_logprobs,
                tokens=tokens,
                finish_reason="stop",
                prompt_tokens=prompt_len,
                completion_tokens=len(generated_ids),
            ))

        return results

    def health_check(self) -> bool:
        return True

    def close(self):
        pass
