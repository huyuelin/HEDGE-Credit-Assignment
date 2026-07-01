"""
vLLM backend for local model inference.

Connects to a vLLM OpenAI-compatible server for text generation
with logprob support (needed for entropy computation).
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    token_logprobs: List[float]
    tokens: List[str]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class VLLMBackend:
    """Client for vLLM OpenAI-compatible API server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.base_url = f"http://{host}:{port}/v1"
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

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
    ) -> List[GenerationResult]:
        """Generate completions with logprobs via vLLM."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "n": n,
        }
        if stop:
            payload["stop"] = stop

        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                results = []
                for choice in data["choices"]:
                    token_logprobs = []
                    tokens = []
                    if logprobs and choice.get("logprobs") and choice["logprobs"].get("content"):
                        for token_info in choice["logprobs"]["content"]:
                            token_logprobs.append(token_info["logprob"])
                            tokens.append(token_info["token"])

                    results.append(GenerationResult(
                        text=choice["message"]["content"],
                        token_logprobs=token_logprobs,
                        tokens=tokens,
                        finish_reason=choice.get("finish_reason", "stop"),
                        prompt_tokens=data["usage"]["prompt_tokens"],
                        completion_tokens=data["usage"]["completion_tokens"],
                    ))
                return results

            except (httpx.HTTPError, KeyError) as e:
                LOGGER.warning(f"vLLM request failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[GenerationResult]:
        """Generate for multiple prompts (sequential for now)."""
        results = []
        for messages in messages_batch:
            result = self.generate(messages, **kwargs)
            results.extend(result)
        return results

    def health_check(self) -> bool:
        """Check if vLLM server is responsive."""
        try:
            resp = self._client.get(f"{self.base_url}/models")
            return resp.status_code == 200
        except Exception:
            return False

    def close(self):
        self._client.close()
