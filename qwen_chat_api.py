#!/usr/bin/env python3
"""Qwen DashScope OpenAI-compatible provider adapter for resilient_llm_client."""

import os
from typing import Any

from openai_compat_client import OpenAICompatChatClient, RequestMetrics


class QwenChatApiClient(OpenAICompatChatClient):
    def __init__(self, api_key: str, base_url: str, model: str, **kwargs: Any) -> None:
        api_key = os.environ.get("DASHSCOPE_API_KEY", os.environ.get("QWEN_API_KEY", api_key))
        base_url = os.environ.get("QWEN_BASE_URL", base_url)
        model = os.environ.get("QWEN_MODEL", model)
        endpoint_paths = kwargs.pop("endpoint_paths", None)
        if endpoint_paths is None:
            endpoint_paths = ["/chat/completions"] if base_url.rstrip("/").endswith("/v1") else None
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            endpoint_paths=endpoint_paths,
            **kwargs,
        )
