#!/usr/bin/env python3
"""Hunyuan provider adapter for resilient_llm_client."""

import os
from typing import Any

from openai_compat_client import OpenAICompatChatClient, RequestMetrics


class HunyuanApiClient(OpenAICompatChatClient):
    def __init__(self, api_key: str, base_url: str, model: str, **kwargs: Any) -> None:
        api_key = os.environ.get("HUNYUAN_API_KEY", api_key)
        base_url = os.environ.get("HUNYUAN_BASE_URL", base_url)
        model = os.environ.get("HUNYUAN_MODEL", model)
        endpoint_paths = kwargs.pop("endpoint_paths", None)
        if endpoint_paths is None:
            endpoint_paths = ["/v1/chat/completions", "/chat/completions"]
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            endpoint_paths=endpoint_paths,
            **kwargs,
        )
