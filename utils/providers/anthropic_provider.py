# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anthropic provider implementation."""

import os

from .base import BaseProvider, LLMResponse
from .env_config import configure_proxy_environment

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None


class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""

    def __init__(self):
        self._original_proxy_env = None
        super().__init__()

    def _initialize_client(self) -> None:
        api_key = self._get_api_key("ANTHROPIC_API_KEY")
        if ANTHROPIC_AVAILABLE and api_key:
            # Configure proxy using centralized utility function
            self._original_proxy_env = configure_proxy_environment()

            # Initialize client (proxy configured via environment variables)
            self.client = Anthropic(api_key=api_key)

    def get_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Anthropic client not available")

        user_content = messages[-1]["content"] if messages else ""
        response = self.client.messages.create(
            model=model_name,
            max_tokens=min(
                kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name)
            ),
            temperature=kwargs.get("temperature", 0.7),
            messages=[{"role": "user", "content": user_content}],
        )

        return LLMResponse(
            content=response.content[0].text, model=model_name, provider=self.name
        )

    def get_multiple_responses(
        self, model_name: str, messages: list[dict[str, str]], n: int = 1, **kwargs
    ) -> list[LLMResponse]:
        return [
            self.get_response(
                model_name,
                messages,
                temperature=kwargs.get("temperature", 0.7) + i * 0.1,
            )
            for i in range(n)
        ]

    def is_available(self) -> bool:
        return ANTHROPIC_AVAILABLE and self.client is not None

    def get_max_tokens_limit(self, model_name: str) -> int:
        # OPENAI_MAX_TOKENS is KA's cross-provider output cap (set by
        # scripts/ka_run.sh and the Deployment). Honor it here so Claude gets the
        # same output budget as the OpenAI/GLM path — the base default (8192)
        # truncates long kernel rewrites and makes a cross-model A/B unfair.
        _env = os.environ.get("OPENAI_MAX_TOKENS")
        return int(_env) if _env else 8192

    @property
    def name(self) -> str:
        return "anthropic"
