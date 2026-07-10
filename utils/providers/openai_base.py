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

"""Base provider for OpenAI-compatible APIs."""

from typing import Any
import logging
import os
from .base import BaseProvider, LLMResponse
from .env_config import configure_proxy_environment

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


def _message_text(message: Any) -> str | None:
    """Extract the assistant text, tolerating reasoning models.

    vLLM reasoning models (GLM/Qwen) may return ``content=None`` with the text in
    ``reasoning`` / ``reasoning_content``. Fall back so downstream code-fence
    parsing always gets a string instead of crashing on None.
    """
    if getattr(message, "content", None):
        return message.content
    extra = getattr(message, "model_extra", None) or {}
    return (
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or extra.get("reasoning_content")
        or extra.get("reasoning")
    )


class OpenAICompatibleProvider(BaseProvider):
    """Base provider for OpenAI-compatible APIs."""

    def __init__(self, api_key_env: str, base_url: str | None = None):
        self.api_key_env = api_key_env
        self.base_url = base_url
        self._original_proxy_env = None
        super().__init__()

    def _initialize_client(self) -> None:
        """Initialize OpenAI-compatible client."""
        if not OPENAI_AVAILABLE:
            return

        api_key = self._get_api_key(self.api_key_env)
        if api_key:
            # Configure proxy using centralized utility function
            self._original_proxy_env = configure_proxy_environment()

            # Initialize client (proxy configured via environment variables)
            if self.base_url:
                self.client = OpenAI(api_key=api_key, base_url=self.base_url)
            else:
                self.client = OpenAI(api_key=api_key)

    def get_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Get single response."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        api_params = self._build_api_params(model_name, messages, **kwargs)
        response = self.client.chat.completions.create(**api_params)
        logging.getLogger(__name__).info(
            "OpenAI chat response (single): %s",
            getattr(response, "model_dump", lambda: str(response))(),
        )

        return LLMResponse(
            content=_message_text(response.choices[0].message),
            model=model_name,
            provider=self.name,
            usage=response.usage.dict()
            if hasattr(response, "usage") and response.usage
            else None,
        )

    def get_multiple_responses(
        self, model_name: str, messages: list[dict[str, str]], n: int = 1, **kwargs
    ) -> list[LLMResponse]:
        """Get multiple responses using n parameter."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        api_params = self._build_api_params(model_name, messages, n=n, **kwargs)
        response = self.client.chat.completions.create(**api_params)
        logging.getLogger(__name__).info(
            "OpenAI chat response (multi): %s",
            getattr(response, "model_dump", lambda: str(response))(),
        )

        return [
            LLMResponse(
                content=_message_text(choice.message),
                model=model_name,
                provider=self.name,
                usage=response.usage.dict()
                if hasattr(response, "usage") and response.usage
                else None,
            )
            for choice in response.choices
        ]

    def _build_api_params(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> dict[str, Any]:
        """Build API parameters for OpenAI-compatible call."""
        params = {
            "model": model_name,
            "messages": messages,
        }

        # GPT-5 and o-series models pin their own sampling behaviour
        if not (model_name.startswith("gpt-5") or model_name.startswith("o")):
            params["temperature"] = kwargs.get("temperature", 0.7)

        # Use max_completion_tokens for newer models like GPT-5, fallback to max_tokens.
        # OPENAI_MAX_TOKENS overrides the cap — reasoning models (thinking ON) need a
        # bigger budget so `content` still populates after the chain-of-thought.
        _env_max = os.environ.get("OPENAI_MAX_TOKENS")
        if _env_max:
            max_tokens_value = int(_env_max)
        else:
            max_tokens_value = min(
                kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name)
            )
        if model_name.startswith("gpt-5") or model_name.startswith("o"):
            params["max_completion_tokens"] = max_tokens_value
        else:
            params["max_tokens"] = max_tokens_value

        # Add n parameter if specified
        if "n" in kwargs:
            params["n"] = kwargs["n"]

        # Auto-enable high reasoning for GPT-5
        if model_name.startswith("gpt-5"):
            params["reasoning_effort"] = "high"
        elif kwargs.get("high_reasoning_effort") and model_name.startswith(
            ("o3", "o1")
        ):
            params["reasoning_effort"] = "high"

        # Self-hosted reasoning models (GLM/Qwen via vLLM) emit chain-of-thought in
        # `reasoning` and leave `content` null until it finishes — slow, and it can
        # exhaust the token budget so `content` never populates. OPENAI_DISABLE_THINKING
        # tells vLLM to skip thinking. No-op for OpenAI/Anthropic.
        if os.environ.get("OPENAI_DISABLE_THINKING"):
            params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        return params

    def is_available(self) -> bool:
        """Check if provider is available."""
        return OPENAI_AVAILABLE and self.client is not None

    def supports_multiple_completions(self) -> bool:
        """OpenAI-compatible APIs support native multiple completions."""
        return True
