"""OpenAIPlanner — OpenAI proper (GPT-5, GPT-4o, o1, o3, o4) via Chat Completions.

Subclass of OpenAICompatiblePlanner with OpenAI-specific defaults:
  - base_url default = None (openai SDK uses https://api.openai.com/v1)
  - api_key from OPENAI_API_KEY env

Routing for `ours`:
  build_planner() in framework/methods/ours/runner.py picks this class for any
  model whose name starts with `gpt`, `o1`, `o3`, or `o4`.

Reasoning-only variants (o1/o3/o4) reject the `temperature` parameter at
the API level — we suppress it via _build_request_kwargs override so the
generic OpenAICompatiblePlanner doesn't send it.

Tool-use is the standard OpenAI Chat Completions tools/tool_choice surface,
which the parent class already handles unmodified.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from framework.models.openai_compatible import OpenAICompatiblePlanner


class OpenAIPlanner(OpenAICompatiblePlanner):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
    ):
        # base_url=None → openai SDK default endpoint (https://api.openai.com/v1).
        # Allow override for Azure OpenAI / proxy: pass an explicit URL.
        super().__init__(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            model=model,
            max_retries=max_retries,
        )

    @property
    def _is_reasoning_only(self) -> bool:
        """True for reasoning-only OpenAI models (o1/o3/o4 + GPT-5 family) —
        these reject `temperature` at the API level (verified 2026-04-30:
        gpt-5* returns 400 'Unsupported parameter' on temperature)."""
        m = (self._model or "").lower()
        return m.startswith(("o1", "o3", "o4")) or m.startswith("gpt-5")

    def _build_request_kwargs(self) -> dict[str, Any]:
        kwargs = super()._build_request_kwargs()
        if self._is_reasoning_only:
            kwargs.pop("temperature", None)
        return kwargs
