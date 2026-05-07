"""GemmaPlanner — Google Gemma 4 family via local vLLM (HuggingFace auto-download).

Covers any Gemma 4 checkpoint that vLLM 0.19.1 can serve:
  - google/gemma-4-E2B-it, google/gemma-4-E4B-it           (small, multimodal)
  - google/gemma-4-26B-A4B-it                              (MoE, multimodal)
  - google/gemma-4-31B-it                                  (dense, multimodal)

vLLM 0.19.1 already registers `Gemma4ForCausalLM` /
`Gemma4ForConditionalGeneration` and ships `gemma4_tool_parser`, so we route
through the existing OpenAICompatiblePlanner with Gemma-specific sampling
defaults baked in here.

Sampling values follow Gemma 4's HF generation_config:
    temperature        = 1.0   (passed via --temperature; not baked here)
    top_p              = 0.95  (baked in this class)
    top_k              = 64    (baked in this class)
    repetition_penalty = 1.0   (HF transformers default; baked for explicitness)
Pass `--temperature 0.0` (greedy) at your own risk — Gemma 4 (like Gemma 2/3)
is sampled at training time and greedy decoding can produce repetitive
output on long generations.

Thinking-mode toggle (`--thinking-budget`) only takes effect when the user
explicitly sets it, mirroring QwenPlanner's pattern. Gemma 4 supports
configurable thinking via `enable_thinking` in `apply_chat_template`, which
we forward as `chat_template_kwargs.enable_thinking` through extra_body.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from framework.models.openai_compatible import (
    JsonPromptPlanner,
    OpenAICompatiblePlanner,
)

# OURS_TOOL_MODE — see framework/models/qwen/planner.py for the full rationale.
#   prompt_json (default) — fenced JSON in system prompt; vLLM parser bypassed.
#   native                — native OpenAI tools=[] API; vLLM's gemma4 parser used.
_TOOL_MODE = os.environ.get("OURS_TOOL_MODE", "prompt_json").strip().lower()
if _TOOL_MODE not in ("native", "prompt_json"):
    raise ValueError(
        f"OURS_TOOL_MODE must be 'native' or 'prompt_json' (got {_TOOL_MODE!r})"
    )
_BasePlanner = OpenAICompatiblePlanner if _TOOL_MODE == "native" else JsonPromptPlanner


class GemmaPlanner(_BasePlanner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # None means "user did not set --thinking-budget" — leave chat template default alone.
        self._enable_thinking: Optional[bool] = None

    async def start_chat(
        self,
        system_prompt: str,
        tool_schemas: Sequence[dict],
        temperature: float = 0.0,
        thinking_budget: Optional[int] = None,
    ) -> None:
        await super().start_chat(
            system_prompt, tool_schemas, temperature, thinking_budget,
        )
        if thinking_budget is None:
            self._enable_thinking = None
        else:
            self._enable_thinking = thinking_budget > 0  # 0 disables, >0 enables

    def _build_request_kwargs(self) -> dict[str, Any]:
        kwargs = super()._build_request_kwargs()

        # Gemma 4 thinking-mode reasoning can be long; cap completions at
        # 32768 (mirrors the May 2026 Qwen anti-loop fix). Override via
        # GEMMA_MAX_TOKENS env var. Only applied when user opted into
        # thinking via --thinking-budget>0.
        import os as _os_max
        if self._enable_thinking:
            kwargs["max_tokens"] = int(_os_max.environ.get("GEMMA_MAX_TOKENS", "32768"))

        # Gemma 4 official-recommended top_p (HF generation_config)
        kwargs["top_p"] = 0.95

        # vLLM-specific samplers go through extra_body. top_k and
        # repetition_penalty are not in OpenAI's typed Chat Completions schema.
        extra: dict[str, Any] = dict(kwargs.get("extra_body") or {})
        extra.setdefault("top_k", 64)
        extra.setdefault("repetition_penalty", 1.0)

        # enable_thinking only when user explicitly set --thinking-budget.
        # Otherwise leave the chat template's own default (avoids sending
        # unknown kwargs if the local jinja template doesn't accept it).
        if self._enable_thinking is not None:
            ctk = dict(extra.get("chat_template_kwargs") or {})
            ctk["enable_thinking"] = self._enable_thinking
            extra["chat_template_kwargs"] = ctk

        kwargs["extra_body"] = extra
        return kwargs
