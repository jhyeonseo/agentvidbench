"""QwenPlanner — Qwen 3.x family via local vLLM (HuggingFace auto-download).

Covers any Qwen 3.x checkpoint that vLLM can serve:
  - Qwen/Qwen3.5-9B, Qwen/Qwen3.5-2B, Qwen/Qwen3.5-32B, ...   (Instruct / general)
  - Qwen/Qwen3.5-Thinking-7B, Qwen/Qwen3.5-Thinking-32B, ...   (with <think> output)
  - Qwen/Qwen3-Coder-*, Qwen/Qwen3-Coder-Next                  (Coder variants)

The MODEL string is auto-discovered from the running vLLM server in
run_qwen.sh, so any Qwen variant routes here purely by lowercase prefix
("qwen") in build_planner. No per-variant hardcoding.

vLLM launch (separate terminal, weights auto-download from HF on first run):
    bash scripts/launch_vllm_qwen.sh Qwen/Qwen3.5-9B
Then:
    bash run_qwen.sh

Qwen3 official sampling defaults (https://qwen.readthedocs.io/en/latest/
inference/generation.html and HF model cards):
    Non-thinking : temperature=0.7, top_p=0.8,  top_k=20, rep_penalty=1.05
    Thinking     : temperature=0.6, top_p=0.95, top_k=20, rep_penalty=1.0
Greedy decoding (temperature=0) is explicitly NOT recommended for Qwen3 —
causes degradation and may trigger infinite loops in thinking mode.

Temperature is plumbed from --temperature (set in run_qwen.sh).
top_p / top_k / repetition_penalty are baked here (chosen automatically from
the variant: thinking vs non-thinking, detected by 'thinking' substring in
the model id).

Thinking-mode toggle (`--thinking-budget`) only takes effect on Thinking
variants — chat_template_kwargs.enable_thinking is injected ONLY when the
variant supports it AND the user explicitly set --thinking-budget. Avoids
sending an unknown kwarg to Instruct chat templates.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from framework.models.openai_compatible import (
    JsonPromptPlanner,
    OpenAICompatiblePlanner,
)

# OURS_TOOL_MODE switches the tool-calling layer at import time:
#   prompt_json (default) — JsonPromptPlanner. Tool schemas embedded in the
#                           system prompt as text; assistant emits fenced JSON;
#                           we parse it ourselves. vLLM's tool parser is never
#                           invoked. Sidesteps vLLM #39056 (Qwen <tool_call>
#                           leaking inside <think>) and #39468 (Gemma missing
#                           closing tag).
#   native                — OpenAICompatiblePlanner. Native OpenAI tools=[] API;
#                           vLLM's --tool-call-parser extracts tool_calls. This
#                           is the pre-May-2026 flow and reproduces the parser
#                           bugs above (used as baseline / for ablation).
_TOOL_MODE = os.environ.get("OURS_TOOL_MODE", "prompt_json").strip().lower()
if _TOOL_MODE not in ("native", "prompt_json"):
    raise ValueError(
        f"OURS_TOOL_MODE must be 'native' or 'prompt_json' (got {_TOOL_MODE!r})"
    )
_BasePlanner = OpenAICompatiblePlanner if _TOOL_MODE == "native" else JsonPromptPlanner


class QwenPlanner(_BasePlanner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # None means "user did not set --thinking-budget". start_chat fills.
        self._enable_thinking: Optional[bool] = None

    @property
    def _is_thinking_variant(self) -> bool:
        """Heuristic: the served model id contains 'thinking' (case-insensitive).
        Matches Qwen/Qwen3.5-Thinking-7B, qwen3.5-thinking-32b, etc."""
        return "thinking" in (self._model or "").lower()

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
            self._enable_thinking = thinking_budget > 0   # 0 disables, >0 enables

    def _build_request_kwargs(self) -> dict[str, Any]:
        kwargs = super()._build_request_kwargs()

        # Qwen Thinking-mode reasoning can be very long; cap completions at
        # 16384 (matches zeroshot/avp Qwen runs) so the final answer JSON
        # always fits after the <think> block. Override via QWEN_MAX_TOKENS env.
        import os as _os_max
        kwargs["max_tokens"] = int(_os_max.environ.get("QWEN_MAX_TOKENS", "16384"))

        # Qwen3 official recommended top_p (varies by variant)
        kwargs["top_p"] = 0.95 if self._is_thinking_variant else 0.8

        # vLLM-specific samplers go through extra_body. top_k and
        # repetition_penalty are not in OpenAI's typed Chat Completions schema.
        extra: dict[str, Any] = dict(kwargs.get("extra_body") or {})
        extra.setdefault("top_k", 20)
        # Qwen3 thinking-mode is prone to in-`<think>`-block repetition loops
        # at the official rep_penalty=1.0 (observed: 100+ repeated phrases that
        # max-out max_tokens before producing a tool_call or final answer).
        # Allow env override; default still follows Qwen recommendation.
        _rep_pen = _os_max.environ.get("QWEN_REPETITION_PENALTY")
        if _rep_pen is not None:
            extra["repetition_penalty"] = float(_rep_pen)
        else:
            extra.setdefault("repetition_penalty", 1.0 if self._is_thinking_variant else 1.05)

        # enable_thinking only on Thinking variants AND only when user
        # explicitly set --thinking-budget. Otherwise leave the chat
        # template's own default (avoids sending unknown kwargs to
        # Instruct templates).
        if self._is_thinking_variant and self._enable_thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": self._enable_thinking}

        # Env-var override for non-thinking-named variants whose chat template
        # still supports enable_thinking (e.g. Qwen3.5-9B, default thinking ON).
        # Set QWEN_FORCE_ENABLE_THINKING=true|false to force the toggle.
        import os as _os
        _force = _os.environ.get("QWEN_FORCE_ENABLE_THINKING")
        if _force is not None:
            extra["chat_template_kwargs"] = {
                "enable_thinking": _force.strip().lower() in ("1", "true", "yes", "on"),
            }

        kwargs["extra_body"] = extra
        return kwargs
