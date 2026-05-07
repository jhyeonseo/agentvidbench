"""KimiVLPlanner — Moonshot AI Kimi-VL family via local vLLM.

Covers Kimi-VL checkpoints that vLLM 0.19.1 serves natively
(`KimiVLForConditionalGeneration`):
  - moonshotai/Kimi-VL-A3B-Thinking-2506   (recommended; latest)
  - moonshotai/Kimi-VL-A3B-Thinking
  - moonshotai/Kimi-VL-A3B-Instruct        (non-thinking variant)

vLLM 0.19.1 ships matching parsers we leverage at the launch-script layer:
  - vllm/tool_parsers/kimi_k2_tool_parser.py
  - vllm/reasoning/kimi_k2_reasoning_parser.py

Naming is `kimi_k2_*` because Moonshot's K2 (large text) and VL share token
conventions; both parsers are documented to apply across the family.

Sampling defaults baked here mirror our QwenPlanner / GemmaPlanner pattern.
For Thinking variants:
    top_p = 0.95, top_k = 20, repetition_penalty = 1.0
For Instruct (non-thinking):
    top_p = 0.95, top_k = 20, repetition_penalty = 1.05
Temperature is plumbed from --temperature (run_kimi_vl.sh default 0.6 per HF
generation_config of Kimi-VL-A3B-Thinking-2506; model card recommends 0.8 for
"long thinking" — we default to the locked-in generation_config value).

Thinking-mode toggle (`--thinking-budget`) only takes effect when the user
explicitly sets it (>0 enables, 0 disables). Forwarded as
`chat_template_kwargs.thinking` (consumed by kimi_k2_reasoning_parser).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from framework.models.openai_compatible import OpenAICompatiblePlanner


class KimiVLPlanner(OpenAICompatiblePlanner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # None means "user did not set --thinking-budget" — leave chat template default alone.
        self._enable_thinking: Optional[bool] = None

    @property
    def _is_thinking_variant(self) -> bool:
        """Heuristic: model id contains 'thinking' (case-insensitive).
        Matches Kimi-VL-A3B-Thinking-2506, kimi-vl-a3b-thinking, etc."""
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
            self._enable_thinking = thinking_budget > 0  # 0 disables, >0 enables

    def _build_request_kwargs(self) -> dict[str, Any]:
        kwargs = super()._build_request_kwargs()

        # top_p uniform across Thinking/Instruct (model card / HF generation_config).
        kwargs["top_p"] = 0.95

        # vLLM-specific samplers via extra_body (top_k / repetition_penalty are
        # not in OpenAI's typed Chat Completions schema).
        extra: dict[str, Any] = dict(kwargs.get("extra_body") or {})
        extra.setdefault("top_k", 20)
        extra.setdefault(
            "repetition_penalty",
            1.0 if self._is_thinking_variant else 1.05,
        )

        # enable_thinking only when user explicitly set --thinking-budget.
        # Otherwise leave the chat template's own default (avoids unknown
        # kwargs leaking to non-thinking templates).
        if self._enable_thinking is not None:
            ctk = dict(extra.get("chat_template_kwargs") or {})
            ctk["thinking"] = self._enable_thinking
            extra["chat_template_kwargs"] = ctk

        kwargs["extra_body"] = extra
        return kwargs
