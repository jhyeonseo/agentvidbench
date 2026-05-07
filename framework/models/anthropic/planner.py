"""AnthropicPlanner — Claude (Anthropic Messages API) planner for the ours/ReAct method.

Implements the PlannerLLM interface against the Anthropic Messages API. Uses
Claude's native tool-calling surface (translates OpenAI-format tool schemas
to Anthropic's `{name, description, input_schema}` format and back).

Routing for `ours`:
  build_planner() in framework/methods/ours/runner.py picks this class for any
  model whose name starts with `claude-`.

Tool calling translation:
  - OpenAI schema {function: {name, description, parameters}}
    → Anthropic tool {name, description, input_schema}
  - Anthropic response.content[type=tool_use] block → PlannerToolCall
  - Sequence[ToolCallResult] → user message with content blocks
    of type "tool_result" (Anthropic's tool-result format)

Temperature handling:
  Claude Opus 4.x and later deprecate explicit `temperature`. Try with
  temperature first, fall back without on the deprecation 400 error.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional, Sequence

from anthropic import AsyncAnthropic

from framework.models.base import (
    PlannerInput,
    PlannerLLM,
    PlannerResponse,
    PlannerToolCall,
    ToolCallResult,
)

logger = logging.getLogger(__name__)

_RETRYABLE_KEYWORDS = (
    "429", "rate limit", "rate_limit",
    "500", "internal", "503", "service",
    "504", "gateway", "timeout", "connection", "overloaded",
)


class AnthropicPlanner(PlannerLLM):

    _MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", 4096))

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        api_key: Optional[str] = None,
        max_retries: int = 5,
    ):
        self._client = AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
        self._model = model
        self._max_retries = max_retries

        # Multi-turn state — Anthropic puts system prompt in a separate field.
        self._system_prompt: str = ""
        self._messages: list[dict] = []
        self._tools: list[dict] = []
        self._temperature: float = 0.0

    # ----- PlannerLLM interface ---------------------------------------------

    async def start_chat(
        self,
        system_prompt: str,
        tool_schemas: Sequence[dict],
        temperature: float = 0.0,
        thinking_budget: Optional[int] = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._messages = []
        # Translate OpenAI-format tool schemas → Anthropic format.
        self._tools = []
        for s in tool_schemas:
            fn = s.get("function", s)
            self._tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {}),
            })
        self._temperature = temperature
        # thinking_budget intentionally ignored at this layer.

    async def send(self, message: PlannerInput) -> Optional[PlannerResponse]:
        if isinstance(message, str):
            self._messages.append({"role": "user", "content": message})
        else:  # Sequence[ToolCallResult]
            content_blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": r.call_id or "",
                    "content": r.result,
                }
                for r in message
            ]
            self._messages.append({"role": "user", "content": content_blocks})

        try:
            from framework._shared.token_tracker import record_retry_wait
        except Exception:
            def record_retry_wait(_s):  # type: ignore
                pass

        kwargs = {
            "model": self._model,
            "max_tokens": self._MAX_TOKENS,
            "messages": self._messages,
        }
        if self._system_prompt:
            kwargs["system"] = self._system_prompt
        if self._tools:
            kwargs["tools"] = self._tools
        # Try with temperature; fall back without on Claude Opus 4.x deprecation.
        kwargs_with_temp = dict(kwargs, temperature=self._temperature)

        for retry in range(self._max_retries):
            try:
                try:
                    resp = await self._client.messages.create(**kwargs_with_temp)
                except Exception as e:
                    if ("temperature" in str(e).lower()
                            and "deprecat" in str(e).lower()):
                        resp = await self._client.messages.create(**kwargs)
                    else:
                        raise
                self._append_assistant_turn(resp)
                return self._parse(resp)
            except Exception as e:
                err_str = str(e).lower()
                if any(kw in err_str for kw in _RETRYABLE_KEYWORDS):
                    wait = 2 ** retry * 5
                    logger.warning("Anthropic transient (%s), waiting %ds...", e, wait)
                    await asyncio.sleep(wait)
                    record_retry_wait(wait)
                    continue
                logger.error("Anthropic call failed: %s", e)
                return None
        logger.error("Anthropic retries exhausted")
        return None

    def attach_token_tracker(self, accumulator: Any) -> None:
        # No shared wrap_anthropic_client helper yet; per-call usage is still
        # available via resp.usage but not aggregated here. Leaving as no-op
        # keeps the API contract intact without breaking other plumbing.
        pass

    # ----- internals --------------------------------------------------------

    def _append_assistant_turn(self, resp) -> None:
        """Echo Claude's reply (text + tool_use content blocks) into messages
        so the next turn has full conversational context."""
        content_blocks: list[dict] = []
        for block in resp.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content_blocks.append({
                    "type": "text",
                    "text": getattr(block, "text", ""),
                })
            elif block_type == "tool_use":
                content_blocks.append({
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}) or {},
                })
            # Skip 'thinking' blocks from echo — they're internal and the model
            # doesn't need its own thinking trace replayed.
        self._messages.append({"role": "assistant", "content": content_blocks})

    def _parse(self, resp) -> PlannerResponse:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[PlannerToolCall] = []
        for block in resp.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                tool_calls.append(PlannerToolCall(
                    name=getattr(block, "name", ""),
                    arguments=dict(getattr(block, "input", {}) or {}),
                    call_id=getattr(block, "id", None),
                ))
            elif block_type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
        return PlannerResponse(
            thinking="\n".join(thinking_parts),
            text="".join(text_parts),
            tool_calls=tool_calls,
        )
