"""OpenAICompatiblePlanner — shared base for any OpenAI-compatible endpoint.

Covers all open-source models served by vLLM (Qwen, Gemma, Llama, Mistral,
DeepSeek, ...) as well as DashScope (hosted Qwen) and OpenAI proper. From the
client's perspective they are identical: vLLM's `--tool-call-parser <name>`
flag normalizes every model family's native tool-call wire format
(<tool_call>{json}</tool_call>, <function=...>, [TOOL_CALLS]{...}, etc.) into
the OpenAI-standard ToolCall Pydantic shape, so this class needs no
per-model branching.

Per-model subclasses (framework/models/qwen/, gemma/, kimi_vl/, ...) inherit
from this class and override only what is genuinely model-specific (e.g.,
system prompt augmentation, custom thinking parsing, retry tweaks). For the
simple case, a one-line subclass is enough.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional, Sequence

from openai import AsyncOpenAI

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
    "500", "internal server", "internal_server",
    "503", "service unavailable", "service_unavailable",
    "504", "gateway timeout", "gateway_timeout",
    "timeout", "connection",
)

# Tool-call wire-format markers. If `tool_calls=[]` AND any of these strings
# appear in the model's plaintext output (content/reasoning_content), the
# response is treated as a parser/markup-leak failure and the same API call is
# retried up to `OURS_RETRY_ON_EMPTY` times (sampling-variance retry, no
# history append between attempts). Covers Qwen (Hermes / qwen3_xml) and
# Gemma 4 (gemma4) wire formats.
_LEAK_MARKERS = (
    # Gemma 4
    "<|tool_call>", "<tool_call|>",
    # Qwen Hermes / qwen3_xml partial markup
    "<tool_call>", "</tool_call>",
    "<function=", "<parameter=",
)


class OpenAICompatiblePlanner(PlannerLLM):
    """Generic planner for any OpenAI-compatible HTTP endpoint.

    Args:
        base_url: e.g. "http://localhost:8000/v1" for self-hosted vLLM,
            "https://dashscope-us.aliyuncs.com/compatible-mode/v1" for DashScope,
            None/default for OpenAI proper.
        api_key: vLLM ignores this by default (pass any string like "EMPTY").
        model: model id as registered with the server. For vLLM this is the
            HF id (e.g., "Qwen/Qwen3.5-9B"). For DashScope it's "qwen3.5-plus"
            etc. For OpenAI it's "gpt-4o" etc.
        max_retries: number of retry attempts for transient errors.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        model: str = "",
        max_retries: int = 5,
    ):
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = model
        self._max_retries = max_retries

        # Multi-turn state (managed manually unlike Gemini's chat object)
        self._messages: list[dict] = []
        self._tools: list[dict] = []
        self._temperature: float = 0.0

        # Raw dump (debug). When set via set_raw_dump_context, every successful
        # chat.completions.create() call writes the full request kwargs +
        # vLLM ChatCompletion response (model_dump) to disk for forensic
        # inspection. Off by default — zero overhead unless enabled.
        self._raw_dump_dir = None  # type: Optional[Any]
        self._raw_qid: Optional[int] = None
        self._raw_iter_count: int = 0

    def set_raw_dump_context(self, dir_path, qid: int) -> None:
        """Enable raw response capture for this planner.

        Each subsequent send() that reaches the model successfully will dump:
            {dir_path}/question{qid}.iter{N}.json
        with `request` (kwargs sent) + `response` (full ChatCompletion JSON
        including reasoning_content, content, tool_calls, finish_reason, usage).
        Iteration counter resets when this is called."""
        self._raw_dump_dir = dir_path
        self._raw_qid = qid
        self._raw_iter_count = 0

    def _maybe_dump_raw(self, request_kwargs: dict, resp) -> None:
        if self._raw_dump_dir is None or self._raw_qid is None:
            return
        try:
            from pathlib import Path as _Path
            d = _Path(self._raw_dump_dir)
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"question{self._raw_qid}.iter{self._raw_iter_count}.json"
            resp_dump = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            payload = {
                "qid": self._raw_qid,
                "iter": self._raw_iter_count,
                "model": self._model,
                "request": request_kwargs,
                "response": resp_dump,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(
                "raw dump failed for q%s iter%s: %s",
                self._raw_qid, self._raw_iter_count, e,
            )
        finally:
            self._raw_iter_count += 1

    # ----- PlannerLLM interface ---------------------------------------------

    async def start_chat(
        self,
        system_prompt: str,
        tool_schemas: Sequence[dict],
        temperature: float = 0.0,
        thinking_budget: Optional[int] = None,
    ) -> None:
        # tool_schemas is already in OpenAI format (BaseTool.to_schema output).
        # No translation needed — pass through.
        self._messages = [{"role": "system", "content": system_prompt}]
        self._tools = list(tool_schemas)
        self._temperature = temperature
        # thinking_budget intentionally ignored at this layer. Per-model
        # thinking behavior is controlled via:
        #   1. Model variant choice (e.g., Qwen3.5-Thinking-... vs ...-Instruct)
        #   2. vLLM launch flag --reasoning-parser <name> which surfaces
        #      <think>...</think> content as msg.reasoning_content
        # Subclasses may override to inject extra_body params if needed.

    @staticmethod
    def _is_corrupted_empty(parsed: PlannerResponse) -> bool:
        """Detect 'silently failed' tool-call responses worth a same-call retry.

        True when:
          - tool_calls is empty AND
          - either the plaintext output contains tool-call wire-format markup
            (A-type leak: native parser failed to extract), OR
          - both content and reasoning_content are empty (C-type: parser
            pipeline ate everything).

        False for normal final-answer responses (text non-empty, no markup) so
        we don't retry legitimate end-of-tool-calling turns.
        """
        if parsed.tool_calls:
            return False
        blob = (parsed.text or "") + (parsed.thinking or "")
        if any(m in blob for m in _LEAK_MARKERS):
            return True   # A-type
        if not blob.strip():
            return True   # C-type
        return False

    async def send(self, message: PlannerInput) -> Optional[PlannerResponse]:
        # Append the new user/tool turn to history (once, BEFORE any attempt —
        # subsequent same-call retries reuse the identical message context).
        if isinstance(message, str):
            self._messages.append({"role": "user", "content": message})
        else:  # Sequence[ToolCallResult]
            for r in message:
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": r.call_id or "",
                    "content": r.result,
                })

        try:
            from framework._shared.token_tracker import record_retry_wait
        except Exception:
            def record_retry_wait(_s):  # type: ignore
                pass

        request_kwargs = self._build_request_kwargs()

        # Same-call retry on A-/C-type empty responses. Each attempt is an
        # independent vLLM sampling — no history append between attempts so the
        # model sees the identical conversation each time. Default 10; opt-out
        # with OURS_RETRY_ON_EMPTY=0.
        max_empty_retries = max(0, int(os.environ.get("OURS_RETRY_ON_EMPTY", "10")))
        # Per-retry temperature step-down. Lower temp on retries biases sampling
        # toward the model's most-likely (typically well-formed) path. Floor at
        # 0.1 — CLAUDE.md: greedy (T=0.0) triggers Qwen3 thinking loops, so we
        # never let attempt-N reduction push temperature below 0.1.
        retry_temp_step = float(os.environ.get("OURS_RETRY_TEMP_STEP", "0.05"))
        retry_temp_floor = float(os.environ.get("OURS_RETRY_TEMP_FLOOR", "0.1"))
        base_temperature = request_kwargs.get("temperature", self._temperature)

        last_resp = None
        last_parsed: Optional[PlannerResponse] = None

        for empty_attempt in range(max_empty_retries + 1):
            # Step-down temperature on retries (attempt 0 = base temp untouched).
            if empty_attempt > 0:
                stepped = base_temperature - retry_temp_step * empty_attempt
                request_kwargs["temperature"] = max(retry_temp_floor, stepped)
            # Inner loop: HTTP/transport retry (rate limits, 5xx, etc.)
            resp = None
            for retry in range(self._max_retries):
                try:
                    resp = await self._client.chat.completions.create(**request_kwargs)
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if any(kw in err_str for kw in _RETRYABLE_KEYWORDS):
                        wait = 2 ** retry * 5
                        logger.warning("OpenAI-compat transient (%s), waiting %ds...", e, wait)
                        await asyncio.sleep(wait)
                        record_retry_wait(wait)
                        continue
                    logger.error("OpenAI-compat call failed: %s", e)
                    return None
            if resp is None:
                logger.error("OpenAI-compat retries exhausted (transport)")
                return None

            self._maybe_dump_raw(request_kwargs, resp)
            parsed = self._parse(resp)
            last_resp, last_parsed = resp, parsed

            if not self._is_corrupted_empty(parsed):
                self._append_assistant_turn(resp)
                return parsed

            if empty_attempt < max_empty_retries:
                next_temp = max(retry_temp_floor,
                                base_temperature - retry_temp_step * (empty_attempt + 1))
                logger.warning(
                    "Empty/leaked tool_call response (same-call retry %d/%d, next temp=%.2f)...",
                    empty_attempt + 1, max_empty_retries, next_temp,
                )

        # All same-call retries exhausted — fall through with the last response
        # (still better than None: react.py can decide to advance to next iter
        # or terminate). History gets the last assistant turn for consistency.
        logger.warning(
            "Same-call retry exhausted after %d attempts — accepting last (corrupted) response",
            max_empty_retries + 1,
        )
        if last_resp is not None:
            self._append_assistant_turn(last_resp)
        return last_parsed

    def attach_token_tracker(self, accumulator: Any) -> None:
        from framework._shared.token_tracker import wrap_openai_client
        wrap_openai_client(self._client, accumulator)

    # ----- subclass override hooks -----------------------------------------

    def _build_request_kwargs(self) -> dict[str, Any]:
        """Construct kwargs for `chat.completions.create`. Subclasses can
        override to inject `extra_body`, `reasoning_effort`, model-family
        flags, etc."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "temperature": self._temperature,
        }
        if self._tools:
            kwargs["tools"] = self._tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    # ----- internals --------------------------------------------------------

    def _append_assistant_turn(self, resp) -> None:
        """Echo the assistant's reply (text + tool_calls) back into history
        so the next turn has full context."""
        msg = resp.choices[0].message
        record: dict[str, Any] = {"role": "assistant"}
        # OpenAI requires content to be present (even if null)
        record["content"] = msg.content or ""
        if msg.tool_calls:
            record["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        self._messages.append(record)

    def _parse(self, resp) -> PlannerResponse:
        msg = resp.choices[0].message

        # Reasoning surfaces under reasoning_content when vLLM is launched
        # with --reasoning-parser, otherwise stays embedded in content.
        thinking = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )
        text = msg.content or ""

        tool_calls: list[PlannerToolCall] = []
        for tc in (msg.tool_calls or []):
            args_raw = tc.function.arguments or ""
            try:
                args = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to JSON-parse tool args for %s: %r",
                    tc.function.name, args_raw[:200],
                )
                args = {}
            tool_calls.append(PlannerToolCall(
                name=tc.function.name,
                arguments=args,
                call_id=tc.id,
            ))

        return PlannerResponse(
            thinking=thinking,
            text=text,
            tool_calls=tool_calls,
        )
