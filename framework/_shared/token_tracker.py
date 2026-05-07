"""Shared token tracker utility for measuring Gemini API usage per question
across all evaluation frameworks.

Usage:
    tracker = TokenAccumulator()
    # ... pass tracker to framework runners
    tracker.add_usage(response.usage_metadata)
    tracker.increment_call()
    # after per-question processing:
    snapshot = tracker.snapshot()
    tracker.reset()

Fields captured (from Gemini usage_metadata):
    prompt_tokens         — input tokens (text + video/image frames)
    candidates_tokens     — output tokens
    thoughts_tokens       — thinking budget used (2.5 Pro only)
    tool_use_tokens       — tool-call prompt overhead
    cached_tokens         — cache-hit token count
    total_tokens          — sum reported by API
    calls                 — number of generate_content invocations
"""
from __future__ import annotations
import contextvars
from dataclasses import dataclass, field
from typing import Any, Optional


# Ambient accumulator — retry sites in frameworks call `record_retry_wait(seconds)`
# and we credit the active per-Q accumulator without threading a param through.
_current_accumulator: contextvars.ContextVar[Optional["TokenAccumulator"]] = \
    contextvars.ContextVar("_current_accumulator", default=None)


def set_current_accumulator(acc: Optional["TokenAccumulator"]) -> None:
    _current_accumulator.set(acc)


def record_retry_wait(seconds: float) -> None:
    acc = _current_accumulator.get()
    if acc is not None:
        acc.add_retry_wait(seconds)


@dataclass
class TokenAccumulator:
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    thoughts_tokens: int = 0
    tool_use_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    retry_wait_seconds: float = 0.0
    retries: int = 0

    def add_usage(self, um: Any) -> None:
        """Add a single response's usage_metadata."""
        if um is None:
            return
        self.prompt_tokens     += int(getattr(um, "prompt_token_count", 0) or 0)
        self.candidates_tokens += int(getattr(um, "candidates_token_count", 0) or 0)
        self.thoughts_tokens   += int(getattr(um, "thoughts_token_count", 0) or 0)
        self.tool_use_tokens   += int(getattr(um, "tool_use_prompt_token_count", 0) or 0)
        self.cached_tokens     += int(getattr(um, "cached_content_token_count", 0) or 0)
        self.total_tokens      += int(getattr(um, "total_token_count", 0) or 0)
        self.calls             += 1

    def add_retry_wait(self, seconds: float) -> None:
        """Record a backoff sleep triggered by a retryable API error.
        `elapsed_seconds - retry_wait_seconds = active compute time`.
        """
        self.retry_wait_seconds += float(seconds)
        self.retries += 1

    def snapshot(self) -> dict:
        return {
            "prompt_tokens":     self.prompt_tokens,
            "candidates_tokens": self.candidates_tokens,
            "thoughts_tokens":   self.thoughts_tokens,
            "tool_use_tokens":   self.tool_use_tokens,
            "cached_tokens":     self.cached_tokens,
            "total_tokens":      self.total_tokens,
            "calls":             self.calls,
            "retries":           self.retries,
            "retry_wait_seconds": round(self.retry_wait_seconds, 2),
        }

    def snapshot_diff(self, prev: dict | None) -> dict:
        """Return delta vs an earlier snapshot. None prev → returns current
        snapshot as-is (treats epoch as zero). Used to attach per-step token
        usage to a trajectory step: capture snapshot before the step, then
        snapshot_diff(prev) after.
        """
        cur = self.snapshot()
        if not prev:
            return cur
        return {
            k: round(cur[k] - prev.get(k, 0), 2) if isinstance(cur[k], float)
               else cur[k] - prev.get(k, 0)
            for k in cur
        }

    def reset(self) -> None:
        self.prompt_tokens = self.candidates_tokens = self.thoughts_tokens = 0
        self.tool_use_tokens = self.cached_tokens = self.total_tokens = 0
        self.calls = 0
        self.retry_wait_seconds = 0.0
        self.retries = 0


def wrap_openai_client(client, accumulator: TokenAccumulator):
    """Monkey-patch an `openai.AsyncOpenAI` (or `openai.OpenAI`) client so every
    `client.chat.completions.create(...)` call feeds usage into `accumulator`.

    Maps OpenAI usage fields onto our shared TokenAccumulator schema:
        prompt_tokens     -> prompt_tokens
        completion_tokens -> candidates_tokens
        total_tokens      -> total_tokens
        prompt_tokens_details.cached_tokens -> cached_tokens (if present)

    Works for both sync and async OpenAI clients (auto-detected). Returns the
    same client (mutated in place) for chaining.
    """
    import inspect as _inspect

    chat = getattr(client, "chat", None)
    if chat is None:
        return client
    completions = getattr(chat, "completions", None)
    if completions is None:
        return client
    orig_create = completions.create

    def _record(resp):
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        accumulator.prompt_tokens     += int(getattr(usage, "prompt_tokens", 0) or 0)
        accumulator.candidates_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        accumulator.total_tokens      += int(getattr(usage, "total_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            accumulator.cached_tokens += int(getattr(details, "cached_tokens", 0) or 0)
        accumulator.calls += 1

    if _inspect.iscoroutinefunction(orig_create):
        async def tracked_create(**kwargs):
            resp = await orig_create(**kwargs)
            _record(resp)
            return resp
    else:
        def tracked_create(**kwargs):
            resp = orig_create(**kwargs)
            _record(resp)
            return resp

    completions.create = tracked_create
    return client


_TOKEN_WRAP_SENTINEL = "_avb_token_wrapped_with"


def wrap_genai_client(client, accumulator: TokenAccumulator):
    """Monkey-patch a google-genai Client so every API call feeds into `accumulator`.

    BUG FIX (double-counting):
    1. **Idempotency**: skip re-wrapping a Models instance that is already wrapped
       with the same accumulator. Prevents `wrap_genai_client(c, acc)` followed by
       a planner that internally calls `wrap_genai_client(planner._client, acc)`
       (when planner._client is c) from counting every call twice.
    2. **Chat redirect avoidance**: `chat.send_message(...)` in google-genai
       internally calls `self._modules.generate_content(...)` (chats.py:252/414),
       which is the same Models/AsyncModels instance attached to the client. We
       therefore wrap ONLY the Models layer; chat usage propagates through there
       automatically. Wrapping `chats.create` separately would double-count every
       chat-based call.

    Patches both sync (`client.models.generate_content`) and async
    (`client.aio.models.generate_content`). Returns the client unchanged.
    """
    # Sync layer
    models = client.models
    if getattr(models, _TOKEN_WRAP_SENTINEL, None) is not accumulator:
        setattr(models, _TOKEN_WRAP_SENTINEL, accumulator)
        orig_generate = models.generate_content
        def tracked_generate(**kwargs):
            resp = orig_generate(**kwargs)
            accumulator.add_usage(getattr(resp, "usage_metadata", None))
            return resp
        models.generate_content = tracked_generate

    # Async layer (covers both direct aio.models.generate_content AND
    # aio.chats.send_message — chats.py routes send_message → aio.models.generate_content).
    aio = getattr(client, "aio", None)
    if aio is not None:
        aio_models = getattr(aio, "models", None)
        if aio_models is not None and getattr(aio_models, _TOKEN_WRAP_SENTINEL, None) is not accumulator:
            setattr(aio_models, _TOKEN_WRAP_SENTINEL, accumulator)
            orig_aio_gen = getattr(aio_models, "generate_content", None)
            if orig_aio_gen is not None:
                async def tracked_aio_gen(**kwargs):
                    resp = await orig_aio_gen(**kwargs)
                    accumulator.add_usage(getattr(resp, "usage_metadata", None))
                    return resp
                aio_models.generate_content = tracked_aio_gen

    return client
