"""Planner LLM interface — backbone-agnostic.

The orchestrator (ReAct loop) talks to the planner only through this
interface. Concrete implementations (GeminiPlanner, QwenPlanner, ...) live
next to this file and translate to/from each backend's wire format
(google-genai chat + function-calling, OpenAI chat completions + tools, etc.).

Tool execution happens in the orchestrator and tools/, not here. Tools
themselves are fixed Gemini and unaffected by which planner is in use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union


@dataclass
class PlannerToolCall:
    """A tool call requested by the planner in one turn."""
    name: str
    arguments: dict
    call_id: Optional[str] = None  # required by OpenAI-style backends; Gemini may omit


@dataclass
class ToolCallResult:
    """Result of executing one tool call, sent back to the planner next turn."""
    name: str
    result: str
    call_id: Optional[str] = None  # must echo the originating PlannerToolCall.call_id


@dataclass
class PlannerResponse:
    """One parsed turn of planner output."""
    thinking: str = ""
    text: str = ""
    tool_calls: list[PlannerToolCall] = field(default_factory=list)


# Convenience alias for the orchestrator's per-turn input
PlannerInput = Union[str, Sequence[ToolCallResult]]


class PlannerLLM(ABC):
    """Abstract planner LLM. The orchestrator depends only on this surface.

    Implementations own:
      - chat session state (multi-turn history)
      - backend-specific tool schema translation
      - retry / backoff on transient errors
      - token-tracking hookup
    """

    @abstractmethod
    async def start_chat(
        self,
        system_prompt: str,
        tool_schemas: Sequence[dict],
        temperature: float = 0.0,
        thinking_budget: Optional[int] = None,
    ) -> None:
        """Initialize a multi-turn chat session.

        `tool_schemas` are OpenAI-style function specs as produced by
        `BaseTool.to_schema()`; the implementation translates to its
        backend's native format.

        `thinking_budget=None` means: leave the backend's default thinking
        behavior alone (do not attach an explicit thinking config).
        """

    @abstractmethod
    async def send(self, message: PlannerInput) -> Optional[PlannerResponse]:
        """Send one user/tool message to the chat session.

        - If `message` is a `str`, it is treated as plain text (e.g. system
          briefing or fallback prompt).
        - If `message` is a sequence of `ToolCallResult`, each result is
          attached as a tool/function response part.

        Returns the parsed response, or `None` if all retries failed.
        Implementations are expected to handle their own retry/backoff
        and credit any sleep time to the active token tracker.
        """

    def attach_token_tracker(self, accumulator: Any) -> None:
        """Hook a token accumulator into this planner's underlying client.

        Default: no-op. Concrete planners override to wrap their backend
        client (e.g. `wrap_genai_client(self._client, accumulator)` for
        Gemini, or an OpenAI-equivalent for Qwen/GPT).
        """
        return None
