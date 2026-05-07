"""GeminiPlanner — Vertex AI Gemini implementation of PlannerLLM.

Encapsulates everything Gemini-specific that the orchestrator used to do
inline: function declarations, GenerateContentConfig, ThinkingConfig,
async chat session management, function-call/thought parsing, and
retry/backoff on transient errors.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, Sequence

from google import genai
from google.genai.types import (
    FunctionDeclaration,
    GenerateContentConfig,
    Part,
    ThinkingConfig,
    Tool,
)

from framework.models.base import (
    PlannerLLM,
    PlannerResponse,
    PlannerToolCall,
    PlannerInput,
    ToolCallResult,
)

logger = logging.getLogger(__name__)

_RETRYABLE_KEYWORDS = (
    "429", "resource_exhausted", "500", "503",
    "internal", "unavailable",
)


class GeminiPlanner(PlannerLLM):

    def __init__(
        self,
        client: genai.Client,
        model: str = "gemini-2.5-pro",
        max_retries: int = 5,
    ):
        self._client = client
        self._model = model
        self._max_retries = max_retries
        self._chat = None  # set by start_chat()

    # ----- PlannerLLM interface ---------------------------------------------

    async def start_chat(
        self,
        system_prompt: str,
        tool_schemas: Sequence[dict],
        temperature: float = 0.0,
        thinking_budget: Optional[int] = None,
    ) -> None:
        # Translate OpenAI-style tool schemas → Gemini FunctionDeclarations
        function_declarations = []
        for schema in tool_schemas:
            func = schema["function"]
            function_declarations.append(FunctionDeclaration(
                name=func["name"],
                description=func["description"],
                parameters=func["parameters"],
            ))
        gemini_tools = (
            [Tool(function_declarations=function_declarations)]
            if function_declarations else None
        )

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": temperature,
        }
        if gemini_tools is not None:
            config_kwargs["tools"] = gemini_tools
        if thinking_budget is not None:
            config_kwargs["thinking_config"] = ThinkingConfig(
                thinking_budget=thinking_budget,
                include_thoughts=True,
            )
        config = GenerateContentConfig(**config_kwargs)

        self._chat = self._client.aio.chats.create(
            model=self._model,
            config=config,
        )

    async def send(self, message: PlannerInput) -> Optional[PlannerResponse]:
        if self._chat is None:
            raise RuntimeError("GeminiPlanner.send called before start_chat")

        wire = self._to_wire(message)

        # Local import — keeps token_tracker dep optional if planner is used standalone.
        try:
            from framework._shared.token_tracker import record_retry_wait
        except Exception:
            def record_retry_wait(_s):  # type: ignore
                pass

        for retry in range(self._max_retries):
            try:
                response = await self._chat.send_message(message=wire)
                return self._parse(response)
            except Exception as e:
                err_str = str(e).lower()
                if any(kw in err_str for kw in _RETRYABLE_KEYWORDS):
                    wait = 2 ** retry * 5
                    logger.warning("Gemini rate-limited / transient (%s), waiting %ds...", e, wait)
                    await asyncio.sleep(wait)
                    record_retry_wait(wait)
                    continue
                logger.error("Gemini API call failed: %s", e)
                return None
        logger.error("Gemini retries exhausted")
        return None

    def attach_token_tracker(self, accumulator: Any) -> None:
        # Local import — keeps token_tracker dep optional if planner is used standalone.
        from framework._shared.token_tracker import wrap_genai_client
        wrap_genai_client(self._client, accumulator)

    # ----- internals --------------------------------------------------------

    @staticmethod
    def _to_wire(message: PlannerInput):
        """Translate orchestrator-level message to Gemini's wire format."""
        if isinstance(message, str):
            return message
        # Sequence[ToolCallResult] → list[Part.from_function_response]
        return [
            Part.from_function_response(
                name=r.name,
                response={"result": r.result},
            )
            for r in message
        ]

    @staticmethod
    def _parse(response) -> PlannerResponse:
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        tool_calls: list[PlannerToolCall] = []

        cand = response.candidates[0] if response.candidates else None
        if cand is not None and cand.content is not None and cand.content.parts:
            for part in cand.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(PlannerToolCall(
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                        # Gemini doesn't surface a stable call_id; leave None
                    ))
                elif hasattr(part, "thought") and part.thought:
                    thinking_parts.append(part.text)
                elif hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

        return PlannerResponse(
            thinking="\n".join(thinking_parts),
            text="\n".join(text_parts),
            tool_calls=tool_calls,
        )
