"""ReAct Orchestrator — ReAct loop on top of an abstract PlannerLLM.

The orchestrator is backbone-agnostic: it talks to the planner only through
`framework.models.base.PlannerLLM`. To swap in Qwen / GPT / etc. later, add a new
PlannerLLM implementation under `framework/models/<family>/` and pick it in
`framework/methods/ours/runner.py:build_planner()` — this file does not change.

Tools (`analyze_video`, `get_transcript`) live in `framework/tools/` (top-level,
shared across methods for fair comparison); the orchestrator just dispatches
them via `ToolRegistry`.

This is the ReAct-specific orchestrator used by methods that follow the
"plan → call tool → observe → repeat" pattern (currently only `methods/ours/`).
The user-facing entry is `inference.py` at the repo root; this file is one
of its building blocks.
"""

import json
import logging
import re
from typing import List, Optional

from framework.models.base import PlannerLLM, ToolCallResult
from framework.tools.base import ToolRegistry
from framework.methods.ours.trajectory import TrajectoryLogger

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 20
MAX_TOOL_CALLS = 15
# Tool *system* failure circuit breaker (opt-in): how many ToolResult
# .success=False outcomes whose `metadata.failure_kind` indicates a
# system-side failure (retry_exhausted | empty_response) we tolerate
# before aborting the question with "?". Validation failures
# (failure_kind=validation, e.g., bad args from the planner) are NOT
# counted — the planner gets a chance to retry with corrected args.
# When set to 1 the very first system failure aborts the question.
# When None (default), the breaker is disabled and the error string is
# fed to the planner so it may recover (mirrors the OLD pre-circuit-
# breaker behavior). Activate via runner.py from `--ours-tool-failure-breaker`.
MAX_TOOL_FAILURES_DEFAULT = None
# failure_kind values that count as system failures (vs validation).
_SYSTEM_FAILURE_KINDS = ("retry_exhausted", "empty_response")


class ReActOrchestrator:
    """ReAct-pattern agent for video multiple-choice QA."""

    def __init__(
        self,
        planner: PlannerLLM,
        tool_registry: ToolRegistry,
        model: str = "",
        system_prompt: Optional[str] = None,
        max_iterations: int = MAX_ITERATIONS,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_tool_failures: Optional[int] = MAX_TOOL_FAILURES_DEFAULT,
        thinking_budget: int = 8192,
        temperature: float = 0.0,
        verbose: bool = False,
    ):
        self._planner = planner
        self._tools = tool_registry
        # Used by _default_prompt to gate gemini-3-only resolution=high tip.
        # Match the planner backbone, not the tool VLM model — the system
        # prompt describes what the *planner* can ask analyze_video for.
        self._model = model
        self._max_iterations = max_iterations
        self._max_tool_calls = max_tool_calls
        # None disables the tool failure circuit breaker entirely (OLD
        # behavior: error strings flow back to the planner unchanged).
        self._max_tool_failures = max_tool_failures
        self._thinking_budget = thinking_budget
        self._temperature = temperature
        self._verbose = verbose
        self._system_prompt = system_prompt or self._default_prompt()

        self._tool_call_count = 0
        self._tool_failure_count = 0
        self.trajectory: Optional[TrajectoryLogger] = None

    def _default_prompt(self) -> str:
        # Resolution support depends on backbone:
        # gemini-3+ supports HIGH on video input; earlier models cap at MEDIUM.
        model_name = (self._model or "").lower()
        high_supported = "gemini-3" in model_name
        if high_supported:
            res_choices = "low/medium/high"
            res_tip = (
                '- Use resolution="high" on a NARROW time window when fine-grained '
                'visual detail (e.g. small text or numbers) is decisive for the '
                'answer. high resolution is token-heavy, so localize the moment '
                'first with low/medium, then re-inspect with high.'
            )
        else:
            res_choices = "low/medium"
            res_tip = (
                '- Use resolution="low" for broad scanning of long segments to save '
                'tokens; resolution="medium" (default) for normal analysis.'
            )

        return f"""\
You are a video analysis agent. Your task is to answer a multiple-choice question
about a video by carefully analyzing its visual content.

The tools available:

- analyze_video(start_time, end_time, fps, focus, resolution): inspect video (or a segment).
  resolution ∈ {{{res_choices}}}, default medium.
  Returns:
  * Focus Answer: direct answer to the focus, or "NOT OBSERVED IN SEGMENT"
  * Observations: what was seen, with timestamps
  * Coverage: rich/partial/none — how well observations cover the focus
  * Uncertainty: what remains unclear

- get_transcript(start_time, end_time): Whisper-transcribed speech segments.
  Returns:
  * segments: list of {{start, end, text}} — speech intervals with spoken text
  * segment_count: length of the segments list
  * note (if any): e.g. "No speech detected in this video (music-only or ambient audio)."

YOUR DECISION PROCESS:
1. Plan your first scan based on the QUESTION TYPE:
   - If the question asks about a specific moment, object, or theme:
     → fps=1, full video in one call is fine even for long videos.
   - If the question requires tracking/counting/ordering across the whole video
     (e.g., "how many times...", "what is the sequence...", "which Nth event..."):
     → Split the video into segments (each ~3-5 minutes) and scan sequentially.
     This ensures you don't miss anything in long videos.
2. Check the returned coverage level:
   - RICH → Accept the observations and synthesize your answer.
   - PARTIAL → Do ONE targeted deep dive on the uncertain area
     (higher fps, higher resolution, specific time range, focused query). Then decide.
   - NONE → Do 1-2 deep dives to gather more evidence. Then decide.
3. After deep dives, synthesize ALL evidence and make your final call.
   Tool outputs (event lists, suggested answers, counts) are advisory —
   you may disagree with any of them, especially when they seem
   inconsistent or don't match your direct observations.

DEEP DIVE TIPS:
- Use higher fps (3-10) on specific time ranges where uncertainty exists.
{res_tip}
- Use the "focus" parameter to ask about the specific uncertain aspect.
- Do NOT re-count things you already counted. Only verify specific instances.
- When scanning in segments, carry forward your findings mentally.

All timestamps are in SECONDS (e.g., 34 seconds, not 0:34).

TURN FORMAT — VERY IMPORTANT:
Before EVERY tool call (and before the final answer), output a short plain-text
block with these two lines, then the tool call (or the final JSON):

PLAN: <what you will do this turn and why — 1 sentence, reference which part of the video / what focus>
REASONING: <what you learned from prior tool results and why this next step follows — 1-2 sentences>

Keep each under ~40 words. Do not skip this block even when the next action seems obvious.

When ready, provide your final answer as JSON (still preceded by PLAN/REASONING):
```json
{{"answer": "<single letter>", "reasoning": "<brief explanation based on observed evidence>"}}
```
"""

    async def answer(
        self,
        question_text: str,
        video_duration: float,
        video_id: int = 0,
    ) -> str:
        """Run the agent to answer a multiple-choice question about a video.

        Returns: single answer letter (e.g., "A")
        """
        self._tool_call_count = 0
        self._tool_failure_count = 0
        self.trajectory = TrajectoryLogger(video_id=video_id)

        # Initialize planner chat with system prompt + tool schemas
        tool_schemas: List[dict] = [t.to_schema() for t in self._tools.list_tools()]
        await self._planner.start_chat(
            system_prompt=self._system_prompt,
            tool_schemas=tool_schemas,
            temperature=self._temperature,
            thinking_budget=self._thinking_budget,
        )

        briefing = "[VIDEO ANALYSIS TASK]\n"
        if video_duration and video_duration > 0:
            briefing += f"Video Duration: {video_duration:.1f} seconds\n"
        briefing += (
            f"\n{question_text}\n\n"
            f"Please begin your analysis. Use the analyze_video tool to examine the video."
        )

        current_input = briefing

        for iteration in range(self._max_iterations):
            iter_num = iteration + 1
            logger.info("--- Iteration %d/%d (tools used: %d) ---",
                        iter_num, self._max_iterations, self._tool_call_count)

            if self._tool_call_count >= self._max_tool_calls:
                logger.warning("Tool call budget exhausted (%d)", self._tool_call_count)
                current_input = (
                    "[SYSTEM] Tool call budget exhausted. "
                    "Please provide your final answer now as JSON: "
                    '{"answer": "<letter>", "reasoning": "<explanation>"}'
                )

            response = await self._planner.send(current_input)
            if response is None:
                break

            # Log thinking
            if response.thinking:
                self.trajectory.log_thinking(iter_num, response.thinking)
                if self._verbose:
                    print(f"\n[Iter {iter_num}] THINKING: {response.thinking[:2000]}")

            # Log text response (pre-action reasoning when tool calls exist, final response otherwise)
            if response.text:
                prefix = "[PRE-ACTION]" if response.tool_calls else "[RESPONSE]"
                self.trajectory.log_thinking(iter_num, f"{prefix} {response.text}")
                if self._verbose:
                    print(f"\n[Iter {iter_num}] {prefix[1:-1]}: {response.text[:2000]}")

            # No tool calls = agent is done (or stuck)
            if not response.tool_calls:
                # Must use at least 1 tool before answering (try up to 2 times)
                if self._tool_call_count == 0 and iter_num <= 2:
                    logger.warning("Agent tried to answer without using any tools, forcing tool use")
                    current_input = (
                        "[SYSTEM] You MUST analyze the video using the analyze_video tool "
                        "before answering. You have not used any tools yet. "
                        "Please call analyze_video to examine the video first."
                    )
                    continue

                answer = self._extract_answer(response.text)
                if answer:
                    reasoning = self._extract_reasoning(response.text)
                    self.trajectory.log_answer(iter_num, answer, reasoning)
                    logger.info("Agent answered: %s (iter %d)", answer, iter_num)
                    return answer
                current_input = (
                    "[SYSTEM] Please provide your final answer as JSON: "
                    '{"answer": "<letter>", "reasoning": "<explanation>"}'
                )
                continue

            # Execute tool calls
            logger.info("Agent requested %d tool call(s)", len(response.tool_calls))
            tool_results: list[ToolCallResult] = []

            # Hook for per-step token tracking — pull current accumulator
            # snapshot before the tool executes so we can compute the delta
            # afterwards. Best-effort: if no accumulator is set, snapshot is None.
            try:
                from framework._shared.token_tracker import _current_accumulator  # type: ignore
                _acc = _current_accumulator.get()
            except Exception:
                _acc = None

            import time as _time

            for tc in response.tool_calls:
                logger.info("  Tool: %s(%s)", tc.name,
                            json.dumps(tc.arguments, ensure_ascii=False))
                # Capture pre-tool snapshot for diff
                _pre_snap = _acc.snapshot() if _acc is not None else None
                _t0 = _time.time()

                self.trajectory.log_tool_call(iter_num, tc.name, tc.arguments)

                result = await self._tools.execute(tc.name, **tc.arguments)
                self._tool_call_count += 1
                # Only count *system* failures (retry exhausted / empty
                # response). Validation failures are the planner's fault and
                # it gets a chance to retry with corrected args.
                if not result.success:
                    _kind = (getattr(result, "metadata", None) or {}).get("failure_kind", "")
                    if _kind in _SYSTEM_FAILURE_KINDS:
                        self._tool_failure_count += 1

                _elapsed = _time.time() - _t0
                _tok_delta = _acc.snapshot_diff(_pre_snap) if _acc is not None else None

                result_str = result.to_str()
                # Pass ToolResult.metadata through (used to be dropped) and
                # attach per-step elapsed/tokens for normalized output.
                self.trajectory.log_tool_result(
                    iter_num, tc.name, result.success, result_str,
                    metadata=getattr(result, "metadata", None),
                    elapsed_seconds=round(_elapsed, 3),
                    tokens=_tok_delta,
                )

                if self._verbose:
                    print(f"  [RESULT] {tc.name}: {result_str[:1000]}")

                tool_results.append(ToolCallResult(
                    name=tc.name,
                    result=result_str,
                    call_id=tc.call_id,
                ))

            # Tool *system* failure circuit breaker (opt-in): when enabled,
            # short-circuit with "?" once enough VLM retry-exhausted /
            # empty_response failures accumulated. Disabled (max_tool_failures
            # is None) means the error strings keep flowing to the planner —
            # OLD behavior, gives the planner a chance to recover.
            if (
                self._max_tool_failures is not None
                and self._tool_failure_count >= self._max_tool_failures
            ):
                logger.warning(
                    "Tool system failure threshold reached (%d) — aborting with '?'",
                    self._tool_failure_count,
                )
                self.trajectory.log_answer(
                    iter_num, "?",
                    f"Tool system failure threshold reached ({self._tool_failure_count}/{self._max_tool_failures}); evidence insufficient",
                )
                return "?"

            current_input = tool_results

        logger.warning("Max iterations reached")
        # OLD-equivalent fallback: emit "A" when the planner exhausts its
        # 20-iteration budget without producing a parseable answer. Matches
        # agentvidbench_old/.../src/agent/orchestrator.py:294-296. The other
        # "?" exit paths (planner returns None, breaker fires, outer
        # try/except in process_one) keep their explicit-failure semantics.
        self.trajectory.log_answer(self._max_iterations, "A", "Max iterations reached, fallback")
        return "A"

    # ------------------------------------------------------------------
    # Final-answer extraction (text parsing only — backend-agnostic)
    # ------------------------------------------------------------------

    def _extract_answer(self, text: str) -> Optional[str]:
        if not text:
            return None

        json_match = re.search(r'\{[^}]*"answer"\s*:\s*"([A-Z])"[^}]*\}', text, re.IGNORECASE)
        if json_match:
            return json_match.group(1).upper()

        m = re.search(r'(?:answer|choice)\s*(?:is|:)\s*([A-Z])\b', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        m = re.search(r'final\s+answer[:\s]*([A-Z])\b', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        return None

    def _extract_reasoning(self, text: str) -> str:
        json_match = re.search(r'\{[^}]*"reasoning"\s*:\s*"([^"]*)"[^}]*\}', text, re.IGNORECASE)
        if json_match:
            return json_match.group(1)
        return ""
