"""Video Analyzer Tool - analyze video segments via separate VLM call."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


def _allowed_resolutions(model_name: str) -> List[str]:
    """Return the resolution levels that the current backbone supports for video input.

    Gemini 2.5 Pro currently rejects MEDIA_RESOLUTION_HIGH on video input;
    Gemini 3.0 Pro accepts all three.
    """
    name = (model_name or "").lower()
    if "gemini-3" in name:
        return ["low", "medium", "high"]
    return ["low", "medium"]


_RESOLUTION_MAP = {
    "low":    "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high":   "MEDIA_RESOLUTION_HIGH",
}


class VideoAnalyzerTool(BaseTool):
    """Analyze a video segment using a separate VLM call.

    Operates on a GCS video URI. The VLM returns a structured response:
    observations, suggested_answer, confidence, key_evidence, uncertainty.
    """

    def __init__(
        self,
        gemini_client=None,
        model_name: str = "gemini-2.5-flash",
        video_uri: str = "",
        video_duration: float = 0.0,
        question_text: str = "",
        temperature: float = 0.0,
        thinking_budget: int = 8192,
        default_fps: int = 1,
    ):
        self._client = gemini_client
        self._model_name = model_name
        self._video_uri = video_uri
        self._video_duration = video_duration
        self._question_text = question_text
        self._temperature = temperature
        self._thinking_budget = thinking_budget
        self._default_fps = default_fps
        self._allowed_res = _allowed_resolutions(model_name)

    @property
    def name(self) -> str:
        return "analyze_video"

    @property
    def description(self) -> str:
        res_options = "/".join(self._allowed_res)
        res_base = (
            f"Use resolution to control per-frame visual detail ({res_options}, default medium). "
            "Use 'low' for broad scanning of long segments to save tokens, "
            "and 'medium' (default) for normal analysis."
        )
        if "high" in self._allowed_res:
            res_hint = (
                res_base
                + " Switch to 'high' for fine-grained reading such as small text or "
                  "numbers — high resolution is token-heavy, so use it only on a narrow "
                  "time window after you have localized the moment of interest."
            )
        else:
            res_hint = res_base
        return (
            "Analyze a segment of the video using VLM. "
            "Returns structured observations including: observations, suggested_answer, "
            "confidence (high/medium/low), key_evidence, and uncertainty. "
            "Use start_time/end_time to focus on specific parts. "
            "Use fps to control sampling density (higher = more frames per second, slower). "
            f"{res_hint} "
            "Use focus to guide what the VLM should pay attention to. "
            "IMPORTANT: If confidence is medium or low, you should investigate further "
            "with higher fps, higher resolution, or different time ranges before "
            "accepting the suggested answer."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="start_time",
                type="number",
                description="Start time in seconds. Omit to start from beginning.",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="end_time",
                type="number",
                description="End time in seconds. Omit to analyze to end.",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="fps",
                type="number",
                description=(
                    "Frames per second to sample. "
                    "1 for broad overview, 2-3 for detailed analysis, 5-20 for frame-level inspection. Max: 20."
                ),
                required=False,
                default=self._default_fps,
            ),
            ToolParameter(
                name="focus",
                type="string",
                description=(
                    "What to focus on during analysis. "
                    "E.g., 'count the number of cars', 'identify the brand on the tool', "
                    "'describe the instrument being played in close-up'."
                ),
                required=False,
                default=None,
            ),
            ToolParameter(
                name="resolution",
                type="string",
                description=(
                    "Per-frame visual resolution. "
                    "'low' = fast scan, fine details may be invisible. "
                    "'medium' = default, balanced visual detail. "
                    + (
                        "'high' = full visual detail, makes small text or numbers "
                        "readable; token-heavy, so only use on a narrow time window "
                        "after you have localized the moment. "
                        if "high" in self._allowed_res else ""
                    )
                    + "Choose based on what you need to see, not what you guess."
                ),
                required=False,
                default="medium",
                enum=list(self._allowed_res),
            ),
        ]

    async def execute(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        fps: Optional[float] = None,
        focus: Optional[str] = None,
        resolution: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        if not self._video_uri:
            return ToolResult(success=False, error="No video_uri set.",
                              metadata={"failure_kind": "validation"})

        effective_fps = max(1, min(int(fps or 1), 20))

        res_str = (resolution or "medium").lower()
        if res_str not in self._allowed_res:
            logger.warning(
                "resolution=%r not allowed for model=%s (allowed=%s); falling back to 'medium'.",
                res_str, self._model_name, self._allowed_res,
            )
            res_str = "medium"
        media_resolution_attr = _RESOLUTION_MAP[res_str]

        # Clamp negative values
        if start_time is not None and start_time < 0:
            start_time = 0
        if end_time is not None and end_time < 0:
            end_time = None
        if start_time is not None and end_time is not None and start_time >= end_time:
            return ToolResult(
                success=False,
                error=f"start_time ({start_time:.0f}s) must be less than end_time ({end_time:.0f}s).",
                metadata={"failure_kind": "validation"},
            )

        # Clamp time range
        if self._video_duration > 0:
            if start_time is not None and start_time > self._video_duration:
                return ToolResult(
                    success=False,
                    error=f"start_time ({start_time:.0f}s) exceeds video duration ({self._video_duration:.0f}s).",
                    metadata={"failure_kind": "validation"},
                )
            if end_time is not None and end_time > self._video_duration:
                end_time = self._video_duration

        time_desc = f"[{start_time or 0:.0f}s-{end_time or 'end'}s]"
        logger.info("analyze_video %s fps=%d res=%s focus=%s",
                    time_desc, effective_fps, res_str, focus)

        try:
            from google.genai.types import (
                GenerateContentConfig,
                MediaResolution,
                Part,
                ThinkingConfig,
                VideoMetadata,
            )

            video_part = Part.from_uri(file_uri=self._video_uri, mime_type="video/mp4")
            video_part.video_metadata = VideoMetadata(
                fps=effective_fps,
                start_offset=f"{start_time}s" if start_time is not None else None,
                end_offset=f"{end_time}s" if end_time is not None else None,
            )

            # Build prompt
            q_text = self._question_text
            for strip_phrase in [
                "Answer with ONLY the letter (A-Z) of your answer.",
                "Answer with ONLY the letter",
            ]:
                q_text = q_text.replace(strip_phrase, "").strip()

            focus_line = focus or "(no specific focus — observe broadly what is relevant to the question)"
            prompt = f"""\
You are a video observation assistant. An upstream multi-step agent is solving a problem
and has called you as one step in that process. Your job is NOT to solve the problem —
your job is to produce accurate, focused observations that the agent will aggregate
with other calls to reach a final answer. Stick to what you are asked.

[QUESTION CONTEXT]
{q_text}

[FOCUS]
{focus_line}

[INSTRUCTIONS]
1. Describe what you DIRECTLY OBSERVE in the video, with timestamps.
2. Prioritize the FOCUS — if a focus is given, answering it directly is your primary task.
3. Include additional observations relevant to the broader question context when useful.
4. If the focus target is NOT visible in the provided segment, explicitly say
   "FOCUS TARGET NOT OBSERVED IN THIS SEGMENT" and move on — do not speculate.
5. Rate how well your observations cover the focus: rich | partial | none.

[RESPONSE FORMAT]
**Focus Answer:** <direct concise answer to the focus, or "NOT OBSERVED IN SEGMENT">

**Observations:** <detailed observations with timestamps, only within the segment if one is specified>

**Coverage:** <rich | partial | none>

**Uncertainty:** <any visual ambiguity or occlusion>"""

            if start_time is not None or end_time is not None:
                prompt += (
                    f"\n\n[SEGMENT]\n"
                    f"You are only shown {start_time or 0:.0f}s to {end_time or 'end'}s.\n"
                    "All reported timestamps must be within this range.\n"
                    "Events outside this range must not be described."
                )

            config = GenerateContentConfig(
                temperature=self._temperature,
                response_modalities=["text"],
                media_resolution=getattr(MediaResolution, media_resolution_attr),
                thinking_config=ThinkingConfig(thinking_budget=self._thinking_budget),
            )

            response = None
            for attempt in range(5):
                try:
                    response = await self._client.aio.models.generate_content(
                        model=self._model_name,
                        contents=[video_part, prompt],
                        config=config,
                    )
                    break
                except Exception as api_err:
                    err_str = str(api_err).lower()
                    retryable = any(kw in err_str for kw in [
                        "429", "500", "503", "rate limit", "resource_exhausted",
                        "internal", "unavailable",
                    ])
                    if retryable and attempt < 4:
                        wait = 2 ** attempt * 3
                        logger.warning("VLM retry %d/5 in %ds: %s", attempt + 1, wait, api_err)
                        await asyncio.sleep(wait)
                        try:
                            from framework._shared.token_tracker import record_retry_wait
                            record_retry_wait(wait)
                        except Exception:
                            pass
                    else:
                        raise

            if response is None:
                return ToolResult(success=False, error="VLM call failed after retries.",
                                  metadata={"failure_kind": "retry_exhausted"})

            result_text = ""
            total_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                total_tokens = response.usage_metadata.total_token_count

            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        result_text += part.text

            if not result_text:
                return ToolResult(success=False, error="VLM returned no text.",
                                  metadata={"failure_kind": "empty_response"})

            return ToolResult(
                success=True,
                data=result_text,
                metadata={
                    "video_uri": self._video_uri,
                    "start_time": start_time,
                    "end_time": end_time,
                    "fps": effective_fps,
                    "resolution": res_str,
                    "tokens_used": total_tokens,
                },
            )

        except Exception as e:
            # Reaches here only when the inner retry loop re-raised after
            # 5 attempts (or hit a non-retryable error). Either way: the API
            # path is unrecoverable for this call → tag as retry_exhausted
            # so the orchestrator can short-circuit instead of letting the
            # planner fabricate an answer from no evidence.
            logger.error("Video analysis failed: %s", e)
            return ToolResult(success=False, error=str(e),
                              metadata={"failure_kind": "retry_exhausted"})
