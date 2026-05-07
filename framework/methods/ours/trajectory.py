"""Trajectory logger - records agent's thinking, tool calls, and results.

Three output formats are written per sample:
  1. Native — `outputs/question<N>.{json,txt}` — minimal step list (legacy,
     compat with existing exp/* analyses).
  2. Normalized — `outputs/question<N>.normalized.json` — unified schema
     with per-step elapsed_seconds + tokens snapshot diff + parsed VLM
     coverage/suggested_answer/confidence.
  3. Judge text — `trajectories/question<N>.txt` — canonical judge-input
     string. Same dir layout every framework writes to, so the evaluator
     reads `inference/trajectories/question<N>.txt` without branching.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from framework._shared.trajectory_schema import (
    NormalizedStep, NormalizedTrajectory, compute_basic_metrics,
)


# Regex helpers — extract structured fields from the VLM's free-text response
# (VideoAnalyzerTool produces "**Coverage:** rich" etc. by prompt design).
_RX_COVERAGE = re.compile(r"\*\*Coverage:\*\*\s*(rich|partial|none)", re.IGNORECASE)
_RX_FOCUS_ANSWER = re.compile(r"\*\*Focus Answer:\*\*\s*([^\n]+)", re.IGNORECASE)
_RX_CONFIDENCE = re.compile(r"confidence[:\s]+(high|medium|low)", re.IGNORECASE)


def _parse_vlm_fields(result_text: str) -> dict:
    """Best-effort regex extraction of structured fields from VLM response.

    Robust to absent fields. Returns empty dict if nothing matches.
    """
    if not isinstance(result_text, str):
        return {}
    out: dict[str, Any] = {}
    m = _RX_COVERAGE.search(result_text)
    if m:
        out["coverage"] = m.group(1).lower()
    m = _RX_FOCUS_ANSWER.search(result_text)
    if m:
        out["suggested_answer"] = m.group(1).strip()
    m = _RX_CONFIDENCE.search(result_text)
    if m:
        out["confidence"] = m.group(1).lower()
    return out


@dataclass
class TrajectoryStep:
    iteration: int
    step_type: str  # "thinking", "tool_call", "tool_result", "answer"
    data: dict = field(default_factory=dict)
    # New (additive — older code reading TrajectoryStep ignores these):
    elapsed_seconds: Optional[float] = None
    tokens: Optional[dict] = None


class TrajectoryLogger:
    """Records the full agent trajectory for inspection."""

    def __init__(self, video_id: int = 0):
        self.video_id = video_id
        self.steps: list[TrajectoryStep] = []

    def log_thinking(self, iteration: int, text: str):
        self.steps.append(TrajectoryStep(
            iteration=iteration,
            step_type="thinking",
            data={"text": text},
        ))

    def log_tool_call(self, iteration: int, tool_name: str, arguments: dict,
                      elapsed_seconds: Optional[float] = None,
                      tokens: Optional[dict] = None):
        self.steps.append(TrajectoryStep(
            iteration=iteration,
            step_type="tool_call",
            data={"tool": tool_name, "arguments": arguments},
            elapsed_seconds=elapsed_seconds,
            tokens=tokens,
        ))

    def log_tool_result(self, iteration: int, tool_name: str, success: bool,
                        result: str, metadata: Optional[dict] = None,
                        elapsed_seconds: Optional[float] = None,
                        tokens: Optional[dict] = None):
        """Record tool result. New optional `metadata` arg captures the
        ToolResult.metadata dict (fps, time_range, tokens_used, ...) which
        used to be dropped. `elapsed_seconds` and `tokens` are step-local.
        """
        data: dict[str, Any] = {"tool": tool_name, "success": success, "result": result}
        if metadata:
            # Preserve as-is under a sub-key so existing consumers reading
            # data["result"] keep working unchanged.
            data["metadata"] = dict(metadata)
        self.steps.append(TrajectoryStep(
            iteration=iteration,
            step_type="tool_result",
            data=data,
            elapsed_seconds=elapsed_seconds,
            tokens=tokens,
        ))

    def log_answer(self, iteration: int, answer: str, reasoning: str = ""):
        self.steps.append(TrajectoryStep(
            iteration=iteration,
            step_type="answer",
            data={"answer": answer, "reasoning": reasoning},
        ))

    def to_text(self) -> str:
        """Render trajectory as readable text."""
        lines = [f"=== Video {self.video_id} Trajectory ===\n"]

        for step in self.steps:
            prefix = f"[Iter {step.iteration}]"

            if step.step_type == "thinking":
                lines.append(f"{prefix} THINKING:")
                lines.append(step.data["text"])
                lines.append("")

            elif step.step_type == "tool_call":
                args_str = json.dumps(step.data["arguments"], ensure_ascii=False)
                lines.append(f"{prefix} TOOL CALL: {step.data['tool']}({args_str})")

            elif step.step_type == "tool_result":
                status = "OK" if step.data["success"] else "FAIL"
                lines.append(f"{prefix} TOOL RESULT [{status}]: {step.data['tool']}")
                lines.append(step.data["result"])
                lines.append("")

            elif step.step_type == "answer":
                lines.append(f"{prefix} FINAL ANSWER: {step.data['answer']}")
                if step.data.get("reasoning"):
                    lines.append(f"  Reasoning: {step.data['reasoning']}")
                lines.append("")

        return "\n".join(lines)

    def save(self, output_dir: str):
        """Save trajectory in NATIVE format (text + json). Normalized form is
        written by `save_normalized` separately so callers can pass the
        question / answer / aggregate metadata."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"question{self.video_id}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_text())

        # Also save as JSON for programmatic access (legacy schema preserved)
        json_path = os.path.join(output_dir, f"question{self.video_id}.json")
        json_data = []
        for s in self.steps:
            row: dict[str, Any] = {
                "iteration": s.iteration,
                "type": s.step_type,
                **s.data,
            }
            # Augment with new fields IFF present (legacy consumers ignore them).
            if s.elapsed_seconds is not None:
                row["elapsed_seconds"] = s.elapsed_seconds
            if s.tokens is not None:
                row["tokens"] = s.tokens
            json_data.append(row)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------------------
    # Normalized output
    # ----------------------------------------------------------------------

    def to_normalized(
        self,
        qid: int,
        model: str,
        question_text: str,
        elapsed_seconds: float,
        tokens: dict | None = None,
    ) -> NormalizedTrajectory:
        """Project the native step list onto NormalizedTrajectory.

        Maps:
          step_type=thinking    → phase=thinking, data={text}
          step_type=tool_call   → phase=tool_call, data={tool, arguments}
          step_type=tool_result → phase=tool_result, data={tool, success,
                                  result_text, fps, time_range, coverage,
                                  suggested_answer, confidence, ...metadata}
          step_type=answer      → phase=answer,   data={letter, reasoning}
        """
        norm_steps: list[NormalizedStep] = []
        for i, s in enumerate(self.steps, 1):
            if s.step_type == "tool_result":
                # Extract structured fields from VLM text (Coverage etc.)
                result_text = s.data.get("result", "")
                parsed = _parse_vlm_fields(result_text)
                meta = s.data.get("metadata") or {}
                data = {
                    "tool": s.data.get("tool"),
                    "success": s.data.get("success", True),
                    "result_text": result_text,
                    # metadata fields (from ToolResult.metadata): fps,
                    # start_time, end_time, tokens_used, video_uri, etc.
                    "fps": meta.get("fps"),
                    "time_range": [meta.get("start_time"), meta.get("end_time")]
                                   if (meta.get("start_time") is not None or
                                       meta.get("end_time") is not None) else None,
                    "tokens_used": meta.get("tokens_used"),
                    # parsed-from-text fields:
                    **parsed,
                }
                # drop None-valued keys to keep JSON tidy
                data = {k: v for k, v in data.items() if v is not None}
                norm_steps.append(NormalizedStep(
                    step_id=i, phase="tool_result",
                    method_native_type="tool_result",
                    data=data,
                    elapsed_seconds=s.elapsed_seconds,
                    tokens=s.tokens,
                ))
            elif s.step_type == "tool_call":
                norm_steps.append(NormalizedStep(
                    step_id=i, phase="tool_call",
                    method_native_type="tool_call",
                    data={
                        "tool": s.data.get("tool"),
                        "arguments": s.data.get("arguments") or {},
                    },
                    elapsed_seconds=s.elapsed_seconds,
                    tokens=s.tokens,
                ))
            elif s.step_type == "thinking":
                norm_steps.append(NormalizedStep(
                    step_id=i, phase="thinking",
                    method_native_type="thinking",
                    data={"text": s.data.get("text", "")},
                    elapsed_seconds=s.elapsed_seconds,
                    tokens=s.tokens,
                ))
            elif s.step_type == "answer":
                norm_steps.append(NormalizedStep(
                    step_id=i, phase="answer",
                    method_native_type="answer",
                    data={
                        "letter": s.data.get("answer", ""),
                        "reasoning": s.data.get("reasoning", ""),
                    },
                ))
            # else: silently drop unknown types (forward compat)

        return NormalizedTrajectory(
            qid=qid, method="ours", model=model,
            question_text=question_text,
            elapsed_seconds=elapsed_seconds,
            tokens=dict(tokens or {}),
            steps=norm_steps,
            metrics=compute_basic_metrics(norm_steps),
        )

    def save_normalized(
        self,
        output_dir: str,
        qid: int,
        model: str,
        question_text: str,
        elapsed_seconds: float,
        tokens: dict | None = None,
    ) -> NormalizedTrajectory:
        """Write outputs/question<N>.normalized.json. Returns the
        NormalizedTrajectory so the caller can hand it to `save_judge_text`
        without re-rendering."""
        os.makedirs(output_dir, exist_ok=True)
        nt = self.to_normalized(
            qid=qid, model=model, question_text=question_text,
            elapsed_seconds=elapsed_seconds, tokens=tokens,
        )
        path = os.path.join(output_dir, f"question{self.video_id}.normalized.json")
        nt.save(path)
        return nt

    # ----------------------------------------------------------------------
    # Canonical judge-input text — every framework writes one of these to
    # inference/trajectories/question<N>.txt so the evaluator reads it
    # without branching on framework.
    # ----------------------------------------------------------------------

    @staticmethod
    def render_judge_text(nt: "NormalizedTrajectory | dict") -> str:
        """Render a normalized trajectory as judge-friendly text.

        Each step is dumped as pretty-printed JSON so no field is silently
        dropped. Letter-revealing wrappers (`Predicted: X`, `[FINAL ANSWER] X`)
        are intentionally NOT included — the judge prompt is letter-leakage-free.
        """
        d = nt.to_dict() if hasattr(nt, "to_dict") else dict(nt)
        qid = d.get("question_id", d.get("qid"))
        lines = [f"=== {d.get('method','?')} trajectory "
                 f"(model={d.get('model','?')}, qid={qid}) ===\n"]
        if d.get("question_text"):
            lines.append(f"Question: {d['question_text']}\n")
        metrics = d.get("metrics") or {}
        if metrics:
            lines.append(f"Metrics: n_steps={metrics.get('n_steps')} "
                         f"n_tool_calls={metrics.get('n_tool_calls')} "
                         f"n_observe={metrics.get('n_observe')} "
                         f"n_rounds={metrics.get('n_rounds')}")
        lines.append("")
        for i, st in enumerate(d.get("steps", []) or [], 1):
            lines.append(f"[Step {i}]")
            lines.append(json.dumps(st, ensure_ascii=False, indent=2))
        return "\n".join(lines)

    def save_judge_text(
        self,
        trajectories_dir: str,
        nt: "NormalizedTrajectory",
    ) -> None:
        """Write inference/trajectories/question<N>.txt — the canonical
        judge-input string for this run."""
        os.makedirs(trajectories_dir, exist_ok=True)
        path = os.path.join(trajectories_dir, f"question{self.video_id}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render_judge_text(nt))
