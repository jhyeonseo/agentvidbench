"""Unified trajectory schema for agentic methods.

`methods/ours/` (ReAct) emits per-sample trajectories in its native format.
This module defines a *normalized* schema it is projected into, so the
LLM-judge evaluation can consume a single dataclass. The schema is designed
to accommodate alternative agentic patterns (plan/observe/reflect, etc.)
should additional methods be added later.

Design choices:
  - Schema is *additive*: a NormalizedTrajectory is written ALONGSIDE the
    method-native artifacts, never replacing them.
  - `phase` is an open vocabulary. The shipped `ours` method emits
    thinking / tool_call / tool_result / answer; the schema reserves
    plan / observe / reflect for plan-observe-reflect-style methods.
    `method_native_type` preserves the original category for audit.
  - `data` is intentionally a free-form dict — phase-specific fields live
    there. See PHASE_DATA_KEYS for the recommended keys per phase.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# Recommended keys for the `data` dict per phase. Not enforced — consumers
# should defensively `.get()` to allow forward compatibility.
PHASE_DATA_KEYS: dict[str, tuple[str, ...]] = {
    # ── ours (ReAct) ──────────────────────────────────────────────────────
    "thinking":    ("text",),
    "tool_call":   ("tool", "arguments"),
    "tool_result": (
        "tool", "success", "result_text",
        # parsed from the VLM's structured response (best-effort regex)
        "fps", "time_range", "coverage",
        "suggested_answer", "confidence",
    ),
    # ── reserved for plan-observe-reflect-style methods ───────────────────
    "plan":    ("watch_config", "rationale", "completion_criteria"),
    "observe": ("key_evidence", "reasoning", "model_call"),
    "reflect": ("sufficient", "query_confidence", "rationale"),
    # ── shared (every method ends with a final answer) ────────────────────
    "answer":  ("letter", "reasoning", "evidence_ts", "cited_evidence_ids"),
}

# Permitted phase values (validation hint, not enforced)
ALL_PHASES = frozenset(PHASE_DATA_KEYS.keys())


@dataclass
class NormalizedStep:
    """One discrete step in an agent's trajectory.

    `step_id` is monotonically increasing within a single trajectory (1-indexed).
    `phase` is one of ALL_PHASES. `method_native_type` preserves the original
    category emitted by the method, for audit.
    `elapsed_seconds`/`tokens` are step-local — None when the source method
    didn't capture them.
    """
    step_id: int
    phase: str
    method_native_type: str
    data: dict = field(default_factory=dict)
    elapsed_seconds: Optional[float] = None
    tokens: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # drop None scalars to keep JSON small (data dict stays as-is)
        for k in ("elapsed_seconds", "tokens"):
            if d.get(k) is None:
                d.pop(k, None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizedStep":
        return cls(
            step_id=int(d["step_id"]),
            phase=str(d["phase"]),
            method_native_type=str(d.get("method_native_type", "")),
            data=dict(d.get("data") or {}),
            elapsed_seconds=d.get("elapsed_seconds"),
            tokens=d.get("tokens"),
        )


@dataclass
class NormalizedTrajectory:
    """Per-sample trajectory in unified form. Pure inference output —
    ground_truth / predicted / correct intentionally absent so this artifact
    can be evaluated against the dataset by a separate downstream step.

    Cross-method consumers should iterate `steps` and dispatch on `phase`.
    `metrics` carries pre-computed counts so reports don't have to re-derive.
    """
    qid: int                # corresponds to dataset's question_id
    method: str             # "ours" (the only producer post-refactor)
    model: str
    question_text: str
    elapsed_seconds: float
    steps: list[NormalizedStep]
    tokens: dict = field(default_factory=dict)        # aggregate (whole sample)
    metrics: dict = field(default_factory=dict)       # n_tool_calls, n_replans, ...
    schema_version: str = "1.1"                       # bumped: dropped eval fields

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "question_id": self.qid,
            "method": self.method,
            "model": self.model,
            "question_text": self.question_text,
            "elapsed_seconds": self.elapsed_seconds,
            "tokens": self.tokens,
            "metrics": self.metrics,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizedTrajectory":
        return cls(
            # Accept either "question_id" (1.1) or legacy "qid" (1.0).
            qid=int(d.get("question_id", d.get("qid", 0))),
            method=str(d["method"]),
            model=str(d.get("model", "")),
            question_text=str(d.get("question_text", "")),
            elapsed_seconds=float(d.get("elapsed_seconds", 0.0) or 0.0),
            tokens=dict(d.get("tokens") or {}),
            metrics=dict(d.get("metrics") or {}),
            steps=[NormalizedStep.from_dict(s) for s in (d.get("steps") or [])],
            schema_version=str(d.get("schema_version", "1.1")),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "NormalizedTrajectory":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Metric helpers — pre-compute counts so reports/judges don't re-derive
# ---------------------------------------------------------------------------

def compute_basic_metrics(steps: list[NormalizedStep]) -> dict:
    """Phase-agnostic counts. Counts every phase in ALL_PHASES so reports
    don't have to re-derive them, regardless of which agentic method
    produced the trajectory."""
    n_tool_calls = sum(1 for s in steps if s.phase == "tool_call")
    n_observe = sum(1 for s in steps if s.phase == "observe")
    n_plans = sum(1 for s in steps if s.phase == "plan")
    n_reflects = sum(1 for s in steps if s.phase == "reflect")
    # rounds ≈ number of plans (plan-observe-reflect style) OR tool_calls
    # (ReAct, where each call is roughly one round).
    n_rounds = max(n_plans, n_tool_calls, n_observe)
    obs_chars = sum(
        len(str(s.data.get("result_text") or s.data.get("reasoning") or ""))
        for s in steps if s.phase in ("tool_result", "observe")
    )
    return {
        "n_steps": len(steps),
        "n_tool_calls": n_tool_calls,
        "n_observe": n_observe,
        "n_plans": n_plans,
        "n_reflects": n_reflects,
        "n_rounds": n_rounds,
        "n_replans": max(0, n_plans - 1),
        "total_observation_chars": obs_chars,
    }
