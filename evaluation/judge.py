"""LLM judge for AgentVidBench process scoring.

Scores P1-P5 process axes + milestone coverage + failure tags for one
(question, framework) pair using an LLM judge (default: Anthropic
claude-opus-4-7).

Prediction is read from the canonical path every framework writes:
    `<run>/inference/trajectories/question<N>.txt`
The runner for each method is responsible for producing this file (singleturn
writes the raw model response; ours renders the normalized trajectory via
`framework.methods.ours.trajectory.TrajectoryLogger.render_judge_text`).

Singleturn predictions tend to score lower on P2 (evidence coverage) and P4
(sweep) because there's no visible inspection process — that low score is
itself the intended signal.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "judge_system.md"

# Per-provider model preference (first one tried, fallbacks if it errors).
PROVIDER_MODELS = {
    "anthropic":     ["claude-opus-4-7", "claude-opus-4-5"],
    "openai":        ["gpt-5", "gpt-5-2025-08-07"],
    "gemini_vertex": ["gemini-2.5-pro"],
}


def _load_system_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


SYSTEM_PROMPT = _load_system_prompt()


# =============================================================================
# Prediction loading
# =============================================================================

def read_prediction(framework: str, qid: int, run_dir: Path) -> str:
    """Read prediction text for `qid` from an inference run dir.

    The framework runner is responsible for producing the canonical
    judge-input file at `<run>/inference/trajectories/question<N>.txt`.
    `framework` is accepted for symmetry with the rest of the pipeline but is
    not used to dispatch — every method writes to the same path.
    """
    p = run_dir / "inference" / "trajectories" / f"question{qid}.txt"
    if not p.exists():
        raise FileNotFoundError(p)
    return p.read_text(encoding="utf-8")


# =============================================================================
# Provider calls
# =============================================================================

def anthropic_call(system: str, user: str, model_pref,
                   max_retries: int = 30, base_backoff: float = 1.0) -> tuple[str, str]:
    """Anthropic call with aggressive retry on transient errors. Permanent
    errors (auth, invalid request) bubble up after one full pass through
    model_pref."""
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(
        api_key=key,
        default_headers={"anthropic-beta": "context-1m-2025-08-07"},
    )
    last_err = None
    for attempt in range(max_retries):
        for model in model_pref:
            for use_temp in (True, False):
                try:
                    kw = {
                        "model": model,
                        "max_tokens": 8000,
                        "system": system,
                        "messages": [{"role": "user", "content": user}],
                    }
                    if use_temp:
                        kw["temperature"] = 0
                    resp = client.messages.create(**kw)
                    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                    return text, model
                except Exception as ex:
                    last_err = ex
                    emsg = str(ex).lower()
                    if use_temp and "temperature" in emsg:
                        continue
                    if any(k in emsg for k in (
                        "401", "403", "permission_denied", "authentication",
                        "model not found",
                    )):
                        raise
                    break
        sleep_for = min(30.0, base_backoff * (2 ** min(attempt, 5))) + random.uniform(0, 1)
        time.sleep(sleep_for)
    raise RuntimeError(f"anthropic_call exhausted {max_retries} retries: {last_err}")


def openai_call(system: str, user: str, model_pref) -> tuple[str, str]:
    import openai
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=key)
    last_err = None
    for model in model_pref:
        for use_temp in (True, False):
            try:
                kw = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {"type": "json_object"},
                    "seed": 42,
                }
                if use_temp:
                    kw["temperature"] = 0
                resp = client.chat.completions.create(**kw)
                return resp.choices[0].message.content or "", model
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if use_temp and ("temperature" in msg or "unsupported" in msg):
                    continue
                break
    raise RuntimeError(f"openai models all failed: {last_err}")


def gemini_vertex_call(system: str, user: str, model_pref) -> tuple[str, str]:
    """Vertex Gemini judge call. system_instruction set on config."""
    from google import genai
    from google.genai.types import GenerateContentConfig
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    last_err = None
    for model in model_pref:
        try:
            client = genai.Client(vertexai=True, project=project, location=location)
            cfg = GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                system_instruction=system,
            )
            resp = client.models.generate_content(model=model, contents=user, config=cfg)
            return resp.text or "", model
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"gemini_vertex models all failed: {last_err}")


PROVIDER_CALLERS = {
    "anthropic":     anthropic_call,
    "openai":        openai_call,
    "gemini_vertex": gemini_vertex_call,
}


# =============================================================================
# Prompt assembly
# =============================================================================

USER_TEMPLATE = """QUESTION_ID: {qid}
FRAMEWORK: {framework}
SWEEP_REQUIRED: {sweep_required}

QUESTION:
{question_text}

OPTIONS:
{options}

GT_MILESTONES (curated, fixed IDs M1..Mn — classify coverage status for EACH, in order):
{milestones_block}

PREDICTED_TRAJECTORY:
{prediction}

Evaluate this prediction against the rubric. Output the JSON object only.
Reminder: milestone_coverage MUST contain exactly {n_milestones} entries with IDs {milestone_ids} in this order."""


def options_block(options: list[dict]) -> str:
    return "\n".join(f"  {o['letter']}. {o['text']}" for o in options)


def milestones_block(ms: list[dict]) -> str:
    if not ms:
        return "(no curated milestones available — derive 3-6 from answer_explanation)"
    return "\n".join(
        f"  {m['id']} type={m.get('type','?')}: {m['description']}" for m in ms
    )


def compact_gt_trajectory(traj: dict | None, max_steps: int = 12, char_cap: int = 2000) -> str:
    if not traj or not traj.get("steps"):
        return "(no GT trajectory)"
    lines = []
    for i, step in enumerate(traj["steps"][:max_steps], 1):
        tool = step.get("tool", "?")
        args = step.get("args", {})
        thought = (step.get("thought") or "")[:120]
        obs = (step.get("observation") or "")[:160]
        args_str = ", ".join(f"{k}={v}" for k, v in (args or {}).items())[:120]
        lines.append(f"[{i}] {tool}({args_str})\n    thought: {thought}\n    obs: {obs}")
    txt = "\n".join(lines)
    if len(txt) > char_cap:
        txt = txt[:char_cap] + "\n... (truncated)"
    if traj.get("final_answer"):
        txt += f"\n[final] {traj['final_answer']}"
    return txt


def get_sweep_required(q: dict) -> str:
    """Return 'yes' / 'no' / 'unknown' based on q['ecom_required'] (bool)."""
    r = q.get("ecom_required")
    if r is True:  return "yes"
    if r is False: return "no"
    return "unknown"


def build_user(qid: int, framework: str, q: dict, pred_text: str) -> str:
    milestones = q.get("milestones") or []
    ids = [m["id"] for m in milestones]
    return USER_TEMPLATE.format(
        qid=qid,
        framework=framework,
        sweep_required=get_sweep_required(q),
        question_text=q["question_text"],
        options=options_block(q["options"]),
        answer_explanation=q.get("answer_explanation", ""),
        milestones_block=milestones_block(milestones),
        n_milestones=len(milestones) if milestones else "3-6",
        milestone_ids=", ".join(ids) if ids else "M1..Mn",
        prediction=pred_text,
    )


def parse_json_response(text: str) -> tuple[dict | None, str | None]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text).rstrip("`").rstrip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None, "no_json_object"
    try:
        return json.loads(m.group()), None
    except json.JSONDecodeError as e:
        return None, f"json_parse_error: {e}"


# =============================================================================
# Judge: two-branch design
#   score_accuracy  — pure compare of pre-extracted letter vs gold (no LLM)
#   score_process   — LLM judge produces P1-P5 + milestone coverage + tags
#   judge_one       — combines both into the canonical record
# =============================================================================

# Tertiary milestone scoring
_MC_SCORE = {"covered": 1.0, "partial": 0.5, "incorrect": 0.0, "missing": 0.0}

# Old verbose axis keys → P1..P5 (for converting LLM output)
_AXIS_KEY_MAP = {
    "task_understanding":      "P1",
    "gt_evidence_coverage":    "P2",
    "evidence_grounding":      "P3",
    "evidence_completeness":   "P4",
    "reasoning_faithfulness":  "P5",
}


def score_accuracy(pred_letter: str | None, gold_letter: str) -> dict:
    """Pure compare. No LLM. Returns the canonical accuracy block."""
    return {
        "gold": gold_letter,
        "pred": pred_letter,
        "correct": pred_letter == gold_letter and pred_letter is not None,
    }


def _process_to_canonical(parsed: dict, judge_model: str, elapsed_s: float) -> dict:
    """Convert raw LLM judge JSON to canonical process block."""
    old_scores = parsed.get("process_scores") or {}
    scores = {_AXIS_KEY_MAP[k]: v for k, v in old_scores.items() if k in _AXIS_KEY_MAP}
    for axis in ("P1", "P2", "P3", "P4", "P5"):
        scores.setdefault(axis, None)
    avail = [v for v in scores.values() if v is not None]
    traj = (sum(avail) / len(avail)) / 2 if avail else None

    new_mc, mc_sum = [], 0.0
    for m in (parsed.get("milestone_coverage") or []):
        status = (m.get("status") or "").lower()
        new_m = {"id": m.get("milestone_id"), "status": status,
                 "evidence": m.get("evidence_in_prediction")}
        new_mc.append(new_m)
        mc_sum += _MC_SCORE.get(status, 0.0)
    mc_rate = (mc_sum / len(new_mc)) if new_mc else None

    return {
        "scores": scores,
        "traj": traj,
        "mc_rate": mc_rate,
        "rationales": parsed.get("axis_rationales") or {},
        "milestones": new_mc,
        "failure_tags": parsed.get("failure_tags") or [],
        "summary": parsed.get("summary") or "",
        "_judge_model": judge_model,
        "_elapsed_s": elapsed_s,
    }


def score_process(qid: int, framework: str, q: dict, run_dir: Path,
                  provider: str = "anthropic") -> dict:
    """LLM judge call. Returns the canonical process block (or {_error:...} on
    failure). Loads prediction, runs LLM judge, normalizes output."""
    t0 = time.time()
    try:
        pred_text = read_prediction(framework, qid, run_dir)
    except FileNotFoundError as e:
        return {"_error": f"missing_pred: {e}", "_elapsed_s": round(time.time() - t0, 2)}
    user = build_user(qid, framework, q, pred_text)
    caller = PROVIDER_CALLERS[provider]
    models = PROVIDER_MODELS[provider]
    try:
        text, model = caller(SYSTEM_PROMPT, user, models)
    except Exception as e:
        return {"_error": f"api_error: {e!r}", "_elapsed_s": round(time.time() - t0, 2)}
    parsed, perr = parse_json_response(text)
    elapsed = round(time.time() - t0, 2)
    if parsed is None:
        return {"_error": f"parse: {perr}", "_judge_model": model,
                "_elapsed_s": elapsed, "_raw": text}
    return _process_to_canonical(parsed, judge_model=model, elapsed_s=elapsed)


def judge_one(qid: int, framework: str, q: dict, run_dir: Path,
              pred_letter: str | None, provider: str = "anthropic") -> dict:
    """Combine both branches into one canonical record.

    pred_letter is computed externally (see evaluation/extract_letters.py);
    we just compare it to q['answer'] for the accuracy block.
    """
    accuracy = score_accuracy(pred_letter, q["answer"])
    process = score_process(qid, framework, q, run_dir, provider=provider)
    judge_model = process.pop("_judge_model", None)
    elapsed_s = process.pop("_elapsed_s", None)
    error = process.pop("_error", None)
    process.pop("_raw", None)
    rec = {
        "question_id": qid,
        "framework": framework,
        "judge_model": judge_model,
        "elapsed_s": elapsed_s,
        "accuracy": accuracy,
        "process": process,
    }
    if error:
        rec["error"] = error
    return rec
