"""LLM-based letter extraction with N-vote majority.

For one (run, qid):
  1. Load model raw output from the canonical per-framework inference path:
       <run>/inference/trajectories/question<N>.txt
     (the runner is responsible for producing this file — see
     framework/methods/<method>/runner.py).
  2. Call Vertex Gemini 2.5 Flash with candidate_count=N (default 8) — single
     API call returns N independent extractions; each vote is the letter the
     model committed to, or null if it refused.
  3. Majority vote → final letter (with consensus strength).

Output (per-Q JSON, schema matches origin extract_letters.py for parity):
  {
    "_meta": {qid, n_runs, model_used, elapsed_seconds},
    "votes": ["C","C",null,...],
    "majority_letter": "C" | null,
    "agreement": 0.875,
    "n_null": 0,
    "raw_output_excerpt": "<last 600 chars>",
    "raw_output_hash": "sha1..."
  }

Ported (and trimmed) from origin/refactor/eval-pipeline-cleanup:benchmark/extract_letters.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

DEFAULT_MODEL = "gemini-2.5-flash"
MODEL_FALLBACK = ["gemini-2.5-flash", "gemini-2.5-pro"]


SYS_PROMPT = """A model answered a 26-option multiple-choice question (A-Z).

The model's final answer is always at the END of its output (last step or
last sentence). Earlier letter mentions are intermediate candidates — ignore
them. Look only at the last commitment.

If the last commitment is a letter, output that letter.
If there is no commitment (refusal / unable / error / empty), output null.

Output ONLY the JSON object below. No preamble like "Here is the JSON" or
"Sure". No markdown code fences. The response MUST start with { and end with }.

{
  "letter": "A"-"Z" or null,
  "reason": "<= 1 sentence"
}"""


def get_raw_output(framework: str, qid: int, run_dir: Path) -> str | None:
    """Return the canonical judge-input text for `qid`.

    Reads `<run>/inference/trajectories/question<N>.txt` — every framework
    runner writes this file; `framework` is accepted for symmetry with the
    rest of the pipeline but is not used to dispatch.
    """
    p = run_dir / "inference" / "trajectories" / f"question{qid}.txt"
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _trim_for_llm(text: str, head: int = 1000, tail: int = 4000) -> str:
    """Keep last `tail` chars (where final answer usually is) + first `head`
    chars. Total ≤ ~5000 chars for letter extraction."""
    if not text:
        return ""
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n\n... [truncated middle] ...\n\n" + text[-tail:]


def _hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def call_llm_n(user_text: str, n: int, model: str = DEFAULT_MODEL,
               temperature: float = 0.3,
               max_retries: int = 30,
               base_backoff: float = 1.0) -> tuple[list[str], str]:
    """Single Gemini API call with candidate_count=n. Aggressive retry on
    transient errors; permanent errors (auth) bubble up."""
    from google import genai
    from google.genai.types import GenerateContentConfig
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    models_to_try = [model] + [x for x in MODEL_FALLBACK if x != model]
    last_err = None
    for attempt in range(max_retries):
        for m in models_to_try:
            try:
                client = genai.Client(vertexai=True, project=project, location=location)
                cfg = GenerateContentConfig(
                    temperature=temperature,
                    candidate_count=n,
                    response_mime_type="application/json",
                    system_instruction=SYS_PROMPT,
                )
                resp = client.models.generate_content(model=m, contents=user_text, config=cfg)
                texts = []
                for cand in (resp.candidates or []):
                    t = ""
                    content = cand.content
                    if content and content.parts:
                        for p in content.parts:
                            if getattr(p, "text", None):
                                t += p.text
                    texts.append(t)
                if not texts:
                    raise RuntimeError("no candidates returned")
                return texts, m
            except Exception as ex:
                last_err = ex
                emsg = str(ex).lower()
                if any(k in emsg for k in (
                    "permission", "unauthorized", "401", "403",
                    "invalid_argument", "quota project",
                )):
                    raise
                continue
        sleep_for = min(30.0, base_backoff * (2 ** min(attempt, 5))) + random.uniform(0, 1)
        time.sleep(sleep_for)
    raise RuntimeError(f"call_llm_n exhausted {max_retries} retries: {last_err}")


def _parse_json(text: str) -> dict | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text).rstrip("`").rstrip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def _norm_letter(x):
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip().upper()
        if len(s) == 1 and "A" <= s <= "Z":
            return s
    return None


def extract_one_record(framework: str, qid: int, run_dir: Path,
                        n_runs: int = 8, model: str = DEFAULT_MODEL) -> dict:
    """Single Gemini API call with candidate_count=n_runs to get N votes at once."""
    t0 = time.time()
    raw = get_raw_output(framework, qid, run_dir)
    if not raw or not raw.strip():
        return {
            "_meta": {"qid": qid, "framework": framework, "n_runs": n_runs,
                      "model_used": None, "elapsed_seconds": 0,
                      "skipped": "no_raw_output"},
            "votes": [], "majority_letter": None, "agreement": 0,
            "n_null": 0, "raw_output_excerpt": "", "raw_output_hash": "",
        }

    user = (f"MODEL RAW OUTPUT:\n\n{_trim_for_llm(raw)}\n\n"
            "Extract the final answer letter. Output JSON only.")

    h = _hash(raw)
    votes: list[str | None] = []
    last_model = model
    try:
        texts, last_model = call_llm_n(user, n_runs, model=model, temperature=0.3)
        for text in texts:
            parsed = _parse_json(text) or {}
            votes.append(_norm_letter(parsed.get("letter")))
        while len(votes) < n_runs:
            votes.append(None)
    except Exception:
        votes = [None] * n_runs

    valid = [v for v in votes if v is not None]
    n_null = sum(1 for v in votes if v is None)
    if not valid:
        majority = None
        agreement = 0.0
    else:
        cnt = Counter(valid)
        top, top_n = cnt.most_common(1)[0]
        majority = top
        agreement = top_n / len(valid)
        # If null votes outnumber the top letter, prefer null (refusal)
        if n_null > top_n:
            majority = None
            agreement = n_null / n_runs

    return {
        "_meta": {
            "qid": qid, "framework": framework, "n_runs": n_runs,
            "model_used": last_model,
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "votes": votes,
        "majority_letter": majority,
        "agreement": round(agreement, 3),
        "n_null": n_null,
        "raw_output_excerpt": (raw or "")[-600:],
        "raw_output_hash": h,
    }
