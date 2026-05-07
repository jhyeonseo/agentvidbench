#!/usr/bin/env python3
"""AgentVidBench inference entry point.

One CLI for two frameworks (`singleturn`, `ours`) and the full set of model
backends (Vertex Gemini, OpenAI, Anthropic, vLLM-served open-source MLLMs).

Quick examples
--------------
  python inference.py --model gemini-2.5-pro     --framework ours       --tag run1
  python inference.py --model gpt-5              --framework singleturn --tag run1
  python inference.py --model claude-opus-4-7    --framework ours       --tag run1
  python inference.py --model Qwen/Qwen3-VL-4B-Instruct --framework singleturn --tag run1

Open-source models require a vLLM server (default endpoint
`http://localhost:8000/v1`). Start one in a separate shell, e.g.:

    pip install vllm==0.19.1
    vllm serve <model-id> --port 8000 \
        --allowed-local-media-path "$(pwd)/dataset/videos" \
        --tensor-parallel-size 1 --gpu-memory-utilization 0.85

Resume: re-running with the same `--tag` resumes; per-question results land in
`exp/<framework>_<model>_<tag>/inference/{progress/question<N>.json,
trajectories/question<N>.txt, outputs/, summary.json}`. A separate
`evaluate.py` populates `<run>/evaluation/` (letters, judge, results, summary).
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make `import framework.*` resolvable even when invoked as `python inference.py`.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.orchestration import (  # noqa: E402  (path setup must precede)
    DEFAULT_OUT_DIR,
    DEFAULT_QUESTIONS_JSONL,
    DEFAULT_VIDEO_DIR,
    DEFAULT_VIDEOS_JSONL,
    FRAMEWORKS,
    filter_questions,
    gcs_config_or_die,
    load_items,
    output_dir_for,
    setup_resume,
    write_summary,
)


# ---------------------------------------------------------------------------
# Backend pre-flight
# ---------------------------------------------------------------------------

def _is_open_source_model(model: str) -> bool:
    """vLLM-served open-source MLLMs (no first-party API)."""
    m = (model or "").lower()
    if m.startswith(("qwen", "gemma")):
        return True
    if "kimi-vl" in m or "kimi_vl" in m:
        return True
    if "/" in m and not m.startswith(("gpt", "o1", "o3", "o4", "claude", "gemini")):
        # Catch HF-prefixed ids like "Qwen/Qwen3-VL-4B-Instruct".
        return True
    return False


def _vllm_base_url() -> str:
    return os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")


def _check_vllm_alive(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/models", timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def _preflight_vllm_or_die(model: str) -> None:
    url = _vllm_base_url()
    if _check_vllm_alive(url):
        return
    sys.exit(
        "\n" + "=" * 72 + "\n"
        f"ERROR: vLLM server not reachable at {url}.\n"
        f"Open-source model '{model}' requires a vLLM server. Start one in a\n"
        f"separate shell:\n\n"
        f"    pip install vllm==0.19.1\n"
        f"    vllm serve {model} --port 8000 \\\n"
        f"        --allowed-local-media-path \"$(pwd)/dataset/videos\" \\\n"
        f"        --tensor-parallel-size 1 --gpu-memory-utilization 0.85\n\n"
        f"Override the endpoint with VLLM_BASE_URL=... if needed.\n"
        + "=" * 72
    )


def _preflight_api_keys_or_die(args) -> None:
    m = (args.model or "").lower()
    if m.startswith(("gpt", "o1", "o3", "o4")) and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY is not set (required for OpenAI models).")
    if m.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY is not set (required for Anthropic models).")
    # Gemini / Vertex auth + GCS bucket are validated by gcs_config_or_die.


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AgentVidBench inference — one entry point for both frameworks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--framework", required=True, choices=list(FRAMEWORKS),
                   help="Evaluation framework: 'singleturn' (one model call per Q) "
                        "or 'ours' (ReAct agent with analyze_video + get_transcript tools).")
    p.add_argument("--model", required=True,
                   help="Model id. Examples: gemini-2.5-pro, gpt-5, "
                        "claude-opus-4-7, Qwen/Qwen3-VL-4B-Instruct, "
                        "Qwen/Qwen3-VL-2B-Instruct, google/gemma-4-E2B-it. "
                        "Open-source models require a running vLLM server.")
    p.add_argument("--tag", default="run1",
                   help="Output dir suffix. Re-running with the same tag resumes.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-concurrent", type=int, default=3,
                   help="Parallel workers / concurrent items.")
    p.add_argument("--questions", default=None,
                   help="Comma-separated question_ids, e.g. '1,5,10' (default: all).")
    p.add_argument("--fps", type=int, default=1,
                   help="[singleturn] sampling fps (with fallback chain on Gemini); "
                        "[ours] default fps for the analyze_video tool.")
    p.add_argument("--max-frames", type=int, default=50,
                   help="[singleturn OpenAI/Anthropic/Kimi-VL] frame-extraction cap.")
    p.add_argument("--thinking-budget", type=int, default=None,
                   help="[singleturn, ours] thinking_budget tokens (None = backend default).")
    p.add_argument("--track-tokens", action="store_true",
                   help="[ours] wrap clients with TokenAccumulator.")
    p.add_argument("--questions-jsonl", default=str(DEFAULT_QUESTIONS_JSONL),
                   help="HF snapshot path to questions.jsonl.")
    p.add_argument("--videos-jsonl", default=str(DEFAULT_VIDEOS_JSONL),
                   help="HF snapshot path to videos.jsonl (cosmetic metadata; optional).")
    p.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR),
                   help="Local video directory (defaults to dataset/videos/).")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--dry-run", action="store_true",
                   help="Load items + print summary, do not call APIs.")
    p.add_argument("--verbose", action="store_true")
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Pre-flight checks fail fast and clear before items load / methods spin up.
    if _is_open_source_model(args.model):
        _preflight_vllm_or_die(args.model)
    _preflight_api_keys_or_die(args)
    gcs_config_or_die(args)  # AVB_GCS_BUCKET for Gemini-family runs.

    items = load_items(
        Path(args.questions_jsonl),
        Path(args.video_dir),
        Path(args.videos_jsonl) if args.videos_jsonl else None,
    )

    # `ours` requires a known duration so the planner's video-duration briefing
    # line is honest. ffprobe is always tried; if the metadata fallback also
    # fails, we abort rather than fabricating a value.
    if args.framework == "ours":
        bad = [it["question_id"] for it in items if not it.get("duration") or it["duration"] <= 0]
        if bad:
            sys.exit(
                f"ERROR (ours): could not determine video duration for question_ids "
                f"{bad[:10]}{'...' if len(bad) > 10 else ''}. Verify "
                f"dataset/videos/ is fully populated (ffprobe should succeed)."
            )

    items = filter_questions(items, args.questions)
    all_items = items  # snapshot before resume removes done items

    output_dir = output_dir_for(args, Path(args.out_dir))
    inference_dir = output_dir / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)

    existing = setup_resume(inference_dir)
    if existing:
        print(f"Resume: {len(existing)} items already done, skipping.")
        items = [it for it in items if it["question_id"] not in existing]

    print(f"=== {args.framework} ===")
    print(f"Model: {args.model} | Items: {len(items)} | Concurrent: {args.max_concurrent}")
    print(f"Output: {output_dir}")
    print("=" * 50)

    if args.dry_run:
        for it in items[:5]:
            print(f"  Q{it['question_id']}: {it['video_local_path']}")
        print(f"(dry-run: would process {len(items)} items)")
        return

    if not items:
        if existing:
            write_summary(inference_dir, list(existing.values()), args, 0.0, items=all_items)
        else:
            print("Nothing to do.")
        return

    # Method dispatch — each framework module exports
    #   run(items, args, output_dir) -> list[dict]
    # Runners receive the inference subdir; their progress/, trajectories/,
    # responses/, raw/ all land under <output_dir>/inference/.
    method_module = importlib.import_module(f"framework.methods.{args.framework}")
    t_start = time.time()
    new_results = method_module.run(items, args, inference_dir)
    total_wall = time.time() - t_start

    all_results = list(existing.values()) + list(new_results)
    write_summary(inference_dir, all_results, args, total_wall, items=all_items)


if __name__ == "__main__":
    main()
