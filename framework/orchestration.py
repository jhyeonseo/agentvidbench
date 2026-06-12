"""AgentVidBench shared orchestration utilities.

Pure library — no CLI. The user-facing entry point is `inference.py` at the
repo root. This module owns:

  - Loading benchmark items from the HF dataset snapshot at `./dataset/`
      questions.jsonl  one row per question (incl. `transcript_path`)
      videos.jsonl     cosmetic metadata (one row per unique video)
      videos/          mp4 files
      transcripts/     Whisper SRTs (per-video)

  - Per-question prompt rendering (in-memory; no prompt files on disk).
  - Output directory resolution for `singleturn` and `ours`.
  - Resume logic (skip qids whose progress/question*.json already looks complete).
  - per-run inference summary aggregation (`<output_dir>/summary.json`).
  - GCS env validation for Gemini-family runs.

Per-method work (process_one, tool wiring, ReAct loop, etc.) lives in
`framework.methods.<framework>.run(items, args, output_dir)`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_QUESTIONS_JSONL = DATASET_DIR / "questions.jsonl"
DEFAULT_VIDEOS_JSONL = DATASET_DIR / "videos.jsonl"
DEFAULT_VIDEO_DIR = DATASET_DIR / "videos"
DEFAULT_TRANSCRIPT_DIR = DATASET_DIR / "transcripts"
DEFAULT_OUT_DIR = REPO_ROOT / "exp"

DEFAULT_GCS_PREFIX = "agentvidbench/videos/eval"

FRAMEWORKS = ("singleturn", "ours")


# ---------------------------------------------------------------------------
# Video duration
# ---------------------------------------------------------------------------

def ffprobe_duration(local_path: Path) -> float | None:
    """Probe duration of an mp4 file. Returns None on failure."""
    if not local_path.exists():
        return None
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    try:
        out = subprocess.check_output(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(local_path)],
            text=True, timeout=15,
        )
        return float(out.strip())
    except Exception:
        return None


def parse_duration(dur) -> float:
    """Parse duration like '2:26', '~17:00', '1:23:45' to seconds.

    Returns 0.0 when the input is missing/unknown/unparseable. Callers
    (e.g. ReActOrchestrator) skip the 'Video Duration: …' briefing line
    when this is 0 — wrong defaults previously leaked into the agent
    prompt as 'video is 10 minutes long' for any video whose metadata
    duration was 'unknown', producing fake plans that overshot the
    actual length.
    """
    if isinstance(dur, (int, float)):
        return float(dur)
    if not isinstance(dur, str):
        return 0.0
    s = dur.strip().lstrip("~").strip()
    if not s:
        return 0.0
    try:
        parts = [int(p) for p in s.split(":")]
    except ValueError:
        return 0.0
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0.0


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def strip_answer_instruction(prompt_text: str) -> str:
    """Remove the trailing 'Answer with ONLY the letter' instruction so methods
    can build their own response-format prompt."""
    for phrase in [
        "Answer with ONLY the letter (A-Z) of your answer.",
        "Answer with ONLY the letter",
    ]:
        prompt_text = prompt_text.replace(phrase, "").strip()
    return prompt_text


def _build_prompt(question_text: str, options: list) -> str:
    """Render the canonical prompt for a question + 26 A-Z options.

    Accepts either plain strings or {"letter","text"} dicts (HF dataset
    shape). Dicts are unwrapped to their `text` so raw dict reprs don't
    leak into the model prompt.
    """
    lines = [f"Question: {question_text}", "", "Options:"]
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i)
        text = opt["text"] if isinstance(opt, dict) else opt
        lines.append(f"{letter}) {text}")
    lines.append("")
    lines.append("Answer with ONLY the letter (A-Z) of your answer.")
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Run-shape predicates
# ---------------------------------------------------------------------------

def is_gemini_family(args) -> bool:
    """Whether this run will hit the Gemini API (and therefore needs GCS).

    `ours` always uses Vertex Gemini for the analyze_video tool, regardless
    of the planner backend. `singleturn` uses Gemini only when --model is
    `gemini-*`.
    """
    if args.framework == "ours":
        return True
    return (args.model or "").lower().startswith("gemini")


def gcs_config_or_die(args) -> tuple[str, str] | None:
    """Validate GCS env for Gemini-family runs. Returns (bucket, prefix) or
    None for non-Gemini runs."""
    if not is_gemini_family(args):
        return None
    bucket = os.environ.get("AVB_GCS_BUCKET", "").strip()
    if not bucket:
        sys.exit(
            "ERROR: AVB_GCS_BUCKET is not set.\n"
            "Gemini-family runs upload local videos to GCS so Vertex AI can read them.\n"
            "  1. Create / pick a bucket:  gcloud storage buckets create gs://<name>\n"
            "  2. Authenticate ADC:        gcloud auth application-default login\n"
            "  3. Set in your .env:        AVB_GCS_BUCKET=<name>\n"
            "                              AVB_GCS_PREFIX=agentvidbench/videos/eval  (optional)"
        )
    prefix = os.environ.get("AVB_GCS_PREFIX", DEFAULT_GCS_PREFIX).strip()
    return bucket, prefix


# ---------------------------------------------------------------------------
# Item loading
# ---------------------------------------------------------------------------

def load_items(
    questions_jsonl: Path,
    video_dir: Path,
    videos_jsonl: Path | None = None,
    transcript_dir: Path | None = None,
):
    """Load benchmark items from the HF dataset snapshot.

    Returns one dict per question with everything any method needs:
        qid, video_id, video_local_path, video_path (alias), gcs_uri (None;
        populated lazily by Gemini-family runners), prompt_text (stripped)
        + prompt_text_raw, duration (ffprobe → metadata fallback), answer,
        video_title, options [{letter,text}], category, difficulty,
        question_text, transcript_path (absolute, must exist).

    Raises SystemExit if a required file is missing — no silent degradation.
    """
    if not questions_jsonl.exists():
        sys.exit(
            f"ERROR: {questions_jsonl} not found.\n"
            "Snapshot the HF dataset first:\n"
            "  hf download agentvidbench/agentvidbench --repo-type dataset --local-dir dataset"
        )

    rows = _read_jsonl(questions_jsonl)

    video_meta: dict[str, dict] = {}
    if videos_jsonl is not None and videos_jsonl.exists():
        for v in _read_jsonl(videos_jsonl):
            video_meta[v["file_name"]] = v

    transcript_dir = transcript_dir or DEFAULT_TRANSCRIPT_DIR

    items = []
    for row in sorted(rows, key=lambda r: r["question_id"]):
        qid = int(row["question_id"])
        rel_video = row["video_path"]              # e.g. "videos/video1.mp4"
        # Prefer --video-dir if it has the file (lets users supply a transcoded
        # copy without modifying the HF snapshot). Fall back to DATASET_DIR.
        local_path = (video_dir / Path(rel_video).name).resolve()
        if not local_path.exists():
            local_path = (DATASET_DIR / rel_video).resolve()
        if not local_path.exists():
            sys.exit(
                f"ERROR: missing video file for Q{qid}: {rel_video}\n"
                f"Run `hf download agentvidbench/agentvidbench --repo-type dataset --local-dir dataset` "
                f"to repopulate dataset/videos/."
            )

        # Resolve transcript path. The HF dataset embeds it per-question
        # ("transcripts/videoN.srt"); fall back to <transcript_dir>/<video-stem>.srt.
        rel_transcript = row.get("transcript_path") or f"transcripts/{Path(rel_video).stem}.srt"
        transcript_path = (DATASET_DIR / rel_transcript).resolve()
        if not transcript_path.exists():
            sys.exit(
                f"ERROR: missing transcript file for Q{qid}: {rel_transcript}\n"
                f"Run `hf download agentvidbench/agentvidbench --repo-type dataset --local-dir dataset` "
                f"to repopulate dataset/transcripts/."
            )

        options_list = list(row.get("options") or [])
        options_dicts = [
            {"letter": opt.get("letter", chr(ord("A") + i)), "text": opt["text"]}
            if isinstance(opt, dict)
            else {"letter": chr(ord("A") + i), "text": opt}
            for i, opt in enumerate(options_list)
        ]

        raw_prompt = _build_prompt(row["question_text"], options_list)
        stripped_prompt = strip_answer_instruction(raw_prompt)

        probed = ffprobe_duration(local_path)
        if probed is None:
            probed = parse_duration(video_meta.get(rel_video, {}).get("duration"))
        duration = probed

        items.append({
            "question_id": qid,
            # Relative path as carried in the HF dataset row, e.g.
            # "videos/video1.mp4". Emitted into per-question artifacts as
            # the canonical video reference.
            "video_path": rel_video,
            "video_local_path": str(local_path),
            "gcs_uri": None,
            "transcript_path": str(transcript_path),
            "prompt_text": stripped_prompt,
            "prompt_text_raw": raw_prompt,
            "duration": duration,
            # answer is preserved in items for prompt construction by some
            # legacy runners but is NOT propagated into the per-question
            # artifact — inference and evaluation are kept separate.
            "answer": row["answer"],
            "video_title": video_meta.get(rel_video, {}).get("title", row.get("title", "")),
            "options": options_dicts,
            "category": row.get("category", row.get("categories", "")),
            "difficulty": row.get("difficulty", ""),
            "question_text": row["question_text"],
        })
    return items


def filter_questions(items, questions_arg: str | None):
    """Filter items by --questions '1,5,10' (comma-separated question_ids)."""
    if not questions_arg:
        return items
    wanted = {int(x) for x in questions_arg.split(",")}
    return [it for it in items if it["question_id"] in wanted]


# ---------------------------------------------------------------------------
# Output dir + resume + summary
# ---------------------------------------------------------------------------

def output_dir_for(args, out_dir: Path) -> Path:
    """Compute output directory by framework: exp/<framework>_<model>_<tag>/."""
    model_slug = args.model.replace("/", "_")
    if args.framework not in FRAMEWORKS:
        raise ValueError(f"Unknown framework: {args.framework}")
    return out_dir / f"{args.framework}_{model_slug}_{args.tag}"


def setup_resume(output_dir: Path):
    """Scan progress/question*.json for already-completed items.

    Returns existing_results: dict[question_id -> result]. A run is "done"
    when the progress file has no `error` key AND carries `elapsed_seconds`
    (every backend writes it). Errored Qs (e.g. 429) re-run on next invocation.
    """
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    existing = {}
    for fname in os.listdir(progress_dir):
        if fname.startswith("question") and fname.endswith(".json"):
            try:
                with open(progress_dir / fname, "r", encoding="utf-8") as f:
                    r = json.load(f)
            except Exception:
                continue
            if "error" in r:
                continue
            if r.get("elapsed_seconds"):
                existing[r["question_id"]] = r
    return existing


def write_summary(output_dir: Path, results, args, total_wall_time: float, items=None):
    """Write summary.json — pure inference manifest, no evaluation fields.

    Correctness / accuracy / per-difficulty scoring is computed downstream by
    `evaluate.py`, which joins this artifact against the HF dataset (each
    result carries `question_id` and `video_path`).
    """
    results = sorted(results, key=lambda r: r["question_id"])
    errored = sum(1 for r in results if "error" in r)

    summary = {
        "experiment": args.framework,
        "model": args.model,
        "tag": args.tag,
        "total": len(results),
        "errored": errored,
        "total_wall_time": round(total_wall_time, 1),
        "results": results,
    }
    if args.framework == "singleturn":
        summary["fps"] = args.fps

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"Inference complete: {len(results)} items ({errored} errored)")
    print(f"Total time: {total_wall_time:.0f}s = {total_wall_time/60:.1f}min")
    print(f"\nSaved {summary_path}")
