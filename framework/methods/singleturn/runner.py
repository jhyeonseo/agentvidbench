"""Single-turn baseline — single generate call per question.

Five backends, dispatched by model name:

  * Gemini family (`gemini-*`): single `generate_content` via Vertex AI / API key.
    - fps fallback chain [10, 5, 3, 1] handles 500 INTERNAL on long videos.
    - Video source: GCS URI (preferred) or local file inline.

  * OpenAI proper (`gpt-*`, `o1*`, `o3*`, `o4*`): OpenAI Chat Completions.
    - OpenAI's API has no native video input; we ffmpeg-extract frames and
      pass them as `image_url` (data:image/jpeg;base64,...) blocks.
    - Frame count capped at OPENAI_MAX_FRAMES via uniform sampling so long
      videos at high fps don't blow up context / cost.
    - Requires OPENAI_API_KEY + local video at <video-dir>/video{qid}.mp4.

  * Anthropic Claude (`claude-*`): Messages API.
    - Same frame-extraction primitive as OpenAI, but Anthropic uses an `image`
      content block with `source: {type: base64, media_type, data}` (not data URI).
    - Requires ANTHROPIC_API_KEY + local video.

  * Image-only vLLM-served MLLM (`kimi-vl*`): vLLM Chat Completions with
    multi-image input. Same frame-extraction primitive as OpenAI proper, but
    POSTed to the local vLLM endpoint. Used because Kimi-VL processes video
    as a frame sequence (image list) rather than a video block — verified
    against VLMEvalKit / lmms-eval which sample frames at the dataset layer.
    - Requires vLLM launched with `--allowed-local-media-path <video-dir>`.

  * Qwen / other MLLM family: vLLM OpenAI-compatible chat completion with a
    `video_url` content block (file:// only — vLLM rejects gs://).
    - Pre-flight checks the served model is multimodal via HF AutoConfig.
    - Requires vLLM launched with `--allowed-local-media-path <video-dir>`.

All backends use ThreadPoolExecutor — each call is sync and constructs its
own client.

Args used: --model, --max-concurrent, --tag, --questions, --temperature, --fps,
           --thinking-budget (Gemini only), --video-dir
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Per-Q output writer — every backend writes the same two files
# ---------------------------------------------------------------------------

def _save_outputs(output_dir: Path, qid: int, result: dict) -> None:
    """Write the universal per-Q files:
      - progress/question{qid}.json — full result dict (resume marker + tokens).
      - trajectories/question{qid}.txt — canonical judge-input string
        (singleturn = the model's raw response_text).

    Both inference frameworks (singleturn, ours) write to these same paths so
    the evaluator reads `inference/trajectories/question{qid}.txt` without
    branching. See evaluation/judge.py:read_prediction.
    """
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    with open(progress_dir / f"question{qid}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    with open(traj_dir / f"question{qid}.txt", "w", encoding="utf-8") as f:
        f.write(result.get("response_text") or "")


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def run(items, args, output_dir: Path):
    """Pick a backend by model family and run."""
    m = args.model.lower()
    # gemini-3* must come before the generic gemini* check.
    if m.startswith("gemini-3"):
        return _run_gemini_3(items, args, output_dir)
    if m.startswith("gemini"):
        return _run_gemini(items, args, output_dir)
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return _run_openai(items, args, output_dir)
    if m.startswith("claude"):
        return _run_anthropic(items, args, output_dir)
    if "kimi-vl" in m or "kimi_vl" in m:
        return _run_vllm_image_mllm(items, args, output_dir)
    return _run_vllm_mllm(items, args, output_dir)


# ===========================================================================
# Gemini backend
# ===========================================================================

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    MediaResolution,
    Part,
    ThinkingConfig,
    VideoMetadata,
)

PROMPT_TEMPLATE = """\
You are given a video and a multiple-choice question about it.
Watch the video carefully, then answer the question.

[QUESTION]
{question}

[INSTRUCTIONS]
Think step by step. Write your analysis FIRST, then give your final answer.

1. Describe what you observe in the video that is relevant to the question.
   Include specific visual details, timestamps, or notable moments.
2. For each option, briefly assess whether it matches your observations.
3. Explain your reasoning for why you chose your answer and ruled out alternatives.
4. Finally, state your final answer as a single letter on a new line in this exact format:
   FINAL ANSWER: <letter>

Be thorough in your analysis but concise in your writing."""

# fps fallback chain: on 500 INTERNAL (likely context-size issue), drop to next
FPS_FALLBACK_CHAIN = [10, 5, 3, 1]


def extract_answer_letter(response: str) -> str:
    if not response:
        return "?"
    # LAST-match: thinking models may emit exploratory phrasings like
    # "FINAL ANSWER: A" mid-reasoning before committing to the real answer
    # ("FINAL ANSWER: C") at the end. Taking the first match risks the wrong one.
    matches = re.findall(r'FINAL\s*ANSWER\s*:\s*([A-Z])', response, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    # Local-context fallback: only inspect last 3 lines (near the end of generation)
    # to avoid false-positive letters embedded deep in thinking text.
    last_lines = response.strip().split('\n')[-3:]
    for line in reversed(last_lines):
        m = re.search(r'(?:answer\s*(?:is|:)\s*)?([A-Z])\)?(?:\s*$|\s*[).])', line, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r'\b([B-HJ-Z])\b', line)
        if m:
            return m.group(1)
    # No reliable signal — return "?" rather than gambling on whole-response
    # last-letter heuristic, which produced random false positives on long
    # thinking-mode outputs (especially Qwen3.5).
    return "?"


# Files API upload cache for Dev API mode. Inline base64 caps the request body
# at 1GB on generativelanguage.googleapis.com (8 of our 100 videos are >750MB
# on disk → >1GB after base64 → 400 INVALID_ARGUMENT). Files API uploads the
# bytes once, then we reference the file by URI like gs:// would in Vertex —
# no body cap. Cache is per-process; entries persist for the 48h Files API TTL.
import threading
_FILES_API_CACHE: dict[str, str] = {}   # absolute local path → file URI
_FILES_API_LOCK = threading.Lock()


def _ensure_uploaded_dev_api(client, local_path: str, mime_type: str = "video/mp4") -> str:
    """Upload local_path via the Dev API Files API once, return its URI.

    Idempotent across concurrent ThreadPoolExecutor workers (same lock).
    Polls until state=ACTIVE (videos are processed async after upload).
    Only used in Dev API mode; Vertex mode keeps the gs:// URI path.
    """
    abs_path = os.path.abspath(local_path)
    with _FILES_API_LOCK:
        if abs_path in _FILES_API_CACHE:
            return _FILES_API_CACHE[abs_path]
        f = client.files.upload(file=abs_path, config={"mime_type": mime_type})
        # Files API processes video uploads async; need ACTIVE before generateContent
        deadline = time.time() + 600  # 10min max for huge files (Q52 = 1.3GB)
        while getattr(f.state, "name", str(f.state)) == "PROCESSING":
            if time.time() > deadline:
                raise RuntimeError(f"Files API upload still PROCESSING after 600s for {abs_path}")
            time.sleep(3)
            f = client.files.get(name=f.name)
        state = getattr(f.state, "name", str(f.state))
        if state != "ACTIVE":
            raise RuntimeError(f"Files API upload not ACTIVE: state={state} for {abs_path}")
        _FILES_API_CACHE[abs_path] = f.uri
        return f.uri


def _attempt_single_call(
    client, model_name, video_source, prompt_text, fps,
    retry_stats=None, temperature=0.0, thinking_budget=None,
):
    """Single API call at given fps. Raises on 500 INTERNAL for caller to fps-fallback.
    Quick transient retry (3 attempts, exp backoff capped at 60s) for 429/503/etc.
    """
    use_dev_api = not bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    if video_source and video_source.startswith("gs://"):
        video_part = Part.from_uri(file_uri=video_source, mime_type="video/mp4")
    elif video_source and os.path.exists(video_source):
        if use_dev_api:
            # Dev API: upload once via Files API, reference by URI. Avoids
            # the 1GB request-body cap that inline base64 hits on >750MB videos.
            file_uri = _ensure_uploaded_dev_api(client, video_source)
            video_part = Part.from_uri(file_uri=file_uri, mime_type="video/mp4")
        else:
            # Vertex inline (rare; only when no gs:// available for the item).
            with open(video_source, "rb") as f:
                video_bytes = f.read()
            video_part = Part.from_bytes(data=video_bytes, mime_type="video/mp4")
    else:
        raise RuntimeError(f"Video source not accessible: {video_source}")

    video_part.video_metadata = VideoMetadata(fps=fps)

    config_kwargs = {
        "temperature": temperature,
        "media_resolution": MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        "response_modalities": ["text"],
    }
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=True,
        )
    config = GenerateContentConfig(**config_kwargs)

    for attempt in range(3):
        try:
            return client.models.generate_content(
                model=model_name, contents=[video_part, prompt_text], config=config,
            )
        except Exception as api_err:
            err_str = str(api_err).lower()
            if "500" in err_str or "internal" in err_str:
                raise  # let caller fallback fps
            is_transient = any(kw in err_str for kw in [
                "429", "503", "rate limit", "resource_exhausted", "unavailable",
            ])
            if is_transient and attempt < 2:
                wait = min(2 ** attempt * 5, 60)
                time.sleep(wait)
                if retry_stats is not None:
                    retry_stats["retries"] = retry_stats.get("retries", 0) + 1
                    retry_stats["retry_wait_seconds"] = retry_stats.get("retry_wait_seconds", 0.0) + wait
            else:
                raise
    raise RuntimeError("exhausted transient retries")


def _resolve_video_source(item, video_dir: Path):
    """Pick the right video reference for the active genai client mode.

    - Vertex (GOOGLE_CLOUD_PROJECT set): lazy-upload the local HF dataset
      video to AVB_GCS_BUCKET (first call only) and return the gs:// URI.
      avp/ours follow the same convention to avoid the ~20MB inline cap.
    - Dev API key (no GOOGLE_CLOUD_PROJECT): gs:// URIs are not accepted by
      generativelanguage.googleapis.com (returns 400 INVALID_ARGUMENT), so
      we use the local file via Part.from_bytes inline. The dev-api-key
      auth path also avoids GCS dependency entirely.
    """
    local_path = Path(item.get("video_local_path") or (video_dir / f"video{item['qid']}.mp4"))

    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        bucket = os.environ.get("AVB_GCS_BUCKET", "").strip()
        if bucket and local_path.exists():
            from framework._shared.gcs import ensure_video_uploaded
            prefix = os.environ.get("AVB_GCS_PREFIX", "agentvidbench/videos/eval").strip()
            return ensure_video_uploaded(local_path, bucket, prefix)
    if local_path.exists():
        return str(local_path.resolve())
    return None


def process_item(item, args, output_dir: Path):
    qid = item["question_id"]
    t0 = time.time()

    initial_fps = args.fps
    fps_chain = [f for f in FPS_FALLBACK_CHAIN if f <= initial_fps]
    if not fps_chain or fps_chain[0] != initial_fps:
        fps_chain = [initial_fps] + [f for f in FPS_FALLBACK_CHAIN if f < initial_fps]

    prompt_text = PROMPT_TEMPLATE.format(question=item["prompt_text"])
    video_source = _resolve_video_source(item, Path(args.video_dir))

    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    else:
        client = genai.Client()  # uses GOOGLE_API_KEY

    response = None
    fps_used = None
    last_err = None
    fallback_history = []
    retry_stats = {"retries": 0, "retry_wait_seconds": 0.0}

    for fps in fps_chain:
        print(f"  q{qid}: trying fps={fps}", flush=True)
        try:
            response = _attempt_single_call(
                client, args.model, video_source, prompt_text, fps,
                retry_stats=retry_stats,
                temperature=args.temperature,
                thinking_budget=args.thinking_budget,
            )
            fps_used = fps
            break
        except Exception as e:
            err_str = str(e).lower()
            last_err = e
            fallback_history.append({"fps": fps, "error": str(e)[:200]})
            if "500" in err_str or "internal" in err_str:
                print(f"  q{qid}: fps={fps} hit 500 INTERNAL -> fallback", flush=True)
                continue
            else:
                print(f"  q{qid}: fps={fps} error: {str(e)[:100]}", flush=True)
                break

    elapsed = time.time() - t0

    if response is not None:
        full_text = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    full_text += part.text
        tokens = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            tokens = {
                "prompt_tokens":     int(getattr(um, "prompt_token_count", 0) or 0),
                "candidates_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
                "thoughts_tokens":   int(getattr(um, "thoughts_token_count", 0) or 0),
                "tool_use_tokens":   int(getattr(um, "tool_use_prompt_token_count", 0) or 0),
                "cached_tokens":     int(getattr(um, "cached_content_token_count", 0) or 0),
                "total_tokens":      int(getattr(um, "total_token_count", 0) or 0),
                "calls":             1,
                "retries":           retry_stats.get("retries", 0),
                "retry_wait_seconds": round(retry_stats.get("retry_wait_seconds", 0.0), 2),
            }
        # Extract a quick preview letter for the console only — not stored.
        # Evaluation re-parses response_text against the dataset downstream.
        preview = extract_answer_letter(full_text)
        print(f"  [done] q{qid}: pred={preview} (fps={fps_used}, {elapsed:.1f}s, "
              f"{tokens.get('prompt_tokens', 0)} in / {tokens.get('candidates_tokens', 0)} out)",
              flush=True)

        result = {
            "question_id": qid,
            "video_path": item["video_path"],
            "elapsed_seconds": round(elapsed, 1),
            "fps_used": fps_used,
            "fallback_history": fallback_history,
            "response_text": full_text,
            "tokens": tokens,
        }
    else:
        err_msg = str(last_err) if last_err else "unknown"
        print(f"  [err] q{qid}: {err_msg[:120]}", flush=True)
        result = {
            "question_id": qid, "video_path": item["video_path"],
            "elapsed_seconds": round(elapsed, 1),
            "fps_used": None, "fallback_history": fallback_history,
            "response_text": "", "tokens": {}, "error": err_msg,
        }

    _save_outputs(output_dir, qid, result)
    return result


def _run_gemini(items, args, output_dir: Path):
    """ThreadPoolExecutor over items, Gemini single-call backend."""
    results = []
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as ex:
        futures = {ex.submit(process_item, it, args, output_dir): it for it in items}
        for fut in as_completed(futures):
            results.append(fut.result())

    from collections import Counter
    fps_dist = Counter(r.get("fps_used") for r in results)
    print(f"FPS used: {dict(sorted(fps_dist.items(), key=lambda x: -(x[0] or 0)))}")

    return results


def _run_gemini_3(items, args, output_dir: Path):
    """Gemini 3.0 (preview) backend. Currently delegates to _run_gemini —
    the genai SDK exposes the same surface for 2.x and 3.x. Separated so
    preview-specific behavior (different fps fallback chain, ThinkingConfig
    schema changes, region defaults, etc.) can be added here without
    touching the gemini-2.x path."""
    return _run_gemini(items, args, output_dir)


# ===========================================================================
# OpenAI proper backend (GPT-5, GPT-4o, o1/o3/o4) — frame extraction
# ===========================================================================

# OpenAI Chat Completions has no native video input. Sample frames at
# args.fps via ffmpeg, cap to OPENAI_MAX_FRAMES via uniform sampling, encode
# each as base64 JPEG and pass as `image_url` content blocks.
#
# Cap rationale: at fps=1 a 1200s video produces 1200 frames — way beyond
# practical context and cost. 50 frames × 85 tokens (detail=low) ≈ 4.25 K
# vision tokens per question, comparable to Gemini's medium-resolution
# video token count.
OPENAI_MAX_FRAMES = 50
OPENAI_MAX_TOKENS = 4096
# Reasoning-only models (GPT-5 family, o-series) consume the completion budget
# for internal reasoning before producing visible text. Empirically 4096 is
# entirely consumed by thinking on multi-image inputs (Q1: 4096/4096 used,
# response empty), so we raise the cap for these models. Headroom = 12K extra.
OPENAI_REASONING_MAX_TOKENS = 16384


def _is_reasoning_only_openai(model: str) -> bool:
    """Reasoning-only OpenAI models reject `temperature` at the API level
    AND require `max_completion_tokens` instead of `max_tokens`. Covers:
      - o-series (o1, o3, o4, ...)
      - GPT-5 family (gpt-5, gpt-5-mini, gpt-5-pro, gpt-5-codex, gpt-5.1, ...)
    Verified against OpenAI API 2026-04-30: gpt-5* returns 400
    "Unsupported parameter: 'max_tokens' is not supported with this model".
    """
    m = (model or "").lower()
    return m.startswith(("o1", "o3", "o4")) or m.startswith("gpt-5")


def _resolve_ffmpeg() -> str:
    """Locate ffmpeg via PATH; fall back to bare 'ffmpeg' so the subprocess
    error surfaces a recognizable message if it's missing."""
    import shutil
    return shutil.which("ffmpeg") or "ffmpeg"


def _extract_frames_b64(video_path: Path, fps: float, max_frames: int = OPENAI_MAX_FRAMES,
                         max_dim: int | None = None, jpeg_q: int = 2) -> list[str]:
    """ffmpeg-sample frames at `fps`, uniform-downsample to ≤ max_frames,
    return list of base64-encoded JPEGs.

    Optional `max_dim` clamps the longest side via ffmpeg's scale filter
    (force_original_aspect_ratio=decrease only ever shrinks). Default None
    leaves source resolution untouched — same behaviour as before for every
    caller that doesn't pass it (OpenAI, Kimi-VL, generic vLLM). Used by the
    Anthropic dispatch as a reactive shrink on HTTP 413 / 400-dim errors.
    `jpeg_q` is ffmpeg's `-q:v` (lower = higher quality, larger files)."""
    import base64
    import subprocess
    import tempfile

    if max_dim is not None:
        vf = (
            f"fps={fps},"
            f"scale='min({max_dim},iw)':'min({max_dim},ih)':force_original_aspect_ratio=decrease"
        )
    else:
        vf = f"fps={fps}"

    with tempfile.TemporaryDirectory(prefix="ovis_") as tmp:
        out_pattern = os.path.join(tmp, "f_%05d.jpg")
        cmd = [
            _resolve_ffmpeg(), "-loglevel", "error", "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-q:v", str(jpeg_q),
            out_pattern,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        files = sorted(f for f in os.listdir(tmp) if f.endswith(".jpg"))
        if not files:
            raise RuntimeError(f"ffmpeg produced no frames for {video_path}")
        if len(files) > max_frames:
            step = len(files) / max_frames
            files = [files[int(i * step)] for i in range(max_frames)]
        out = []
        for fname in files:
            with open(os.path.join(tmp, fname), "rb") as fh:
                out.append(base64.b64encode(fh.read()).decode("ascii"))
        return out


def _process_item_openai(item, client, args, output_dir: Path):
    qid = item["question_id"]
    t0 = time.time()
    prompt_text = PROMPT_TEMPLATE.format(question=item["prompt_text"])

    local_path = Path(item.get("video_local_path") or (Path(args.video_dir) / f"video{qid}.mp4"))
    if not local_path.exists():
        # OpenAI proper can't take gs:// URIs and inline upload by URL is
        # 20MB-limited via SDK helpers — require a local file. (gsutil cp
        # is the user's responsibility, mirroring the qwen singleturn flow.)
        return _record_skip_openai(qid, item, args, output_dir,
                                   f"missing local video: {local_path}")

    try:
        frames_b64 = _extract_frames_b64(local_path, float(args.fps),
                                         max_frames=args.max_frames)
    except Exception as e:
        return _record_skip_openai(qid, item, args, output_dir,
                                   f"frame extraction failed: {e}")

    # Image blocks first, then prompt text last (closer to question = better attention).
    content: list = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low",
            },
        })
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    response_text = ""
    error = None
    usage = None
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            kwargs: dict = {
                "model": args.model,
                "messages": messages,
            }
            if _is_reasoning_only_openai(args.model):
                # GPT-5 / o-series: max_completion_tokens (with reasoning headroom),
                # no temperature.
                kwargs["max_completion_tokens"] = OPENAI_REASONING_MAX_TOKENS
            else:
                kwargs["max_tokens"] = OPENAI_MAX_TOKENS
                kwargs["temperature"] = args.temperature
            resp = client.chat.completions.create(**kwargs)
            response_text = resp.choices[0].message.content or ""
            usage = resp.usage
            break
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            transient = any(kw in err_str for kw in [
                "429", "rate limit", "rate_limit", "503", "internal",
                "unavailable", "timeout", "connection",
            ])
            if transient and attempt < 4:
                wait = min(2 ** attempt * 5, 60)
                time.sleep(wait)
                continue
            error = str(e)[:300]
            break

    elapsed = time.time() - t0
    preview = extract_answer_letter(response_text) if response_text else "?"
    status = "done" if response_text else "err"

    tokens: dict = {}
    if usage is not None:
        # OpenAI usage fields: prompt_tokens / completion_tokens / total_tokens.
        # Map onto our shared schema (candidates_tokens = completion_tokens).
        tokens = {
            "prompt_tokens":     int(getattr(usage, "prompt_tokens", 0) or 0),
            "candidates_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens":      int(getattr(usage, "total_tokens", 0) or 0),
            "calls":             1,
            "frames_sampled":    len(frames_b64),
        }
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            tokens["cached_tokens"] = int(getattr(details, "cached_tokens", 0) or 0)

    print(f"  [{status}] q{qid}: pred={preview} "
          f"(fps={args.fps}, frames={len(frames_b64)}, {elapsed:.1f}s, "
          f"{tokens.get('prompt_tokens', 0)} in / {tokens.get('candidates_tokens', 0)} out)",
          flush=True)

    # Effective fps = frames_sampled / video_duration. With ffmpeg sampling at
    # `args.fps` then capping to `args.max_frames`, the model effectively sees
    # this many frames per second of original footage. Useful for cross-cap
    # comparison (e.g. 50-frame cap on 240s video → eff 0.21 fps).
    dur = float(item.get("duration") or 0.0)
    eff_fps = round(len(frames_b64) / dur, 4) if dur > 0 else None

    result = {
        "question_id": qid, "video_path": item["video_path"],
        "elapsed_seconds": round(elapsed, 1), "fps_used": args.fps,
        "frames_sampled": len(frames_b64),
        "max_frames_cap": args.max_frames,
        "video_duration": dur if dur > 0 else None,
        "effective_fps": eff_fps,
        "response_text": response_text, "tokens": tokens,
    }
    if error:
        result["error"] = error

    _save_outputs(output_dir, qid, result)
    return result


def _record_skip_openai(qid, item, args, output_dir: Path, reason: str):
    print(f"  [skip] q{qid}: {reason}", file=sys.stderr)
    dur = float(item.get("duration") or 0.0)
    result = {
        "question_id": qid, "video_path": item["video_path"],
        "elapsed_seconds": 0.0, "fps_used": args.fps, "frames_sampled": 0,
        "max_frames_cap": args.max_frames,
        "video_duration": dur if dur > 0 else None,
        "effective_fps": None,
        "response_text": "", "tokens": {}, "error": reason,
    }
    _save_outputs(output_dir, qid, result)
    return result


def _run_openai(items, args, output_dir: Path):
    """OpenAI Chat Completions backend with ffmpeg frame extraction."""
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        print("=" * 70, file=sys.stderr)
        print("ERROR: OPENAI_API_KEY env var not set.", file=sys.stderr)
        print(f"Required for singleturn --model '{args.model}'.", file=sys.stderr)
        print("  export OPENAI_API_KEY=sk-...", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(2)

    client = OpenAI()  # picks up OPENAI_API_KEY automatically
    results = []
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as ex:
        futures = {ex.submit(_process_item_openai, it, client, args, output_dir): it
                   for it in items}
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


# ===========================================================================
# Anthropic Claude backend (frame extraction + Messages API)
# ===========================================================================

# Claude has no native video input. Same primitive as OpenAI (ffmpeg sample +
# uniform downsample to args.max_frames + base64 JPEG), but content blocks use
# Anthropic's "image" type with a `source` object instead of a data URI.

ANTHROPIC_MAX_TOKENS = 4096
ANTHROPIC_API_VERSION = "2023-06-01"


def _process_item_anthropic(item, args, output_dir: Path):
    qid = item["question_id"]
    t0 = time.time()
    prompt_text = PROMPT_TEMPLATE.format(question=item["prompt_text"])

    local_path = Path(item.get("video_local_path") or (Path(args.video_dir) / f"video{qid}.mp4"))
    if not local_path.exists():
        return _record_skip_anthropic(qid, item, args, output_dir,
                                      f"missing local video: {local_path}")

    try:
        frames_b64 = _extract_frames_b64(local_path, float(args.fps),
                                         max_frames=args.max_frames)
    except Exception as e:
        return _record_skip_anthropic(qid, item, args, output_dir,
                                      f"frame extraction failed: {e}")

    def _build_body(frames_list):
        c: list = []
        for b in frames_list:
            c.append({"type": "image",
                      "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
        c.append({"type": "text", "text": prompt_text})
        return json.dumps({
            "model": args.model,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "messages": [{"role": "user", "content": c}],
        }).encode()

    body = _build_body(frames_b64)

    import urllib.request as ur
    response_text = ""
    error = None
    usage = {}
    # Shrink tiers walked through in order on 413 / 400-dim failures.
    # Tier-1 (1568px / q3) covers most cases. Tier-2 (1024px / q5) recovers
    # high-fps × huge-video edges (e.g. 100 frames × 1080p source still exceeds
    # 32MB body cap after tier-1).
    SHRINK_TIERS = [(1568, 3), (1024, 5)]
    shrink_tier = 0
    resized = False         # True after any tier shrink
    max_dim_used: Optional[int] = None
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            req = ur.Request(
                "https://api.anthropic.com/v1/messages",
                data=body,
                headers={
                    "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                    "anthropic-version": ANTHROPIC_API_VERSION,
                    "Content-Type": "application/json",
                },
            )
            with ur.urlopen(req, timeout=900) as resp:
                d = json.loads(resp.read())
            response_text = "".join(b.get("text", "")
                                    for b in d.get("content", [])
                                    if b.get("type") == "text")
            usage = d.get("usage", {}) or {}
            break
        except ur.HTTPError as e:
            last_err = e
            err_body = e.read().decode()
            # Reactive shrink: walk through SHRINK_TIERS on body-size / dim
            # failures. First attempt always uses source resolution; most Q's
            # never hit this branch.
            is_size_err = (
                e.code == 413
                or (e.code == 400 and (
                    "max allowed size" in err_body
                    or "exceeds the maximum size" in err_body
                ))
            )
            if is_size_err and shrink_tier < len(SHRINK_TIERS):
                next_dim, next_q = SHRINK_TIERS[shrink_tier]
                try:
                    frames_b64 = _extract_frames_b64(
                        local_path, float(args.fps),
                        max_frames=args.max_frames,
                        max_dim=next_dim, jpeg_q=next_q,
                    )
                    body = _build_body(frames_b64)
                    shrink_tier += 1
                    resized = True
                    max_dim_used = next_dim
                    print(f"  [RETRY] q{qid}: HTTP {e.code} → tier-{shrink_tier} "
                          f"resize to {next_dim}px q{next_q} and retry",
                          flush=True)
                    continue
                except Exception as re_err:
                    error = f"HTTP {e.code}; resize failed: {re_err}"
                    break
            transient = e.code in (429, 500, 502, 503, 504)
            if transient and attempt < 4:
                wait = min(2 ** attempt * 5, 60)
                time.sleep(wait)
                continue
            error = f"HTTP {e.code}: {err_body[:300]}"
            break
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if any(kw in err_str for kw in ("timeout", "connection")) and attempt < 4:
                wait = min(2 ** attempt * 5, 60)
                time.sleep(wait)
                continue
            error = str(e)[:300]
            break

    elapsed = time.time() - t0
    preview = extract_answer_letter(response_text) if response_text else "?"
    status = "done" if response_text else "err"

    tokens: dict = {}
    if usage:
        in_tok  = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        cache_read     = int(usage.get("cache_read_input_tokens", 0) or 0)
        tokens = {
            "prompt_tokens":     in_tok,
            "candidates_tokens": out_tok,
            "total_tokens":      in_tok + out_tok,
            "calls":             1,
            "frames_sampled":    len(frames_b64),
            "cached_tokens":     cache_read,
            "cache_creation_tokens": cache_creation,
        }

    print(f"  [{status}] q{qid}: pred={preview} "
          f"(fps={args.fps}, frames={len(frames_b64)}, {elapsed:.1f}s, "
          f"{tokens.get('prompt_tokens', 0)} in / {tokens.get('candidates_tokens', 0)} out"
          f"{', resized=' + str(max_dim_used) if resized else ''})",
          flush=True)

    dur = float(item.get("duration") or 0.0)
    eff_fps = round(len(frames_b64) / dur, 4) if dur > 0 else None

    result = {
        "question_id": qid, "video_path": item["video_path"],
        "elapsed_seconds": round(elapsed, 1), "fps_used": args.fps,
        "frames_sampled": len(frames_b64),
        "max_frames_cap": args.max_frames,
        "video_duration": dur if dur > 0 else None,
        "effective_fps": eff_fps,
        "resized": resized,
        "max_dim_used": max_dim_used,
        "response_text": response_text, "tokens": tokens,
    }
    if error:
        result["error"] = error

    _save_outputs(output_dir, qid, result)
    return result


def _record_skip_anthropic(qid, item, args, output_dir: Path, reason: str):
    print(f"  [skip] q{qid}: {reason}", file=sys.stderr)
    dur = float(item.get("duration") or 0.0)
    result = {
        "question_id": qid, "video_path": item["video_path"],
        "elapsed_seconds": 0.0, "fps_used": args.fps, "frames_sampled": 0,
        "max_frames_cap": args.max_frames,
        "video_duration": dur if dur > 0 else None,
        "effective_fps": None,
        "response_text": "", "tokens": {}, "error": reason,
    }
    _save_outputs(output_dir, qid, result)
    return result


def _run_anthropic(items, args, output_dir: Path):
    """Anthropic Claude backend with ffmpeg frame extraction."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("=" * 70, file=sys.stderr)
        print("ERROR: ANTHROPIC_API_KEY env var not set.", file=sys.stderr)
        print(f"Required for singleturn --model '{args.model}'.", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(2)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as ex:
        futures = {ex.submit(_process_item_anthropic, it, args, output_dir): it
                   for it in items}
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


# ===========================================================================
# vLLM-served IMAGE-ONLY MLLM backend (Kimi-VL etc.) — frame extraction
# ===========================================================================

# Kimi-VL processes video as a sequence of images (no video token in chat
# template). Reuse the OpenAI proper frame-extraction primitive but POST to
# the local vLLM endpoint instead of api.openai.com. Frame budget = 50 to
# match OPENAI_MAX_FRAMES (fair comparison vs GPT-5 OpenAI-proper run).


def _process_item_vllm_image(item, client, args, output_dir: Path):
    qid = item["question_id"]
    t0 = time.time()
    prompt_text = PROMPT_TEMPLATE.format(question=item["prompt_text"])

    local_path = Path(item.get("video_local_path") or (Path(args.video_dir) / f"video{qid}.mp4"))
    if not local_path.exists():
        return _record_skip_openai(qid, item, args, output_dir,
                                   f"missing local video: {local_path}")

    try:
        frames_b64 = _extract_frames_b64(local_path, float(args.fps),
                                         max_frames=args.max_frames)
    except Exception as e:
        return _record_skip_openai(qid, item, args, output_dir,
                                   f"frame extraction failed: {e}")

    content: list = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low",
            },
        })
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    response_text = ""
    error = None
    usage = None
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=VLLM_MAX_TOKENS,
                temperature=args.temperature,
            )
            response_text = resp.choices[0].message.content or ""
            usage = resp.usage
            break
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            transient = any(kw in err_str for kw in [
                "429", "rate limit", "rate_limit", "503", "internal",
                "unavailable", "timeout", "connection",
            ])
            if transient and attempt < 4:
                wait = min(2 ** attempt * 5, 60)
                time.sleep(wait)
                continue
            error = str(e)[:300]
            break

    elapsed = time.time() - t0
    preview = extract_answer_letter(response_text) if response_text else "?"
    status = "done" if response_text else "err"

    tokens: dict = {}
    if usage is not None:
        tokens = {
            "prompt_tokens":     int(getattr(usage, "prompt_tokens", 0) or 0),
            "candidates_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens":      int(getattr(usage, "total_tokens", 0) or 0),
            "calls":             1,
            "frames_sampled":    len(frames_b64),
        }

    print(f"  [{status}] q{qid}: pred={preview} "
          f"(fps={args.fps}, frames={len(frames_b64)}, {elapsed:.1f}s, "
          f"{tokens.get('prompt_tokens', 0)} in / {tokens.get('candidates_tokens', 0)} out)",
          flush=True)

    dur = float(item.get("duration") or 0.0)
    eff_fps = round(len(frames_b64) / dur, 4) if dur > 0 else None

    result = {
        "question_id": qid, "video_path": item["video_path"],
        "elapsed_seconds": round(elapsed, 1), "fps_used": args.fps,
        "frames_sampled": len(frames_b64),
        "max_frames_cap": args.max_frames,
        "video_duration": dur if dur > 0 else None,
        "effective_fps": eff_fps,
        "response_text": response_text, "tokens": tokens,
    }
    if error:
        result["error"] = error

    _save_outputs(output_dir, qid, result)
    return result


def _run_vllm_image_mllm(items, args, output_dir: Path):
    """Image-only vLLM-served MLLM (Kimi-VL etc.) — frames as image_url list."""
    from openai import OpenAI

    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    client = OpenAI(base_url=base_url, api_key=api_key)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as ex:
        futures = {ex.submit(_process_item_vllm_image, it, client, args, output_dir): it
                   for it in items}
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


# ===========================================================================
# vLLM-served MLLM backend (Qwen-VL etc.)
# ===========================================================================

VLLM_MAX_TOKENS = 16384


def _assert_multimodal(model_id: str) -> None:
    """SystemExit if served model is text-only. Optimistic if HF config unreachable."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"[singleturn vllm] WARN: failed to load HF config for '{model_id}': {e}",
              file=sys.stderr)
        print("[singleturn vllm] cannot verify multimodal; proceeding optimistically.",
              file=sys.stderr)
        return

    has_image_token = getattr(cfg, "image_token_id", None) is not None
    has_vision_cfg = bool(getattr(cfg, "vision_config", None))
    archs = getattr(cfg, "architectures", None) or []
    arch_str = " ".join(archs).lower()
    has_mm_arch = any(kw in arch_str for kw in ("conditionalgeneration", "vision", "vl"))

    if not (has_image_token or has_vision_cfg or has_mm_arch):
        print("=" * 70, file=sys.stderr)
        print(f"ERROR: model '{model_id}' appears to be TEXT-ONLY.", file=sys.stderr)
        print("Zero-shot evaluation requires a multimodal (video-input) model.", file=sys.stderr)
        print(f"  architectures: {archs}", file=sys.stderr)
        print("Use a Qwen MLLM (e.g. Qwen/Qwen3-VL-4B-Instruct), or use", file=sys.stderr)
        print("METHOD=ours in run_qwen.sh to run text-only Qwen as a planner.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(2)


def _process_item_vllm(item, client, args, output_dir: Path):
    qid = item["question_id"]
    t0 = time.time()
    prompt_text = PROMPT_TEMPLATE.format(question=item["prompt_text"])

    local_path = Path(item.get("video_local_path") or (Path(args.video_dir) / f"video{qid}.mp4"))
    if not local_path.exists():
        print(f"  [skip] q{qid}: local video missing at {local_path}", file=sys.stderr)
        result = {
            "question_id": qid, "video_path": item["video_path"],
            "elapsed_seconds": 0.0, "fps_used": args.fps,
            "response_text": "", "tokens": {}, "error": f"missing video file: {local_path}",
        }
        _save_outputs(output_dir, qid, result)
        return result

    messages = [{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": f"file://{local_path.resolve()}"}},
            {"type": "text", "text": prompt_text},
        ],
    }]
    extra_body = {"mm_processor_kwargs": {"fps": float(args.fps)}}
    # Optional chat_template_kwargs via env var (e.g. enable_thinking=false for
    # Qwen3.5 Instruct mode). JSON dict expected.
    _ct_kwargs_json = os.environ.get("VLLM_CHAT_TEMPLATE_KWARGS_JSON")
    if _ct_kwargs_json:
        try:
            extra_body["chat_template_kwargs"] = json.loads(_ct_kwargs_json)
        except json.JSONDecodeError:
            print(f"  WARN: failed to parse VLLM_CHAT_TEMPLATE_KWARGS_JSON: {_ct_kwargs_json!r}", file=sys.stderr)

    response_text = ""
    error = None
    usage = None
    try:
        resp = client.chat.completions.create(
            model=args.model, messages=messages,
            max_tokens=VLLM_MAX_TOKENS, temperature=args.temperature,
            extra_body=extra_body,
            timeout=7200,
        )
        response_text = resp.choices[0].message.content or ""
        usage = resp.usage
    except Exception as e:
        error = str(e)[:300]

    elapsed = time.time() - t0
    preview = extract_answer_letter(response_text) if response_text else "?"
    status = "done" if response_text else "err"

    tokens = {}
    if usage:
        tokens = {
            "prompt_tokens":     int(getattr(usage, "prompt_tokens", 0) or 0),
            "candidates_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens":      int(getattr(usage, "total_tokens", 0) or 0),
            "calls":             1,
        }

    print(f"  [{status}] q{qid}: pred={preview} "
          f"(fps={args.fps}, {elapsed:.1f}s, "
          f"{tokens.get('prompt_tokens', 0)} in / {tokens.get('candidates_tokens', 0)} out)",
          flush=True)

    result = {
        "question_id": qid, "video_path": item["video_path"],
        "elapsed_seconds": round(elapsed, 1), "fps_used": args.fps,
        "response_text": response_text, "tokens": tokens,
    }
    if error:
        result["error"] = error

    _save_outputs(output_dir, qid, result)
    return result


def _run_vllm_mllm(items, args, output_dir: Path):
    """Qwen-VL / other vLLM-served MLLM via OpenAI-compatible chat completions."""
    from openai import OpenAI

    _assert_multimodal(args.model)

    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    client = OpenAI(base_url=base_url, api_key=api_key)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as ex:
        futures = {ex.submit(_process_item_vllm, it, client, args, output_dir): it
                   for it in items}
        for fut in as_completed(futures):
            results.append(fut.result())
    return results
