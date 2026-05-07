"""Agentic ReAct method (the framework formerly known as agentic_ours).

Per question:
  1. Build a Vertex AI Gemini client (always — used by the analyze_video tool).
  2. Build a planner via build_planner() — Gemini for gemini-*, Qwen for qwen-*,
     OpenAICompatible fallback for everything else (vLLM-served).
  3. Build a tool registry: VideoAnalyzerTool (Vertex Gemini, FIXED) + GetTranscriptTool (offline SRT).
  4. Run ReActOrchestrator.answer() — the ReAct loop (planner.send → tool.execute → ...).
  5. Save trajectory + per-Q result.

asyncio-based — Gemini chat object + tool VLM both use the async API
(`aio.chats.create.send_message`, `aio.models.generate_content`). Concurrency
controlled by an asyncio.Semaphore at args.max_concurrent.

Tool model is FIXED to Vertex AI Gemini regardless of --model. Only the planner
is swappable. See framework/tools/video_analyzer.py for why.

Args used: --model, --max-concurrent, --tag, --questions, --temperature, --fps,
           --thinking-budget, --track-tokens, --verbose
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from google import genai

from framework.methods.ours.react import ReActOrchestrator
from framework.models import GeminiPlanner, OpenAICompatiblePlanner, PlannerLLM
from framework.tools.base import ToolRegistry
from framework.tools.video_analyzer import VideoAnalyzerTool
from framework.tools.transcript_reader import GetTranscriptTool
from framework._shared.token_tracker import (
    TokenAccumulator, wrap_genai_client, set_current_accumulator,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # framework/methods/ours/runner.py → repo root

# Tool model is FIXED to Vertex AI Gemini regardless of --model. The --model
# flag controls only the planner (orchestrator) which decides how to use tools.
# When the planner is swapped to Qwen / GPT / etc., the tool keeps calling
# Gemini for video understanding (the tool's role is not negotiable here).
TOOL_MODEL = "gemini-2.5-pro"


def build_planner(model: str, gemini_client) -> PlannerLLM:
    """Pick a planner backend by --model name.

    Per-model packages (e.g., QwenPlanner, GemmaPlanner) live as
    framework/models/<family>/{__init__.py, planner.py}. To add one:

        1. Create framework/models/<family>/ with planner.py defining
           `class <Family>Planner(OpenAICompatiblePlanner): ...` and
           an __init__.py that re-exports the class.
        2. Add the import + branch BEFORE the OpenAICompatiblePlanner fallback
           below. Branch matches by lowercase model-name prefix or pattern.
        3. The fallback already routes any non-Gemini model to a generic
           OpenAICompatiblePlanner against the local vLLM server, so a
           subclass is only needed when you want model-specific behavior.

    Endpoint defaults (override via env):
      VLLM_BASE_URL  default http://localhost:8000/v1
      VLLM_API_KEY   default "EMPTY"  (vLLM ignores key by default)
    """
    m = (model or "").lower()
    # gemini-3* must come before the generic gemini* check.
    if m.startswith("gemini-3"):
        from framework.models.gemini_3 import Gemini3Planner
        return Gemini3Planner(client=gemini_client, model=model)
    if m.startswith("gemini"):
        return GeminiPlanner(client=gemini_client, model=model)

    # Qwen 3.x family (Instruct / Thinking / Coder, any HF size).
    if m.startswith("qwen"):
        from framework.models.qwen import QwenPlanner
        return QwenPlanner(
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            model=model,
        )

    # Gemma 4 family (E2B / E4B / 26B-A4B / 31B, dense + MoE, all multimodal).
    # Matches both bare model ids ("gemma-4-...") and HF-prefixed ones
    # ("google/gemma-4-...") returned by /v1/models auto-discover.
    if m.startswith("gemma") or "/gemma" in m:
        from framework.models.gemma import GemmaPlanner
        return GemmaPlanner(
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            model=model,
        )

    # Kimi-VL family (Moonshot AI MoE; Kimi-VL-A3B-Thinking-2506 etc.).
    # Matches bare ids ('kimi-vl', 'kimi_vl') and HF-prefixed
    # ('moonshotai/kimi-vl-...') returned by /v1/models auto-discover.
    if (m.startswith(("kimi-vl", "kimi_vl"))
            or "/kimi-vl" in m or "/kimi_vl" in m):
        from framework.models.kimi_vl import KimiVLPlanner
        return KimiVLPlanner(
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            model=model,
        )

    # OpenAI proper (GPT-5, GPT-4o, o1/o3/o4) — uses OPENAI_API_KEY against
    # https://api.openai.com/v1 by default. analyze_video tool stays Gemini.
    if m.startswith(("gpt", "o1", "o3", "o4")):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY env var must be set for --model "
                f"'{model}'. export OPENAI_API_KEY=sk-..."
            )
        from framework.models.openai import OpenAIPlanner
        return OpenAIPlanner(model=model)

    # Anthropic Claude — uses ANTHROPIC_API_KEY against api.anthropic.com.
    # analyze_video tool stays Gemini, so this is a hybrid by construction
    # (Claude plans/replans, Gemini does video understanding via the tool).
    if m.startswith("claude"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY env var must be set for --model "
                f"'{model}'. export ANTHROPIC_API_KEY=sk-ant-..."
            )
        from framework.models.anthropic import AnthropicPlanner
        return AnthropicPlanner(model=model)

    # Generic OpenAI-compatible fallback — works for any vLLM-served model
    # without per-family customization.
    return OpenAICompatiblePlanner(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        model=model,
    )


async def process_one(item, args, output_dir: Path):
    qid = item["question_id"]
    t0 = time.time()

    # Tool client — always Vertex AI Gemini (fixed). Powers analyze_video.
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT not set — required for the `ours` framework "
            "(analyze_video tool runs on Vertex AI Gemini). See README §3."
        )
    tool_client = genai.Client(
        vertexai=True,
        project=project,
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
    # Planner — chosen by --model via the factory.
    planner = build_planner(args.model, gemini_client=tool_client)

    # OURS_RAW_DUMP=1 → forensic dump of every chat.completions.create
    # request+response under {output_dir}/outputs/raw/question{qid}.iter{N}.json.
    # Used to inspect parser bugs (vLLM #39056 / #39468) at the wire level.
    # Only effective on planners that subclass OpenAICompatiblePlanner.
    if os.environ.get("OURS_RAW_DUMP", "").strip().lower() in ("1", "true", "yes", "on"):
        if hasattr(planner, "set_raw_dump_context"):
            planner.set_raw_dump_context(output_dir / "outputs" / "raw", qid)

    tokens_acc = None
    if args.track_tokens:
        tokens_acc = TokenAccumulator()
        wrap_genai_client(tool_client, tokens_acc)              # tool always Gemini
        planner.attach_token_tracker(tokens_acc)                # planner-specific hookup
        # Expose accumulator via contextvar so retry sites in tools/planner
        # can credit backoff sleeps to this Q's accumulator.
        set_current_accumulator(tokens_acc)

    # Lazy GCS upload — load_items() leaves gcs_uri=None; runners that need
    # Vertex AI access (always for `ours` since the analyze_video tool is
    # Gemini-fixed) translate the local path to gs:// here. Cached per-process.
    from framework._shared.gcs import ensure_video_uploaded
    bucket = os.environ.get("AVB_GCS_BUCKET", "").strip()
    prefix = os.environ.get("AVB_GCS_PREFIX", "agentvidbench/videos/eval").strip()
    video_uri = ensure_video_uploaded(Path(item["video_local_path"]), bucket, prefix)

    registry = ToolRegistry()
    registry.register(VideoAnalyzerTool(
        gemini_client=tool_client,
        model_name=TOOL_MODEL,            # FIXED — does not follow --model
        video_uri=video_uri,
        video_duration=item["duration"],
        question_text=item["prompt_text"],
        temperature=args.temperature,
        # Hardcoded to 8192 — value used in past Gemini+Ours runs. Kept fixed
        # so Qwen's CLI --thinking-budget (used as enable_thinking toggle, not
        # a token count) doesn't leak into Vertex Gemini's analyze_video tool,
        # which rejects values outside 128-32768.
        thinking_budget=8192,
        default_fps=args.fps,
    ))
    # Transcript path comes from the dataset row (resolved by load_items).
    # If a custom item builder didn't set it, fall back to the canonical layout.
    transcript_path = item.get("transcript_path") or str(
        REPO_ROOT / "dataset" / "transcripts" / f"video{qid}.srt"
    )
    registry.register(GetTranscriptTool(
        srt_path=transcript_path,
        qid=qid,
    ))

    agent = ReActOrchestrator(
        planner=planner,
        tool_registry=registry,
        # Pass the planner's --model so _default_prompt() can advertise the
        # gemini-3 high-resolution option only when the backbone supports it.
        model=args.model,
        # No breaker by default — tool errors flow back to the planner so it
        # can retry with corrected args. Set max_tool_failures=1 explicitly to
        # abort the question with "?" on the first VLM retry-exhausted failure.
        max_tool_failures=None,
        # Same fixed value (8192) — keeps Ours method consistent with past
        # Gemini+Ours runs across both VLM tool and planner thinking budgets.
        thinking_budget=8192,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    err_msg = None
    try:
        await agent.answer(item["prompt_text"], item["duration"], video_id=qid)
    except Exception as e:
        err_msg = str(e)
        print(f"  [ERR] q{qid}: {err_msg}", flush=True)

    elapsed = time.time() - t0
    if not err_msg:
        print(f"  [done] q{qid}: {elapsed:.1f}s", flush=True)

    outputs_dir = output_dir / "outputs"          # method-specific intermediates
    traj_dir = output_dir / "trajectories"        # canonical judge-input text
    outputs_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    if agent.trajectory:
        # Legacy native pair (compat): outputs/question{qid}.{json,txt}
        agent.trajectory.save(str(outputs_dir))
        # Normalized JSON: outputs/question{qid}.normalized.json
        nt = agent.trajectory.save_normalized(
            str(outputs_dir),
            qid=qid,
            model=args.model,
            question_text=item.get("question_text") or item["prompt_text"],
            elapsed_seconds=round(elapsed, 3),
            tokens=(tokens_acc.snapshot() if tokens_acc is not None else None),
        )
        # Canonical judge-input text — every framework writes one of these to
        # inference/trajectories/question{qid}.txt. Evaluator reads this path
        # without branching on framework.
        agent.trajectory.save_judge_text(str(traj_dir), nt)

    result = {
        "question_id": qid,
        "video_path": item["video_path"],   # HF-relative path
        "gcs_uri": video_uri,
        "elapsed_seconds": round(elapsed, 1),
    }
    if err_msg is not None:
        result["error"] = err_msg
    if tokens_acc is not None:
        result["tokens"] = tokens_acc.snapshot()

    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    with open(progress_dir / f"question{qid}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


async def _run_async(items, args, output_dir: Path):
    sem = asyncio.Semaphore(args.max_concurrent)

    async def _wrap(item):
        async with sem:
            return await process_one(item, args, output_dir)

    return await asyncio.gather(*[_wrap(it) for it in items])


def run(items, args, output_dir: Path):
    """asyncio.Semaphore + asyncio.gather over items."""
    return asyncio.run(_run_async(items, args, output_dir))
