"""Per-method execution packages.

Each sub-package (singleturn/, ours/) exports a top-level `run`:
    def run(items: list[dict], args: argparse.Namespace, output_dir: Path) -> list[dict]:
        ...

`inference.py` at the repo root does
    importlib.import_module(f"framework.methods.{framework}")
then calls .run(...) and aggregates returned results into summary.json.

Currently registered:
  - singleturn: one model call per question (Gemini / OpenAI / Anthropic /
                Kimi-VL / generic vLLM, dispatched by model-name prefix)
  - ours:       ReAct agent — swappable planner + VideoAnalyzerTool (always
                Vertex Gemini) + GetTranscriptTool (offline SRT)
"""
