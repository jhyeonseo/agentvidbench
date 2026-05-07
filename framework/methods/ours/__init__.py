"""Agentic ReAct method package.

Owns the ReAct loop (react.py + trajectory.py). Tools are top-level
(`framework/tools/`) so other methods can reuse the same implementations
for fair comparison. Planner backends are also top-level
(`framework/models/`). Cross-method utilities (token_tracker) live in
`framework/_shared/`.

Re-exports `run` so the orchestrator can `import framework.methods.ours` and
call `ours.run(items, args, output_dir)`.
"""
from .runner import run

__all__ = ["run"]
