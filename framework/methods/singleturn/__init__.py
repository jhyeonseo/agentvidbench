"""Single-turn method package — one API call per question.

Re-exports `run` so the orchestrator can `import framework.methods.singleturn` and
call `singleturn.run(items, args, output_dir)`.
"""
from .runner import run

__all__ = ["run"]
