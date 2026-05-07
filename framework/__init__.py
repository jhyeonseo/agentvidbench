"""Top-level eval package.

Entry point: `inference.py` at the repo root.

Sub-packages:
  - framework.orchestration: shared library used by inference.py
  - framework._shared: cross-method shared abstractions (PlannerLLM, BaseTool,
    ReActOrchestrator, token_tracker)
  - framework.methods: per-method execution modules (singleturn, ours)
"""
