"""Planner LLM backends.

Each model family lives in its own sub-package (one folder per family),
with `planner.py` holding the implementation. All implement `PlannerLLM`
from `base.py`. The orchestrator depends only on the abstract interface,
so adding a new backbone is:

    1. Create framework/models/<family>/__init__.py + framework/models/<family>/planner.py
       - Subclass OpenAICompatiblePlanner (for any vLLM/OpenAI-compatible
         endpoint — Qwen, Gemma, Llama, Mistral, DeepSeek, ...)
       - Or subclass PlannerLLM directly for a non-OpenAI-compatible API
    2. Add the import + __all__ entry below
    3. Add a dispatch branch in framework/methods/ours/runner.py:build_planner()
"""
from .base import (
    PlannerInput,
    PlannerLLM,
    PlannerResponse,
    PlannerToolCall,
    ToolCallResult,
)
from .gemini import GeminiPlanner
from .gemini_3 import Gemini3Planner
from .openai_compatible import OpenAICompatiblePlanner
from .qwen import QwenPlanner
from .openai import OpenAIPlanner
from .anthropic import AnthropicPlanner
from .gemma import GemmaPlanner
from .kimi_vl import KimiVLPlanner

# Per-model subclasses go above. Devs adding a new family edit this list.
# Example:
#     from .gemma import GemmaPlanner

__all__ = [
    "PlannerInput",
    "PlannerLLM",
    "PlannerResponse",
    "PlannerToolCall",
    "ToolCallResult",
    "GeminiPlanner",
    "Gemini3Planner",
    "OpenAICompatiblePlanner",
    "QwenPlanner",
    "OpenAIPlanner",
    "AnthropicPlanner",
    "GemmaPlanner",
    "KimiVLPlanner",
]
