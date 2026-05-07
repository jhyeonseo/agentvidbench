"""Qwen 3.x planner backend (text-only, used as ReAct planner).

Inherits OpenAICompatiblePlanner; adds Qwen-official sampling defaults
(top_p, top_k, repetition_penalty) and Thinking-variant detection.
"""
from .planner import QwenPlanner

__all__ = ["QwenPlanner"]
