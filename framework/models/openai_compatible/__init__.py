"""Generic OpenAI-compatible planner backend.

Works with any vLLM-served model (Qwen, Gemma, Llama, Mistral, DeepSeek, ...)
since vLLM normalizes tool-call wire formats via --tool-call-parser. Per-family
subclasses (e.g., QwenPlanner) inherit this for family-specific tweaks.
"""
from .planner import OpenAICompatiblePlanner
from .json_prompt import JsonPromptPlanner

__all__ = ["OpenAICompatiblePlanner", "JsonPromptPlanner"]
