"""Gemini3Planner — Gemini 3.0 (Pro Preview etc.) implementation.

Currently a thin subclass of GeminiPlanner. The Vertex AI google-genai SDK
exposes the same interface for 2.x and 3.x families (Part / FunctionDeclaration
/ GenerateContentConfig / ThinkingConfig), so most preview models work without
override.

Add overrides here only when preview-specific behavior diverges (e.g.,
ThinkingConfig schema change, new field, different default temperature). Keep
the gemini-2.x GeminiPlanner unmodified so existing experiments stay
reproducible.
"""
from __future__ import annotations

from framework.models.gemini.planner import GeminiPlanner


class Gemini3Planner(GeminiPlanner):
    pass
