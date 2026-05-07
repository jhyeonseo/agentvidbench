"""Shared cross-method utilities.

Modules:
  - token_tracker: token/retry accounting (wrap_genai_client, wrap_openai_client)
  - agent: ReActOrchestrator, TrajectoryLogger
  - models: PlannerLLM ABC + GeminiPlanner, QwenPlanner, OpenAICompatiblePlanner
  - tools: BaseTool ABC, ToolRegistry, VideoAnalyzerTool, GetTranscriptTool
"""
