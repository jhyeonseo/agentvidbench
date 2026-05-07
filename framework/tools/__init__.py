"""Cross-method tool implementations.

Tools are top-level so any method can import the same implementation —
ensuring fair comparison across methods that use a common toolset
(e.g., a future ReAct framework comparing to `ours` should use the same
VideoAnalyzerTool / GetTranscriptTool).

Re-exports for convenience:
    from framework.tools import (
        VideoAnalyzerTool, GetTranscriptTool,
        BaseTool, ToolRegistry, ToolParameter, ToolResult,
    )
"""
from .base import BaseTool, ToolParameter, ToolResult, ToolRegistry
from .video_analyzer import VideoAnalyzerTool
from .transcript_reader import GetTranscriptTool

__all__ = [
    "BaseTool", "ToolParameter", "ToolResult", "ToolRegistry",
    "VideoAnalyzerTool", "GetTranscriptTool",
]
