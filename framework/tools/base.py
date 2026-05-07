"""Base tool interface for all agent tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class ToolParameter:
    name: str
    type: str  # "string", "number", "boolean"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_str(self) -> str:
        if not self.success:
            return f"[ERROR] {self.error}"
        if isinstance(self.data, str):
            return self.data
        if isinstance(self.data, (dict, list)):
            return json.dumps(self.data, ensure_ascii=False, indent=2)
        return str(self.data)


class BaseTool(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]: ...

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult: ...

    def to_schema(self) -> Dict[str, Any]:
        properties = {}
        required = []
        for param in self.parameters:
            prop: Dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(success=False, error=f"Tool execution failed: {e}")
