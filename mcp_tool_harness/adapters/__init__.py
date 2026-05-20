"""Common adapter utilities for optional Agent framework integrations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from mcp_tool_harness.mcp.discovery import ToolSpec, normalize_tool


class OptionalDependencyError(RuntimeError):
    """Raised when native adapter output requires a missing optional dependency."""


def ensure_tool_spec(spec: Any) -> ToolSpec:
    if isinstance(spec, ToolSpec):
        return spec
    if isinstance(spec, Mapping):
        return normalize_tool(spec)
    raise TypeError("tool spec must be a ToolSpec or mapping")


def coerce_arguments(arguments: Any = None, **kwargs: Any) -> Mapping[str, Any]:
    """Coerce framework-specific tool input into MCP object arguments."""

    if kwargs:
        merged = {}
        if isinstance(arguments, Mapping):
            merged.update(arguments)
        elif arguments not in (None, ""):
            merged["input"] = arguments
        merged.update(kwargs)
        return merged
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return {"input": arguments}
        if isinstance(decoded, Mapping):
            return dict(decoded)
        return {"input": decoded}
    return {"input": arguments}


def format_mcp_result(result: Any) -> Any:
    """Format an MCP tool result for frameworks that expect simple returns."""

    if isinstance(result, Mapping):
        content = result.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, default=str))
            return "\n".join(part for part in parts if part)
        if "structuredContent" in result:
            return json.dumps(result["structuredContent"], ensure_ascii=False, default=str)
    if isinstance(result, (str, int, float, bool)) or result is None:
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


def make_invoker(client: Any, spec: Any, *, return_raw: bool = False) -> Callable[..., Any]:
    tool = ensure_tool_spec(spec)

    def invoke(arguments: Any = None, **kwargs: Any) -> Any:
        payload = coerce_arguments(arguments, **kwargs)
        result = client.call_tool(tool.name, payload)
        return result if return_raw else format_mcp_result(result)

    invoke.__name__ = _safe_identifier(tool.name)
    invoke.__doc__ = tool.description or f"Call MCP tool {tool.name}."
    return invoke


@dataclass
class FrameworkTool:
    """Standard-library fallback wrapper returned when a framework is absent."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    invoke: Callable[..., Any]
    native: Any = None

    def __call__(self, arguments: Any = None, **kwargs: Any) -> Any:
        return self.invoke(arguments, **kwargs)

    def run(self, tool_input: Any = None, **kwargs: Any) -> Any:
        return self.invoke(tool_input, **kwargs)

    async def arun(self, tool_input: Any = None, **kwargs: Any) -> Any:
        return self.invoke(tool_input, **kwargs)

    def as_openai_tool(self) -> Mapping[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


def fallback_tool(client: Any, spec: Any, *, return_raw: bool = False, native: Any = None) -> FrameworkTool:
    tool = ensure_tool_spec(spec)
    return FrameworkTool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.as_openai_parameters(),
        invoke=make_invoker(client, tool, return_raw=return_raw),
        native=native,
    )


def _safe_identifier(name: str) -> str:
    candidate = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not candidate or candidate[0].isdigit():
        candidate = f"tool_{candidate}"
    return candidate


__all__ = [
    "FrameworkTool",
    "OptionalDependencyError",
    "coerce_arguments",
    "ensure_tool_spec",
    "fallback_tool",
    "format_mcp_result",
    "make_invoker",
]

