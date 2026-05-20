"""LangChain adapter for MCP tools."""

from __future__ import annotations

from typing import Any, Iterable, List

from . import ensure_tool_spec, fallback_tool, make_invoker


def to_langchain_tool(client: Any, spec: Any, *, return_raw: bool = False, native: bool = True) -> Any:
    tool = ensure_tool_spec(spec)
    invoke = make_invoker(client, tool, return_raw=return_raw)
    if native:
        try:
            from langchain_core.tools import StructuredTool, Tool
        except ImportError:
            try:
                from langchain.tools import StructuredTool, Tool
            except ImportError:
                return fallback_tool(client, tool, return_raw=return_raw)
        try:
            return StructuredTool.from_function(
                func=invoke,
                name=tool.name,
                description=tool.description or f"Call MCP tool {tool.name}.",
            )
        except Exception:
            return Tool.from_function(
                func=lambda tool_input="": invoke(tool_input),
                name=tool.name,
                description=tool.description or f"Call MCP tool {tool.name}.",
            )
    return fallback_tool(client, tool, return_raw=return_raw)


def to_langchain_tools(client: Any, specs: Iterable[Any], *, return_raw: bool = False, native: bool = True) -> List[Any]:
    return [to_langchain_tool(client, spec, return_raw=return_raw, native=native) for spec in specs]


__all__ = ["to_langchain_tool", "to_langchain_tools"]

