"""LlamaIndex adapter for MCP tools."""

from __future__ import annotations

from typing import Any, Iterable, List

from . import ensure_tool_spec, fallback_tool, make_invoker


def to_llamaindex_tool(client: Any, spec: Any, *, return_raw: bool = False, native: bool = True) -> Any:
    tool = ensure_tool_spec(spec)
    invoke = make_invoker(client, tool, return_raw=return_raw)
    if native:
        try:
            from llama_index.core.tools import FunctionTool
        except ImportError:
            return fallback_tool(client, tool, return_raw=return_raw)
        return FunctionTool.from_defaults(
            fn=invoke,
            name=tool.name,
            description=tool.description or f"Call MCP tool {tool.name}.",
        )
    return fallback_tool(client, tool, return_raw=return_raw)


def to_llamaindex_tools(client: Any, specs: Iterable[Any], *, return_raw: bool = False, native: bool = True) -> List[Any]:
    return [to_llamaindex_tool(client, spec, return_raw=return_raw, native=native) for spec in specs]


__all__ = ["to_llamaindex_tool", "to_llamaindex_tools"]

