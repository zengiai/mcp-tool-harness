"""OpenAI Agents SDK adapter for MCP tools."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping

from . import ensure_tool_spec, fallback_tool, make_invoker


def to_openai_tool_schema(spec: Any) -> Mapping[str, Any]:
    tool = ensure_tool_spec(spec)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.as_openai_parameters(),
        },
    }


def to_openai_agents_tool(client: Any, spec: Any, *, return_raw: bool = False, native: bool = True) -> Any:
    tool = ensure_tool_spec(spec)
    invoke = make_invoker(client, tool, return_raw=return_raw)
    if native:
        try:
            from agents import function_tool
        except ImportError:
            return fallback_tool(client, tool, return_raw=return_raw)

        try:
            return function_tool(
                name_override=tool.name,
                description_override=tool.description or f"Call MCP tool {tool.name}.",
            )(invoke)
        except TypeError:
            decorated = function_tool(invoke)
            return decorated
    return fallback_tool(client, tool, return_raw=return_raw)


def to_openai_agents_tools(client: Any, specs: Iterable[Any], *, return_raw: bool = False, native: bool = True) -> List[Any]:
    return [to_openai_agents_tool(client, spec, return_raw=return_raw, native=native) for spec in specs]


__all__ = ["to_openai_agents_tool", "to_openai_agents_tools", "to_openai_tool_schema"]

