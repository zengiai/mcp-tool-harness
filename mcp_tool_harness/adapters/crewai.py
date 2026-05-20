"""CrewAI adapter for MCP tools."""

from __future__ import annotations

from typing import Any, Iterable, List

from . import ensure_tool_spec, fallback_tool, make_invoker


def to_crewai_tool(client: Any, spec: Any, *, return_raw: bool = False, native: bool = True) -> Any:
    tool = ensure_tool_spec(spec)
    invoke = make_invoker(client, tool, return_raw=return_raw)
    if native:
        try:
            from crewai.tools import BaseTool
        except ImportError:
            return fallback_tool(client, tool, return_raw=return_raw)

        class MCPCrewAITool(BaseTool):  # type: ignore[misc, valid-type]
            name: str = tool.name
            description: str = tool.description or f"Call MCP tool {tool.name}."

            def _run(self, *args: Any, **kwargs: Any) -> Any:
                argument = args[0] if args else None
                return invoke(argument, **kwargs)

        return MCPCrewAITool()
    return fallback_tool(client, tool, return_raw=return_raw)


def to_crewai_tools(client: Any, specs: Iterable[Any], *, return_raw: bool = False, native: bool = True) -> List[Any]:
    return [to_crewai_tool(client, spec, return_raw=return_raw, native=native) for spec in specs]


__all__ = ["to_crewai_tool", "to_crewai_tools"]

