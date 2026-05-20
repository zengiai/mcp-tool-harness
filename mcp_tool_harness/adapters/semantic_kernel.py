"""Semantic Kernel adapter for MCP tools."""

from __future__ import annotations

from typing import Any, Iterable, List

from . import ensure_tool_spec, fallback_tool, make_invoker


def to_semantic_kernel_function(client: Any, spec: Any, *, return_raw: bool = False, native: bool = True) -> Any:
    tool = ensure_tool_spec(spec)
    invoke = make_invoker(client, tool, return_raw=return_raw)
    if native:
        try:
            from semantic_kernel.functions import kernel_function
        except ImportError:
            return fallback_tool(client, tool, return_raw=return_raw)

        @kernel_function(name=tool.name, description=tool.description or f"Call MCP tool {tool.name}.")
        def mcp_tool(arguments: str = "") -> Any:
            return invoke(arguments)

        return mcp_tool
    return fallback_tool(client, tool, return_raw=return_raw)


def to_semantic_kernel_functions(
    client: Any,
    specs: Iterable[Any],
    *,
    return_raw: bool = False,
    native: bool = True,
) -> List[Any]:
    return [to_semantic_kernel_function(client, spec, return_raw=return_raw, native=native) for spec in specs]


__all__ = ["to_semantic_kernel_function", "to_semantic_kernel_functions"]

