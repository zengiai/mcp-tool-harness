"""Tool discovery helpers for MCP servers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class ToolSpec:
    """Normalized MCP tool metadata used by adapters."""

    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    annotations: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Optional[Mapping[str, Any]] = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def as_openai_parameters(self) -> Mapping[str, Any]:
        """Return a JSON-schema object suitable for function-tool parameters."""

        if self.input_schema:
            return self.input_schema
        return {"type": "object", "properties": {}}


def normalize_tool(tool: Mapping[str, Any]) -> ToolSpec:
    """Normalize one MCP tool object into a ToolSpec."""

    name = tool.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("MCP tool is missing a non-empty name")
    description = tool.get("description")
    input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}
    annotations = tool.get("annotations") or {}
    output_schema = tool.get("outputSchema") or tool.get("output_schema")
    if not isinstance(input_schema, Mapping):
        input_schema = {}
    if not isinstance(annotations, Mapping):
        annotations = {}
    if output_schema is not None and not isinstance(output_schema, Mapping):
        output_schema = None
    return ToolSpec(
        name=name,
        description=description if isinstance(description, str) else "",
        input_schema=dict(input_schema),
        annotations=dict(annotations),
        output_schema=dict(output_schema) if output_schema is not None else None,
        raw=dict(tool),
    )


def normalize_tools(tools: Iterable[Mapping[str, Any]]) -> List[ToolSpec]:
    return [normalize_tool(tool) for tool in tools]


def discover_tools(client: Any, *, cursor: Optional[str] = None) -> List[ToolSpec]:
    """Discover tools through an MCPClient-like object."""

    response = client.list_tools(cursor=cursor)
    if isinstance(response, Mapping):
        tools = response.get("tools", [])
    else:
        tools = response
    if not isinstance(tools, Iterable):
        raise ValueError("MCP tools/list response must contain an iterable tools field")
    normalized: List[ToolSpec] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            raise ValueError("MCP tools/list contains a non-object tool entry")
        normalized.append(normalize_tool(tool))
    return normalized


def tool_map(tools: Iterable[ToolSpec]) -> Dict[str, ToolSpec]:
    """Index discovered tools by name."""

    return {tool.name: tool for tool in tools}

