"""MCP execution layer for tool-harness."""

from .client import (
    DEFAULT_PROTOCOL_VERSION,
    InMemoryTransport,
    MCPClient,
    MCPError,
    MCPProtocolError,
    MCPTransportError,
    MockTransport,
    SSETransport,
    StdioTransport,
    StreamableHTTPTransport,
    Transport,
)
from .discovery import ToolSpec, discover_tools, normalize_tool, normalize_tools, tool_map

__all__ = [
    "DEFAULT_PROTOCOL_VERSION",
    "InMemoryTransport",
    "MCPClient",
    "MCPError",
    "MCPProtocolError",
    "MCPTransportError",
    "MockTransport",
    "SSETransport",
    "StdioTransport",
    "StreamableHTTPTransport",
    "ToolSpec",
    "Transport",
    "discover_tools",
    "normalize_tool",
    "normalize_tools",
    "tool_map",
]

