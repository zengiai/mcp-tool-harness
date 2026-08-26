"""Protocol-level tests for MCP mainstream-ecosystem compatibility (方案B).

Covers the five protocol gaps closed by feature MCPProtocolCompliance:
initialized notification, streamable HTTP session id, SSE endpoint discovery,
tools/list pagination, and the server-side /mcp handshake (initialize/ping/
notifications/standard content/reserved error codes).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import pytest

from mcp_tool_harness.config import MCPServerConfig
from mcp_tool_harness.mcp import InMemoryTransport, MCPClient, SSETransport, StreamableHTTPTransport
from mcp_tool_harness.mcp.discovery import discover_tools
from mcp_tool_harness.server import ToolGateway
from mcp_tool_harness.server.api import _handle_json_rpc


class _FakeHeaders:
    def __init__(self, headers: Mapping[str, str]) -> None:
        self._values = {key.lower(): value for key, value in headers.items()}

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key.lower(), default)

    def get_content_charset(self, default: Any = None) -> Any:
        return default


class _FakeResponse:
    """Minimal stand-in for urllib.response.addinfourl used by monkeypatched urlopen."""

    def __init__(self, body: str, headers: Mapping[str, str] | None = None) -> None:
        self._body = body.encode("utf-8")
        self.headers = _FakeHeaders(headers or {})

    def read(self) -> bytes:
        return self._body

    def __iter__(self):
        return iter(self._body.splitlines())

    def close(self) -> None:
        return None

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


# ---------------------------------------------------------------------------
# 客户端：握手通知 / session id / SSE 发现 / 分页 / 配置默认值
# ---------------------------------------------------------------------------


def test_initialize_sends_initialized_notification() -> None:
    transport = InMemoryTransport()
    client = MCPClient.with_mock(transport, auto_initialize=True)
    client.list_tools()
    methods = [item["method"] for item in transport.requests]
    assert methods[0] == "initialize"
    assert "notifications/initialized" in methods
    assert "tools/list" in methods


def test_manual_initialize_sends_notification() -> None:
    transport = InMemoryTransport()
    client = MCPClient.with_mock(transport)
    client.initialize()
    assert {"method": "notifications/initialized", "params": {}} in transport.requests


def test_streamable_http_echoes_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, str]] = []
    responses = [
        _FakeResponse(
            '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18",'
            '"capabilities":{},"serverInfo":{"name":"s","version":"1"}}}',
            {"Mcp-Session-Id": "abc123"},
        ),
        # notifications/initialized 的响应体为空（202 风格）。
        _FakeResponse("", {}),
        _FakeResponse('{"jsonrpc":"2.0","id":2,"result":{"tools":[]}}', {}),
    ]

    def fake_urlopen(request: Any, timeout: float | None = None) -> Any:
        seen.append({key.lower(): value for key, value in request.headers.items()})
        return responses.pop(0)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = MCPClient(StreamableHTTPTransport("https://example.com/mcp"), auto_initialize=True)
    client.list_tools()
    assert len(seen) == 3
    assert seen[1].get("mcp-session-id") == "abc123"
    assert seen[2].get("mcp-session-id") == "abc123"


def test_sse_discovers_message_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    sse_body = "event: endpoint\ndata: /mcp?session_id=sess-1\n\n"

    def fake_urlopen(request: Any, timeout: float | None = None) -> Any:
        calls.append(request)
        if request.method == "GET":
            return _FakeResponse(sse_body, {})
        request_id = json.loads(request.data).get("id", 1)
        return _FakeResponse(
            json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {"tools": []}}),
            {},
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = MCPClient(SSETransport("https://example.com/sse"), auto_initialize=True)
    client.list_tools()
    assert calls[0].method == "GET"
    assert calls[0].full_url == "https://example.com/sse"
    posts = [request for request in calls if request.method == "POST"]
    assert posts
    # 发现的 endpoint（含 session id query）被用于消息 POST。
    assert posts[0].full_url == "https://example.com/mcp?session_id=sess-1"


def test_sse_explicit_message_endpoint_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> Any:
        calls.append(request)
        request_id = json.loads(request.data).get("id", 1)
        return _FakeResponse(
            json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {"tools": []}}),
            {},
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = MCPClient(SSETransport("https://example.com/sse", message_endpoint="/mcp"), auto_initialize=True)
    client.list_tools()
    assert calls and all(request.method == "POST" for request in calls)
    assert calls[0].full_url == "https://example.com/mcp"


class _PagedToolsClient:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = list(pages)
        self.cursors: list[str | None] = []

    def list_tools(self, *, cursor: str | None = None) -> dict[str, Any]:
        self.cursors.append(cursor)
        return self._pages.pop(0)


def test_discover_tools_paginates_next_cursor() -> None:
    client = _PagedToolsClient(
        [
            {
                "tools": [
                    {"name": "alpha", "description": "a", "inputSchema": {"type": "object"}},
                    {"name": "beta", "description": "b", "inputSchema": {"type": "object"}},
                ],
                "nextCursor": "page-2",
            },
            {
                "tools": [
                    {"name": "beta", "description": "b again", "inputSchema": {"type": "object"}},
                    {"name": "gamma", "description": "c", "inputSchema": {"type": "object"}},
                ]
            },
        ]
    )
    specs = discover_tools(client)
    assert [spec.name for spec in specs] == ["alpha", "beta", "gamma"]
    assert client.cursors == [None, "page-2"]


def test_mcp_server_config_auto_initialize_defaults_true() -> None:
    assert MCPServerConfig(server_id="x").auto_initialize is True


# ---------------------------------------------------------------------------
# 服务端：initialize / ping / 通知 / 标准 content / 错误码
# ---------------------------------------------------------------------------


def _make_gateway() -> ToolGateway:
    gateway = ToolGateway(default_rate_limit_per_minute=None)
    gateway.register_tool("text.echo", lambda text: {"echo": text})
    return gateway


@pytest.mark.asyncio
async def test_mcp_endpoint_initialize_handshake() -> None:
    result = await _handle_json_rpc(
        _make_gateway(),
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1"},
            },
        },
    )
    assert result is not None
    payload = result["result"]
    assert payload["protocolVersion"] == "2025-06-18"
    assert payload["capabilities"] == {"tools": {}}
    assert payload["serverInfo"]["name"] == "mcp-tool-harness"


@pytest.mark.asyncio
async def test_mcp_endpoint_ping() -> None:
    result = await _handle_json_rpc(_make_gateway(), {"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert result == {"jsonrpc": "2.0", "id": 1, "result": {}}


@pytest.mark.asyncio
async def test_mcp_endpoint_tools_list_uses_mcp_schema() -> None:
    # 冒烟验证发现：官方客户端要求 camelCase 的 inputSchema；内部 snake_case 不透出。
    result = await _handle_json_rpc(_make_gateway(), {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert result is not None
    tools = result["result"]["tools"]
    assert tools[0]["name"] == "text.echo"
    assert "inputSchema" in tools[0]
    assert "input_schema" not in tools[0]


@pytest.mark.asyncio
async def test_mcp_endpoint_notification_returns_none() -> None:
    result = await _handle_json_rpc(
        _make_gateway(), {"jsonrpc": "2.0", "method": "notifications/initialized"}
    )
    assert result is None


@pytest.mark.asyncio
async def test_mcp_endpoint_tools_call_standard_content() -> None:
    result = await _handle_json_rpc(
        _make_gateway(),
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "text.echo", "arguments": {"text": "hello"}},
        },
    )
    assert result is not None
    payload = result["result"]
    assert payload["content"] == [{"type": "text", "text": '{"echo": "hello"}'}]
    assert payload["structuredContent"] == {"echo": "hello"}


@pytest.mark.asyncio
async def test_mcp_endpoint_error_uses_reserved_code() -> None:
    result = await _handle_json_rpc(
        _make_gateway(),
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "no.such.tool", "arguments": {}},
        },
    )
    assert result is not None
    assert result["error"]["code"] == -32000
    assert result["error"]["data"]["status_code"] == 404


@pytest.mark.asyncio
async def test_mcp_endpoint_repeated_initialize_idempotent() -> None:
    message = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    first = await _handle_json_rpc(_make_gateway(), message)
    second = await _handle_json_rpc(_make_gateway(), message)
    assert first == second
