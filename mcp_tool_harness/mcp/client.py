"""Minimal MCP client abstractions.

This module intentionally depends only on the Python standard library.  It
defines a small synchronous JSON-RPC client surface plus transport shapes for
stdio, SSE, streamable HTTP, and in-memory tests.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence


DEFAULT_PROTOCOL_VERSION = "2025-06-18"


class MCPError(Exception):
    """Base exception for MCP client errors."""


class MCPTransportError(MCPError):
    """Raised when a transport cannot send or receive a message."""


class MCPProtocolError(MCPError):
    """Raised when a JSON-RPC response is malformed or contains an error."""

    def __init__(self, message: str, *, code: Optional[int] = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class Transport(Protocol):
    """Synchronous request/response transport contract used by MCPClient.

    MCPClient 只依赖这个最小协议，因此 stdio、HTTP、SSE 和测试内存实现可以自由替换。
    """

    def start(self) -> None:
        """Open transport resources."""

    def close(self) -> None:
        """Release transport resources."""

    def request(
        self,
        method: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        """Send a JSON-RPC request and return the response result."""


def _json_rpc_request(request_id: int, method: str, params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    # MCP 工具调用基于 JSON-RPC 2.0；每次请求都带 id，用来匹配响应。
    payload: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
    }
    if params is not None:
        payload["params"] = dict(params)
    return payload


def _parse_json_rpc_response(message: Mapping[str, Any], expected_id: Optional[int] = None) -> Any:
    # 这里做协议层校验，调用方只拿到 result；错误统一抛 MCPProtocolError。
    if message.get("jsonrpc") != "2.0":
        raise MCPProtocolError("Invalid JSON-RPC response: missing jsonrpc=2.0")
    if expected_id is not None and message.get("id") != expected_id:
        raise MCPProtocolError(
            "Invalid JSON-RPC response: response id does not match request id",
            data={"expected": expected_id, "actual": message.get("id")},
        )
    if "error" in message:
        error = message.get("error") or {}
        if isinstance(error, Mapping):
            raise MCPProtocolError(
                str(error.get("message", "MCP server returned an error")),
                code=error.get("code") if isinstance(error.get("code"), int) else None,
                data=error.get("data"),
            )
        raise MCPProtocolError(str(error))
    if "result" not in message:
        raise MCPProtocolError("Invalid JSON-RPC response: missing result")
    return message["result"]


class StdioTransport:
    """Line-delimited JSON-RPC transport for MCP stdio servers.

    适用于本地启动的 MCP Server，例如通过 `python server.py` 或二进制命令暴露 stdio。
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        encoding: str = "utf-8",
        default_timeout: float = 30.0,
    ) -> None:
        if not command:
            raise ValueError("stdio command must not be empty")
        self.command = tuple(command)
        self.cwd = cwd
        self.env = dict(env) if env is not None else None
        self.encoding = encoding
        self.default_timeout = default_timeout
        self._ids = count(1)
        self._process: Optional[subprocess.Popen[str]] = None
        self._responses: "queue.Queue[Any]" = queue.Queue()
        self._stderr_tail: List[str] = []
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        try:
            # stdio server 由 Harness 拉起；stdin/stdout 承载 JSON-RPC，stderr 只保留尾部用于排错。
            self._process = subprocess.Popen(
                list(self.command),
                cwd=self.cwd,
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding=self.encoding,
                bufsize=1,
            )
        except OSError as exc:
            raise MCPTransportError(f"Failed to start MCP stdio server: {exc}") from exc

        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._started = True

    def close(self) -> None:
        process = self._process
        if process is None:
            self._started = False
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._started = False

    def request(
        self,
        method: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise MCPTransportError("MCP stdio server is not running")
        if process.poll() is not None:
            raise MCPTransportError(self._format_process_exit())

        request_id = next(self._ids)
        payload = _json_rpc_request(request_id, method, params)
        wait_seconds = self.default_timeout if timeout is None else timeout

        with self._lock:
            # stdio 是单连接顺序写入，写和等响应放在同一把锁里，避免并发请求响应串线。
            try:
                process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
                process.stdin.flush()
            except OSError as exc:
                raise MCPTransportError(f"Failed to write MCP stdio request: {exc}") from exc

            deadline = time.monotonic() + wait_seconds if wait_seconds is not None else None
            while True:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining == 0.0:
                    raise MCPTransportError(f"Timed out waiting for MCP response to {method}")
                try:
                    message = self._responses.get(timeout=remaining)
                except queue.Empty as exc:
                    raise MCPTransportError(f"Timed out waiting for MCP response to {method}") from exc
                if isinstance(message, Exception):
                    raise MCPTransportError(str(message)) from message
                if not isinstance(message, Mapping):
                    continue
                if message.get("id") == request_id:
                    # 只消费当前 request_id 对应的响应；无关输出会继续等待。
                    return _parse_json_rpc_response(message, request_id)

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError as exc:
                self._responses.put(MCPTransportError(f"Invalid JSON from MCP stdio server: {exc}"))

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr_tail.append(line.rstrip())
            if len(self._stderr_tail) > 20:
                del self._stderr_tail[0]

    def _format_process_exit(self) -> str:
        process = self._process
        code = process.poll() if process is not None else None
        stderr = "\n".join(self._stderr_tail[-5:])
        detail = f" stderr tail: {stderr}" if stderr else ""
        return f"MCP stdio server exited with code {code}.{detail}"


class StreamableHTTPTransport:
    """HTTP JSON-RPC transport compatible with MCP streamable HTTP endpoints.

    适用于远程 MCP Server；认证头、网关地址、超时都在这个 transport 层处理。
    """

    def __init__(
        self,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        default_timeout: float = 30.0,
    ) -> None:
        self.url = url
        self.headers = dict(headers or {})
        self.default_timeout = default_timeout
        self._ids = count(1)

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def request(
        self,
        method: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        request_id = next(self._ids)
        payload = _json_rpc_request(request_id, method, params)
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        # Accept 同时声明 JSON 和 text/event-stream，兼容普通 JSON 响应和 MCP streamable HTTP 响应。
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            **self.headers,
        }
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.default_timeout if timeout is None else timeout) as response:
                raw = response.read().decode(response.headers.get_content_charset("utf-8"))
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.URLError as exc:
            raise MCPTransportError(f"HTTP MCP request failed: {exc}") from exc

        message = _decode_http_response(raw, content_type)
        return _parse_json_rpc_response(message, request_id)


class SSETransport:
    """SSE transport shape for MCP servers using a separate POST endpoint.

    The transport accepts a known message endpoint.  Full SSE session discovery
    can be layered above this class without changing MCPClient.
    """

    def __init__(
        self,
        sse_url: str,
        *,
        message_endpoint: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        default_timeout: float = 30.0,
    ) -> None:
        self.sse_url = sse_url
        self.message_endpoint = message_endpoint
        self.headers = dict(headers or {})
        self.default_timeout = default_timeout
        self._http: Optional[StreamableHTTPTransport] = None

    def start(self) -> None:
        if self.message_endpoint is None:
            return
        # 第一版只支持已知 message endpoint；完整 SSE 会话发现可以在外层 discovery 中增强。
        url = urllib.parse.urljoin(self.sse_url, self.message_endpoint)
        self._http = StreamableHTTPTransport(url, headers=self.headers, default_timeout=self.default_timeout)

    def close(self) -> None:
        if self._http is not None:
            self._http.close()

    def request(
        self,
        method: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        if self._http is None:
            self.start()
        if self._http is None:
            raise MCPTransportError(
                "SSETransport requires message_endpoint for request/response calls; "
                "provide a discovered MCP messages endpoint or a custom Transport."
            )
        return self._http.request(method, params, timeout=timeout)


def _decode_http_response(raw: str, content_type: str) -> Mapping[str, Any]:
    if "text/event-stream" in content_type:
        # streamable HTTP 可能返回 SSE 包装的数据帧，这里只取第一个 JSON data 事件作为响应。
        for event in _iter_sse_events(raw):
            data = event.get("data")
            if data:
                decoded = json.loads(data)
                if isinstance(decoded, Mapping):
                    return decoded
        raise MCPTransportError("HTTP MCP response did not contain an SSE data event")
    decoded = json.loads(raw)
    if not isinstance(decoded, Mapping):
        raise MCPProtocolError("HTTP MCP response must be a JSON object")
    return decoded


def _iter_sse_events(raw: str) -> Iterable[Dict[str, str]]:
    event: Dict[str, List[str]] = {}
    for line in raw.splitlines():
        if not line:
            if event:
                yield {key: "\n".join(value) for key, value in event.items()}
                event = {}
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        event.setdefault(field, []).append(value.lstrip())
    if event:
        yield {key: "\n".join(value) for key, value in event.items()}


ToolHandler = Callable[[Mapping[str, Any]], Any]


@dataclass
class _MockTool:
    name: str
    handler: ToolHandler
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    annotations: Mapping[str, Any] = field(default_factory=dict)

    def as_mcp_tool(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": dict(self.input_schema),
            "annotations": dict(self.annotations),
        }


class InMemoryTransport:
    """Deterministic MCP transport for tests and local harness composition.

    这个实现不走网络，适合单测 Gateway、Adapter，以及把本地函数模拟成 MCP Server。
    """

    def __init__(
        self,
        *,
        tools: Optional[Iterable[_MockTool]] = None,
        handlers: Optional[Mapping[str, Callable[[Mapping[str, Any]], Any]]] = None,
        server_info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # tools 保存 MCP 工具元数据和 handler，handlers 用于覆盖 initialize/tools/list 等原始方法。
        self._tools: Dict[str, _MockTool] = {tool.name: tool for tool in tools or ()}
        self._handlers = dict(handlers or {})
        self._server_info = dict(server_info or {"name": "in-memory-mcp", "version": "0.1.0"})
        self.started = False
        self.requests: List[Dict[str, Any]] = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.started = False

    def add_handler(self, method: str, handler: Callable[[Mapping[str, Any]], Any]) -> None:
        self._handlers[method] = handler

    def add_tool(
        self,
        name: str,
        handler: ToolHandler,
        *,
        description: str = "",
        input_schema: Optional[Mapping[str, Any]] = None,
        annotations: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._tools[name] = _MockTool(
            name=name,
            handler=handler,
            description=description,
            input_schema=dict(input_schema or {"type": "object", "properties": {}}),
            annotations=dict(annotations or {}),
        )

    def request(
        self,
        method: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        del timeout
        self.start()
        payload = dict(params or {})
        self.requests.append({"method": method, "params": payload})

        if method in self._handlers:
            # 自定义 handler 优先，便于测试异常、分页、协议兼容等场景。
            return self._handlers[method](payload)
        if method == "initialize":
            return {
                "protocolVersion": payload.get("protocolVersion", DEFAULT_PROTOCOL_VERSION),
                "capabilities": {"tools": {}},
                "serverInfo": self._server_info,
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [tool.as_mcp_tool() for tool in self._tools.values()]}
        if method == "tools/call":
            return self._call_tool(payload)
        raise MCPProtocolError(f"No in-memory handler registered for MCP method {method}")

    def _call_tool(self, params: Mapping[str, Any]) -> Any:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise MCPProtocolError("tools/call requires a non-empty tool name")
        tool = self._tools.get(name)
        if tool is None:
            raise MCPProtocolError(f"Unknown in-memory MCP tool: {name}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, Mapping):
            raise MCPProtocolError("tools/call arguments must be an object")
        result = tool.handler(arguments)
        # handler 可以返回普通 dict/string，也可以直接返回 MCP 标准 ToolResult。
        return _as_tool_result(result)


MockTransport = InMemoryTransport


def _as_tool_result(result: Any) -> Any:
    if isinstance(result, Mapping) and ("content" in result or "structuredContent" in result or "isError" in result):
        return result
    if isinstance(result, str):
        text = result
    else:
        text = json.dumps(result, ensure_ascii=False, default=str)
    payload: Dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": False}
    if isinstance(result, (Mapping, list)):
        payload["structuredContent"] = result
    return payload


class MCPClient:
    """Small synchronous MCP client facade.

    它只负责 MCP 协议调用，不做鉴权、审批、限流、审计；这些治理能力放在 ToolGateway。
    """

    def __init__(
        self,
        transport: Transport,
        *,
        client_name: str = "mcp-tool-harness",
        client_version: str = "0.1.0",
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        capabilities: Optional[Mapping[str, Any]] = None,
        auto_initialize: bool = False,
    ) -> None:
        self.transport = transport
        self.client_name = client_name
        self.client_version = client_version
        self.protocol_version = protocol_version
        self.capabilities = dict(capabilities or {})
        self.server_info: Dict[str, Any] = {}
        self.server_capabilities: Dict[str, Any] = {}
        self.initialized = False
        if auto_initialize:
            self.initialize()

    @classmethod
    def from_stdio(cls, command: Sequence[str], **kwargs: Any) -> "MCPClient":
        transport_keys = {"cwd", "env", "encoding", "default_timeout"}
        transport_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in transport_keys}
        return cls(StdioTransport(command, **transport_kwargs), **kwargs)

    @classmethod
    def from_sse(cls, sse_url: str, **kwargs: Any) -> "MCPClient":
        transport_keys = {"message_endpoint", "headers", "default_timeout"}
        transport_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in transport_keys}
        return cls(SSETransport(sse_url, **transport_kwargs), **kwargs)

    @classmethod
    def from_streamable_http(cls, url: str, **kwargs: Any) -> "MCPClient":
        transport_keys = {"headers", "default_timeout"}
        transport_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in transport_keys}
        return cls(StreamableHTTPTransport(url, **transport_kwargs), **kwargs)

    @classmethod
    def with_mock(cls, transport: Optional[InMemoryTransport] = None, **kwargs: Any) -> "MCPClient":
        return cls(transport or InMemoryTransport(), **kwargs)

    def __enter__(self) -> "MCPClient":
        self.transport.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        self.transport.close()

    def initialize(self, *, timeout: Optional[float] = None) -> Mapping[str, Any]:
        # initialize 记录服务端能力，后续可以据此判断是否支持 tools/resources/prompts。
        result = self.request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": self.capabilities,
                "clientInfo": {"name": self.client_name, "version": self.client_version},
            },
            timeout=timeout,
        )
        if not isinstance(result, Mapping):
            raise MCPProtocolError("initialize result must be an object")
        self.initialized = True
        self.server_info = dict(result.get("serverInfo") or {})
        self.server_capabilities = dict(result.get("capabilities") or {})
        return result

    def request(
        self,
        method: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        # 统一出口：上层 list_tools/call_tool 都会落到 transport.request。
        return self.transport.request(method, params, timeout=timeout)

    def ping(self, *, timeout: Optional[float] = None) -> Any:
        return self.request("ping", timeout=timeout)

    def list_tools(self, *, cursor: Optional[str] = None, timeout: Optional[float] = None) -> Any:
        # MCP tools/list 返回工具描述，Harness discovery 会把它规范化成 ToolSpec。
        params: Dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        return self.request("tools/list", params or None, timeout=timeout)

    def call_tool(
        self,
        name: str,
        arguments: Optional[Mapping[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        if not name:
            raise ValueError("tool name must not be empty")
        # MCP 标准调用参数是 name + arguments；业务参数必须包在 arguments 对象里。
        params = {"name": name, "arguments": dict(arguments or {})}
        return self.request("tools/call", params, timeout=timeout)
