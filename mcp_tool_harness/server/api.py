"""Server-facing API for the first-stage MCP Tool Harness SDK."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Mapping

from mcp_tool_harness.core.audit import default_audit_log_path
from mcp_tool_harness.observability.metrics import (
    default_metrics_log_path,
    read_metrics_events,
    summarize_tool_call_metrics,
)

from .approval import ALLOW, DENY, REQUIRES_APPROVAL, ApprovalPolicy
from .registry_api import RegisteredTool, ToolRegistry

try:  # FastAPI is optional for SDK users that only need the in-process gateway.
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response
except ModuleNotFoundError as exc:  # pragma: no cover - exact branch depends on env.
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    HTMLResponse = None  # type: ignore[assignment]
    Response = None  # type: ignore[assignment]
    _FASTAPI_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:  # pragma: no cover - exercised only when FastAPI is installed.
    _FASTAPI_IMPORT_ERROR = None


class ToolHarnessError(Exception):
    status_code = 500
    error_code = "tool_harness_error"


class ToolNotFoundError(ToolHarnessError):
    status_code = 404
    error_code = "tool_not_found"


class PermissionDeniedError(ToolHarnessError):
    status_code = 403
    error_code = "permission_denied"


class ApprovalRequiredError(ToolHarnessError):
    status_code = 202
    error_code = "approval_required"

    def __init__(self, message: str, *, approval_id: str | None = None) -> None:
        super().__init__(message)
        self.approval_id = approval_id


class RateLimitExceededError(ToolHarnessError):
    status_code = 429
    error_code = "rate_limit_exceeded"


class CircuitOpenError(ToolHarnessError):
    status_code = 503
    error_code = "circuit_open"


class IdempotencyConflictError(ToolHarnessError):
    status_code = 409
    error_code = "idempotency_conflict"


class ToolInputValidationError(ToolHarnessError):
    status_code = 400
    error_code = "tool_input_validation_error"


class ToolTimeoutError(ToolHarnessError):
    status_code = 504
    error_code = "tool_timeout"


class ToolExecutionError(ToolHarnessError):
    status_code = 500
    error_code = "tool_execution_error"


@dataclass(frozen=True)
class InvocationRequest:
    tool_name: str
    arguments: Mapping[str, Any] | None = None
    principal: str | Mapping[str, Any] | None = None
    idempotency_key: str | None = None
    request_id: str | None = None


@dataclass(frozen=True)
class InvocationResponse:
    status: str
    tool_name: str
    result: Any = None
    error: str | None = None
    approval_id: str | None = None
    cached: bool = False
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FixedWindowRateLimiter:
    """Small in-memory fixed-window limiter keyed by principal and tool."""

    def __init__(self, time_func: Callable[[], float] | None = None) -> None:
        self._time = time_func or time.monotonic
        self._windows: dict[str, tuple[int, int]] = {}

    def check(self, key: str, limit_per_minute: int) -> None:
        now_window = int(self._time() // 60)
        window, count = self._windows.get(key, (now_window, 0))
        if window != now_window:
            window, count = now_window, 0
        if count >= limit_per_minute:
            raise RateLimitExceededError(f"rate limit exceeded for '{key}'")
        self._windows[key] = (window, count + 1)


class CircuitBreaker:
    """In-memory per-tool circuit breaker."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_seconds: float = 30.0,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._time = time_func or time.monotonic
        self._state: dict[str, tuple[int, float | None]] = {}

    def before_call(self, tool_name: str) -> None:
        failures, opened_at = self._state.get(tool_name, (0, None))
        if opened_at is None:
            return
        if self._time() - opened_at >= self._recovery_seconds:
            self._state[tool_name] = (0, None)
            return
        raise CircuitOpenError(f"circuit is open for tool '{tool_name}'")

    def record_success(self, tool_name: str) -> None:
        self._state[tool_name] = (0, None)

    def record_failure(self, tool_name: str) -> None:
        failures, opened_at = self._state.get(tool_name, (0, None))
        if opened_at is not None:
            return
        failures += 1
        self._state[tool_name] = (
            failures,
            self._time() if failures >= self._failure_threshold else None,
        )


class ToolGateway:
    """Single-process gateway used by SDK tests and optional HTTP endpoints."""

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        approval_policy: ApprovalPolicy | None = None,
        default_rate_limit_per_minute: int | None = 60,
        default_timeout_ms: int | None = None,
        circuit_failure_threshold: int = 3,
        circuit_recovery_seconds: float = 30.0,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        if default_timeout_ms is not None and default_timeout_ms < 1:
            raise ValueError("default_timeout_ms must be positive when provided")
        self.registry = registry or ToolRegistry()
        self.approval_policy = approval_policy or ApprovalPolicy()
        self.default_rate_limit_per_minute = default_rate_limit_per_minute
        self.default_timeout_ms = default_timeout_ms
        self.rate_limiter = FixedWindowRateLimiter(time_func=time_func)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            recovery_seconds=circuit_recovery_seconds,
            time_func=time_func,
        )
        self._idempotency_cache: dict[str, tuple[str, InvocationResponse]] = {}

    def register_tool(self, name: str, handler: Callable[..., Any], **metadata: Any) -> RegisteredTool:
        return self.registry.register(name, handler, **metadata)

    def invoke(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        principal: str | Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> InvocationResponse:
        return _run_sync(
            self.ainvoke(
                tool_name,
                arguments,
                principal=principal,
                idempotency_key=idempotency_key,
                request_id=request_id,
            )
        )

    async def ainvoke(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        principal: str | Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> InvocationResponse:
        args = dict(arguments or {})
        registered = self._get_tool(tool_name)
        _validate_input_schema(registered.metadata.input_schema, args)
        self._enforce_approval(tool_name, principal, args)

        fingerprint = _fingerprint(tool_name, args, principal)
        if idempotency_key is not None:
            cached = self._idempotency_cache.get(idempotency_key)
            if cached is not None:
                cached_fingerprint, cached_response = cached
                if cached_fingerprint != fingerprint:
                    raise IdempotencyConflictError("idempotency key reused with different request data")
                return InvocationResponse(
                    status=cached_response.status,
                    tool_name=cached_response.tool_name,
                    result=cached_response.result,
                    error=cached_response.error,
                    approval_id=cached_response.approval_id,
                    cached=True,
                    request_id=request_id or cached_response.request_id,
                )

        limit = registered.metadata.rate_limit_per_minute
        if limit is None:
            limit = self.default_rate_limit_per_minute
        if limit is not None:
            self.rate_limiter.check(_limit_key(tool_name, principal), limit)

        timeout_ms = registered.metadata.timeout_ms or self.default_timeout_ms
        self.circuit_breaker.before_call(tool_name)
        try:
            result = await _call_handler(registered.handler, args, timeout_ms)
        except ToolTimeoutError:
            self.circuit_breaker.record_failure(tool_name)
            raise
        except Exception as exc:  # noqa: BLE001 - handlers are user code boundaries.
            self.circuit_breaker.record_failure(tool_name)
            raise ToolExecutionError(f"tool '{tool_name}' failed: {exc}") from exc

        self.circuit_breaker.record_success(tool_name)
        response = InvocationResponse(
            status="success",
            tool_name=tool_name,
            result=result,
            request_id=request_id,
        )
        if idempotency_key is not None:
            self._idempotency_cache[idempotency_key] = (fingerprint, response)
        return response

    def list_tools(self) -> list[dict[str, Any]]:
        return [metadata.to_dict() for metadata in self.registry.list()]

    def _get_tool(self, tool_name: str) -> RegisteredTool:
        try:
            return self.registry.get(tool_name)
        except KeyError as exc:
            raise ToolNotFoundError(f"tool '{tool_name}' is not registered") from exc

    def _enforce_approval(
        self,
        tool_name: str,
        principal: str | Mapping[str, Any] | None,
        arguments: Mapping[str, Any],
    ) -> None:
        decision = self.approval_policy.evaluate(
            tool_name=tool_name,
            principal=principal,
            arguments=arguments,
        )
        if decision.status == ALLOW:
            return
        if decision.status == DENY:
            raise PermissionDeniedError(decision.reason)
        if decision.status == REQUIRES_APPROVAL:
            raise ApprovalRequiredError(decision.reason, approval_id=decision.approval_id)
        raise ToolHarnessError(f"unknown approval decision '{decision.status}'")


def create_app(gateway: ToolGateway | None = None) -> Any:
    """Create a FastAPI app when the optional dependency is installed."""

    if FastAPI is None:
        detail = f": {_FASTAPI_IMPORT_ERROR}" if _FASTAPI_IMPORT_ERROR else ""
        raise RuntimeError(
            "FastAPI is optional for mcp_tool_harness.server. "
            "Install fastapi and uvicorn to use HTTP serving"
            f"{detail}."
        )

    app_gateway = gateway or ToolGateway()
    app = FastAPI(title="MCP Tool Harness", version="0.1.0")  # type: ignore[misc]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tools")
    async def list_tools() -> dict[str, Any]:
        return {"tools": app_gateway.list_tools()}

    @app.post("/tools/{tool_name}/invoke")
    async def invoke_tool(tool_name: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await app_gateway.ainvoke(
                tool_name,
                body.get("arguments") or {},
                principal=body.get("principal"),
                idempotency_key=body.get("idempotency_key"),
                request_id=body.get("request_id"),
            )
            return response.to_dict()
        except ToolHarnessError as exc:
            raise HTTPException(  # type: ignore[misc]
                status_code=exc.status_code,
                detail=_error_payload(exc),
            ) from exc

    @app.post("/mcp")
    async def mcp_json_rpc(message: dict[str, Any]) -> dict[str, Any]:
        return await _handle_json_rpc(app_gateway, message)

    @app.get("/console", response_class=HTMLResponse)
    async def console() -> Any:
        return HTMLResponse(_console_html())  # type: ignore[misc]

    @app.get("/console/assets/{asset_name}")
    async def console_asset(asset_name: str) -> Any:
        return _console_asset_response(asset_name)

    @app.get("/console/api/tools")
    async def console_tools() -> dict[str, Any]:
        return await _console_tools_payload(app_gateway)

    @app.get("/console/api/chains")
    async def console_chains(limit: int = 200) -> dict[str, Any]:
        return await asyncio.to_thread(_console_chains_payload, limit=limit)

    @app.get("/console/api/chains/{chain_id:path}")
    async def console_chain_detail(chain_id: str, limit: int = 1_000) -> dict[str, Any]:
        return await asyncio.to_thread(_console_chain_detail_payload, chain_id, limit=limit)

    @app.get("/console/api/metrics")
    async def console_metrics(limit: int = 1_000) -> dict[str, Any]:
        return await asyncio.to_thread(_console_metrics_payload, limit=limit)

    return app


async def _console_tools_payload(gateway: Any) -> dict[str, Any]:
    lister = getattr(gateway, "list_tools", None)
    if lister is None:
        registry = getattr(gateway, "registry", None)
        lister = getattr(registry, "list_tools", None) or getattr(registry, "list", None)
    if lister is None:
        return {"tools": [], "count": 0}
    tools = await _maybe_await(lister())
    normalized = [_normalize_tool(item) for item in tools or ()]
    return {"tools": normalized, "count": len(normalized)}


def _console_chains_payload(
    *,
    audit_path: str | Path | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    limit = _bounded_limit(limit, default=200)
    events = _read_json_lines(default_audit_log_path() if audit_path is None else audit_path, limit=max(limit * 10, limit))
    groups = _group_chain_events(events)
    chains = [_chain_summary(chain_id, chain_events) for chain_id, chain_events in groups.items()]
    chains.sort(key=lambda item: item["finished_at"] or 0, reverse=True)
    return {
        "chains": chains[:limit],
        "count": min(len(chains), limit),
        "total_available": len(chains),
    }


def _console_chain_detail_payload(
    chain_id: str,
    *,
    audit_path: str | Path | None = None,
    limit: int = 1_000,
) -> dict[str, Any]:
    limit = _bounded_limit(limit, default=1_000)
    events = _read_json_lines(default_audit_log_path() if audit_path is None else audit_path, limit=limit)
    matching = [event for event in events if _chain_id(event) == chain_id]
    matching.sort(key=_event_timestamp)
    return {
        "chain": _chain_summary(chain_id, matching) if matching else None,
        "events": matching,
        "count": len(matching),
    }


def _console_metrics_payload(
    *,
    metrics_path: str | Path | None = None,
    limit: int = 1_000,
) -> dict[str, Any]:
    limit = _bounded_limit(limit, default=1_000)
    events = read_metrics_events(metrics_path or default_metrics_log_path(), limit=limit)
    return {
        "summary": summarize_tool_call_metrics(events),
        "events": events,
        "count": len(events),
    }


def _group_chain_events(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        chain_id = _chain_id(event)
        groups.setdefault(chain_id, []).append(event)
    return groups


def _chain_summary(chain_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_events = sorted(events, key=_event_timestamp)
    status_counts: dict[str, int] = {}
    tools: list[str] = []
    for event in sorted_events:
        status = _event_status(event)
        status_counts[status] = status_counts.get(status, 0) + 1
        tool_name = _event_tool_name(event)
        if tool_name and tool_name not in tools:
            tools.append(tool_name)

    started_at = _event_timestamp(sorted_events[0]) if sorted_events else None
    finished_at = _event_timestamp(sorted_events[-1]) if sorted_events else None
    duration_ms = None
    if started_at is not None and finished_at is not None:
        duration_ms = max(0.0, (finished_at - started_at) * 1000)

    first = sorted_events[0] if sorted_events else {}
    metadata = _event_metadata(first)
    return {
        "chain_id": chain_id,
        "trace_id": first.get("correlation_id") or metadata.get("trace_id"),
        "request_id": first.get("request_id") or metadata.get("request_id"),
        "principal": first.get("actor") or metadata.get("principal"),
        "event_count": len(sorted_events),
        "tools": tools,
        "status_counts": status_counts,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
    }


def _chain_id(event: Mapping[str, Any]) -> str:
    metadata = _event_metadata(event)
    return str(
        event.get("correlation_id")
        or metadata.get("trace_id")
        or event.get("request_id")
        or metadata.get("request_id")
        or event.get("event_id")
        or "unknown"
    )


def _event_metadata(event: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = event.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _event_status(event: Mapping[str, Any]) -> str:
    metadata = _event_metadata(event)
    return str(metadata.get("status") or event.get("outcome") or "unknown")


def _event_tool_name(event: Mapping[str, Any]) -> str:
    metadata = _event_metadata(event)
    return str(metadata.get("tool_name") or event.get("action") or "")


def _event_timestamp(event: Mapping[str, Any]) -> float:
    try:
        return float(event.get("timestamp") or 0)
    except (TypeError, ValueError):
        return 0.0


def _read_json_lines(path: str | Path, *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except OSError:
        return []

    events: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _bounded_limit(value: int, *, default: int, maximum: int = 5_000) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(resolved, 1), maximum)


def _normalize_tool(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    to_dict = getattr(item, "to_dict", None)
    if to_dict is not None:
        result = to_dict()
        return dict(result) if isinstance(result, Mapping) else {"value": result}
    if hasattr(item, "__dataclass_fields__"):
        return asdict(item)
    return {"name": str(item)}


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


_CONSOLE_PACKAGE = "mcp_tool_harness.server"
_CONSOLE_ASSET_DIR = "console"
_CONSOLE_ASSET_TYPES = {
    "console.css": "text/css; charset=utf-8",
    "console.js": "application/javascript; charset=utf-8",
}


def _console_html() -> str:
    return _read_console_asset("index.html")


def _console_asset_response(asset_name: str) -> Any:
    media_type = _console_asset_media_type(asset_name)
    if media_type is None:
        raise HTTPException(status_code=404, detail="console asset not found")  # type: ignore[misc]
    return Response(_read_console_asset(asset_name), media_type=media_type)  # type: ignore[misc]


def _console_asset_media_type(asset_name: str) -> str | None:
    return _CONSOLE_ASSET_TYPES.get(asset_name)


def _read_console_asset(asset_name: str) -> str:
    try:
        asset = resources.files(_CONSOLE_PACKAGE).joinpath(_CONSOLE_ASSET_DIR, asset_name)
        return asset.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise RuntimeError(f"console asset is missing: {asset_name}") from exc


async def _handle_json_rpc(gateway: ToolGateway, message: Mapping[str, Any]) -> dict[str, Any]:
    request_id = message.get("id")
    method = message.get("method")
    try:
        if method == "tools/list":
            result = {"tools": gateway.list_tools()}
        elif method == "tools/call":
            params = dict(message.get("params") or {})
            response = await gateway.ainvoke(
                params["name"],
                params.get("arguments") or {},
                principal=params.get("principal"),
                idempotency_key=params.get("idempotency_key") or params.get("idempotencyKey"),
                request_id=str(request_id) if request_id is not None else None,
            )
            result = {
                "content": [{"type": "json", "json": response.result}],
                "structuredContent": response.result,
                "cached": response.cached,
            }
        else:
            return _json_rpc_error(request_id, -32601, f"unknown method '{method}'")
    except KeyError as exc:
        return _json_rpc_error(request_id, -32602, f"missing required parameter: {exc}")
    except ToolHarnessError as exc:
        return _json_rpc_error(request_id, exc.status_code, str(exc), data=_error_payload(exc))

    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _json_rpc_error(
    request_id: Any,
    code: int,
    message: str,
    *,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = dict(data)
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _error_payload(exc: ToolHarnessError) -> dict[str, Any]:
    payload = {"code": exc.error_code, "message": str(exc)}
    if isinstance(exc, ApprovalRequiredError):
        payload["approval_id"] = exc.approval_id
    return payload


async def _call_handler(
    handler: Callable[..., Any],
    args: Mapping[str, Any],
    timeout_ms: int | None,
) -> Any:
    async def call() -> Any:
        if timeout_ms is not None and not inspect.iscoroutinefunction(handler):
            result = await asyncio.to_thread(handler, **dict(args))
        else:
            result = handler(**args)
        if inspect.isawaitable(result):
            return await result
        return result

    if timeout_ms is None:
        return await call()
    try:
        return await asyncio.wait_for(call(), timeout=timeout_ms / 1000)
    except asyncio.TimeoutError as exc:
        raise ToolTimeoutError(f"tool handler timed out after {timeout_ms}ms") from exc


def _validate_input_schema(schema: Mapping[str, Any] | None, args: Mapping[str, Any]) -> None:
    """Validate a small JSON-schema subset before executing user code."""

    if not schema:
        return
    if schema.get("type") not in (None, "object"):
        raise ToolInputValidationError("tool input_schema root must be an object schema")

    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    missing = [name for name in required if name not in args]
    if missing:
        raise ToolInputValidationError(f"missing required arguments: {', '.join(missing)}")

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": Mapping,
        "array": list,
    }
    for name, value in args.items():
        expected = properties.get(name, {}).get("type")
        if expected is None:
            continue
        python_type = type_map.get(expected)
        if python_type is None:
            continue
        if expected == "integer" and isinstance(value, bool):
            raise ToolInputValidationError(f"argument {name} must be integer")
        if expected == "number" and isinstance(value, bool):
            raise ToolInputValidationError(f"argument {name} must be number")
        if not isinstance(value, python_type):
            raise ToolInputValidationError(f"argument {name} must be {expected}")


def _fingerprint(
    tool_name: str,
    arguments: Mapping[str, Any],
    principal: str | Mapping[str, Any] | None,
) -> str:
    return json.dumps(
        {"tool": tool_name, "arguments": arguments, "principal": principal},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _limit_key(tool_name: str, principal: str | Mapping[str, Any] | None) -> str:
    return f"{_principal_key(principal)}:{tool_name}"


def _principal_key(principal: str | Mapping[str, Any] | None) -> str:
    if principal is None:
        return "anonymous"
    if isinstance(principal, str):
        return principal
    return str(principal.get("subject") or principal.get("user") or principal.get("id") or "anonymous")


def _run_sync(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("ToolGateway.invoke cannot run inside an active event loop; use ainvoke instead")
