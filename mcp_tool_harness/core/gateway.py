"""Unified governed tool invocation gateway."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from .models import (
    DecisionEffect,
    PolicyDecision,
    ToolCallContext,
    ToolCallStatus,
    ToolResult,
    stable_json_hash,
)
from .registry import ToolNotFoundError as RegistryToolNotFoundError


class ToolGatewayError(Exception):
    """Base exception for gateway-level failures."""


class ToolNotFoundError(ToolGatewayError):
    """Raised when a tool cannot be found in the registry."""


class ToolInputValidationError(ToolGatewayError):
    """Raised when arguments do not satisfy a tool input schema."""


class ToolGateway:
    """Advanced governance entry point for framework adapters.

    Thread-safety说明：
    - Gateway 自身不保存单次调用状态。
    - 限流、熔断、幂等等可变状态由 runtime 组件用自己的锁语义保证。

    事务边界说明：
    - Gateway 不做跨 MCP Server、RPC、数据库的分布式事务。
    - 业务写操作由下游服务保证本地事务、幂等与补偿。
    """

    def __init__(
        self,
        *,
        registry: Any,
        security: Any | None,
        mcp_client: Any,
        audit: Any | None = None,
        approval_center: Any | None = None,
        limiter: Any | None = None,
        circuit_breaker: Any | None = None,
        idempotency_store: Any | None = None,
        metrics: Any | None = None,
        tracer: Any | None = None,
        default_timeout_ms: int = 3_000,
    ) -> None:
        self.registry = registry
        self.security = security
        self.mcp_client = mcp_client
        self.audit = audit
        self.approval_center = approval_center
        self.limiter = limiter
        self.circuit_breaker = circuit_breaker
        self.idempotency_store = idempotency_store
        self.metrics = metrics
        self.tracer = tracer
        self.default_timeout_ms = default_timeout_ms

    async def invoke(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        context: ToolCallContext,
        *,
        version: str | None = None,
    ) -> ToolResult:
        """Invoke a registered tool through validation, policy, runtime protection, and MCP."""

        started = asyncio.get_running_loop().time()
        status = ToolCallStatus.SUCCEEDED
        error_code: str | None = None
        normalized_args = dict(args)
        tool = await self._get_tool(tool_name, version)
        resolved_tool_name = self._tool_name(tool)
        timeout_ms = await self._resolve_timeout_ms(tool, context)
        idempotency_key = context.idempotency_key
        idempotency_started = False

        try:
            # 主链路第一步只做本地 schema 校验，失败时不触发 MCP/RPC 下游，避免无效流量打到业务系统。
            self._validate_input_schema(getattr(tool, "input_schema", None), normalized_args)

            # 限流放在鉴权前，优先挡住突增流量；维度由 limiter 实现决定，默认兼容 tenant/agent/tool/server。
            if self.limiter is not None:
                allowed = await self._acquire_limit(context, tool, normalized_args)
                if not allowed:
                    status = ToolCallStatus.RATE_LIMITED
                    error_code = "RATE_LIMITED"
                    return ToolResult.failed(
                        call_id=context.call_id,
                        trace_id=context.trace_id,
                        error_code=error_code,
                        error_message="tool invocation rate limited",
                        status=status,
                    )

            # 权限和风险判断必须在执行前完成；模型或框架传来的调用意图不能被默认信任。
            decision = await self._check_permission(context, tool, normalized_args)
            if decision.effect is DecisionEffect.DENY:
                status = ToolCallStatus.DENIED
                error_code = decision.reason_code or "PERMISSION_DENIED"
                return ToolResult.rejected(
                    call_id=context.call_id,
                    trace_id=context.trace_id,
                    error_code=error_code,
                    error_message=decision.reason or "tool invocation denied",
                )

            # 高风险工具默认走人工审批；没有审批中心时按失败处理，避免误放行写操作。
            if decision.effect is DecisionEffect.REQUIRE_APPROVAL:
                approved = await self._request_approval(context, tool, normalized_args, decision)
                if not approved:
                    status = ToolCallStatus.DENIED
                    error_code = "APPROVAL_REJECTED"
                    return ToolResult.rejected(
                        call_id=context.call_id,
                        trace_id=context.trace_id,
                        error_code=error_code,
                        error_message="tool invocation rejected by approval center",
                    )

            # 幂等记录只包住真正的执行阶段，确保重试不会重复触发写操作类工具。
            if idempotency_key and self.idempotency_store is not None:
                cached = await self._get_cached_idempotent_result(idempotency_key)
                if cached is not None:
                    return cached
                idempotency_started = await self._start_idempotency(
                    idempotency_key,
                    tool,
                    normalized_args,
                    context,
                )

            # 真实调用统一收敛到 MCP client；RPC 工具也应先包装成 MCP/Harness tool 再进入这里。
            result = await self._execute_with_protection(tool, normalized_args, context, timeout_ms)
            if idempotency_key and self.idempotency_store is not None and result.success:
                await self._store_idempotent_success(idempotency_key, result, idempotency_started)
            return result
        except TimeoutError as exc:
            status = ToolCallStatus.FAILED
            error_code = "TOOL_TIMEOUT"
            if idempotency_key and idempotency_started:
                await self._store_idempotent_failure(idempotency_key, exc)
            return ToolResult.failed(
                call_id=context.call_id,
                trace_id=context.trace_id,
                error_code=error_code,
                error_message=str(exc) or "tool invocation timed out",
            )
        except Exception as exc:  # noqa: BLE001 - gateway must normalize downstream failures.
            status = ToolCallStatus.FAILED
            error_code = exc.__class__.__name__
            if idempotency_key and idempotency_started:
                await self._store_idempotent_failure(idempotency_key, exc)
            return ToolResult.failed(
                call_id=context.call_id,
                trace_id=context.trace_id,
                error_code=error_code,
                error_message=str(exc),
            )
        finally:
            # 指标和审计不能影响主调用结果；内部会吞掉埋点异常，保障工具调用路径稳定。
            latency_ms = int((asyncio.get_running_loop().time() - started) * 1000)
            await self._record_metrics(resolved_tool_name, status, latency_ms)
            await self._write_audit(context, tool, normalized_args, status, error_code, latency_ms)

    async def _get_tool(self, tool_name: str, version: str | None) -> Any:
        # 优先按 server/name/version 查，支持多 MCP Server 下相同 tool_name 的隔离。
        by_identity = getattr(self.registry, "get_tool_by_identity", None)
        if by_identity is not None:
            server_id = None
            resolved_name = tool_name
            if "/" in tool_name:
                server_id, resolved_name = tool_name.split("/", 1)
            try:
                return await self._maybe_await(
                    by_identity(server_id or "local", resolved_name, version or "1.0.0")
                )
            except (RegistryToolNotFoundError, KeyError):
                if server_id is not None:
                    raise

        # 兼容按 tool_id 查询的 Registry 实现，方便测试和轻量 SDK 入口复用同一个 Gateway。
        getter = getattr(self.registry, "get_tool", None)
        if getter is None:
            raise ToolGatewayError("registry must expose get_tool or get_tool_by_identity")
        try:
            tool = await self._maybe_await(getter(tool_name, version=version))
        except TypeError:
            tool = await self._maybe_await(getter(tool_name))
        if tool is None:
            raise ToolNotFoundError(f"tool not found: {tool_name}")
        return tool

    async def _resolve_timeout_ms(self, tool: Any, context: ToolCallContext) -> int:
        if self.security is not None and hasattr(self.security, "resolve_timeout_ms"):
            timeout = await self._maybe_await(self.security.resolve_timeout_ms(context, tool))
            if timeout:
                return int(timeout)
        return int(getattr(tool, "timeout_ms", None) or self.default_timeout_ms)

    async def _check_permission(
        self,
        context: ToolCallContext,
        tool: Any,
        args: Mapping[str, Any],
    ) -> PolicyDecision:
        if self.security is None:
            return PolicyDecision.allowed()

        # 兼容两类安全组件：显式 check_permission，或通用 policy engine evaluate。
        checker = getattr(self.security, "check_permission", None)
        if checker is not None:
            decision = await self._maybe_await(checker(context, tool, args))
        elif hasattr(self.security, "evaluate"):
            decision = await self._maybe_await(
                self.security.evaluate(
                    context,
                    self._tool_name(tool),
                    tool,
                    {
                        "arguments": dict(args),
                        "tenant_id": context.tenant_id,
                        "trace_id": context.trace_id,
                        **context.metadata,
                    },
                )
            )
        else:
            return PolicyDecision.allowed()

        if isinstance(decision, PolicyDecision):
            return decision
        if hasattr(decision, "approval_required") and getattr(decision, "approval_required"):
            return PolicyDecision.require_approval(getattr(decision, "reason", "approval required"))
        if hasattr(decision, "allowed") and getattr(decision, "allowed"):
            return PolicyDecision.allowed(getattr(decision, "reason", "allowed"))
        if hasattr(decision, "allowed") and not getattr(decision, "allowed"):
            return PolicyDecision.denied(getattr(decision, "reason", "permission denied"))
        if decision is True:
            return PolicyDecision.allowed()
        if decision == "approval_required":
            return PolicyDecision.require_approval("approval required")
        return PolicyDecision.denied("permission denied")

    async def _request_approval(
        self,
        context: ToolCallContext,
        tool: Any,
        args: Mapping[str, Any],
        decision: PolicyDecision,
    ) -> bool:
        if self.approval_center is None:
            return False
        requester = getattr(self.approval_center, "request_approval", None)
        if requester is None:
            return False
        approval_result = await self._maybe_await(requester(context, tool, args, decision))
        if isinstance(approval_result, bool):
            return approval_result
        return getattr(approval_result, "approved", False) is True

    async def _execute_with_protection(
        self,
        tool: Any,
        args: Mapping[str, Any],
        context: ToolCallContext,
        timeout_ms: int,
    ) -> ToolResult:
        tool_name = self._tool_name(tool)

        async def call_mcp() -> Any:
            return await self._call_mcp_tool(tool_name, args, context, tool)

        # 熔断器只包住下游调用，不包住本地校验/鉴权，避免策略失败污染下游健康状态。
        if self.circuit_breaker is not None:
            protected_call: Callable[[], Awaitable[Any]] = lambda: self._call_with_circuit(
                tool_name,
                call_mcp,
            )
        else:
            protected_call = call_mcp

        try:
            raw = await asyncio.wait_for(protected_call(), timeout=timeout_ms / 1000)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"tool {tool_name} timed out after {timeout_ms}ms") from exc
        return self._normalize_result(raw, context)

    def _normalize_result(self, raw: Any, context: ToolCallContext) -> ToolResult:
        if isinstance(raw, ToolResult):
            return raw
        if isinstance(raw, Mapping):
            # MCP 标准结果通常是 content/structuredContent/isError，这里归一成内部 ToolResult。
            success = bool(raw.get("success", not raw.get("isError", False)))
            if success:
                data = raw.get("structuredContent")
                if data is None:
                    data = raw.get("data", raw.get("result", raw))
                return ToolResult.success_result(
                    call_id=context.call_id,
                    trace_id=context.trace_id,
                    data=data,
                )
            return ToolResult.failed(
                call_id=context.call_id,
                trace_id=context.trace_id,
                error_code=str(raw.get("error_code", "TOOL_FAILED")),
                error_message=str(raw.get("error_message", raw.get("message", "tool failed"))),
            )
        return ToolResult.success_result(call_id=context.call_id, trace_id=context.trace_id, data=raw)

    def _validate_input_schema(self, schema: Mapping[str, Any] | None, args: Mapping[str, Any]) -> None:
        """Small JSON-schema subset validator."""

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

    async def _record_metrics(self, tool_name: str, status: ToolCallStatus, latency_ms: int) -> None:
        if self.metrics is None:
            return
        try:
            recorder = getattr(self.metrics, "record_tool_call", None)
            if recorder is not None:
                await self._maybe_await(recorder(tool_name, status.value, latency_ms))
                return
            increment = getattr(self.metrics, "increment", None)
            observe = getattr(self.metrics, "observe", None)
            if increment is not None:
                increment("tool_call_total", labels={"tool_name": tool_name, "status": status.value})
            if observe is not None:
                observe("tool_call_latency_ms", latency_ms, labels={"tool_name": tool_name})
        except Exception:
            return

    async def _write_audit(
        self,
        context: ToolCallContext,
        tool: Any,
        args: Mapping[str, Any],
        status: ToolCallStatus,
        error_code: str | None,
        latency_ms: int,
    ) -> None:
        if self.audit is None:
            return
        try:
            writer = getattr(self.audit, "record_call", None)
            if writer is not None:
                await self._maybe_await(
                    writer(
                        context=context,
                        tool=tool,
                        args=args,
                        status=status,
                        error_code=error_code,
                        latency_ms=latency_ms,
                    )
                )
                return

            logger = getattr(self.audit, "log", None)
            if logger is not None:
                await self._maybe_await(
                    logger(
                        "tool_call",
                        actor=context.principal,
                        action=self._tool_name(tool),
                        resource=f"{getattr(tool, 'server_id', 'local')}/{self._tool_name(tool)}",
                        outcome="success" if status is ToolCallStatus.SUCCEEDED else "failure",
                        correlation_id=context.trace_id,
                        request_id=context.request_id,
                        metadata={
                            "arguments": dict(args),
                            "status": status.value,
                            "error_code": error_code,
                            "latency_ms": latency_ms,
                        },
                    )
                )
        except Exception:
            return

    async def _acquire_limit(
        self,
        context: ToolCallContext,
        tool: Any,
        args: Mapping[str, Any],
    ) -> bool:
        acquire = getattr(self.limiter, "acquire", None)
        if acquire is None:
            return True

        # 默认隔离键覆盖租户、Agent、工具、MCP Server，避免单个热点工具拖垮整条链路。
        key = ":".join(
            [
                str(context.tenant_id or "default"),
                context.agent_id,
                self._tool_name(tool),
                str(getattr(tool, "server_id", "local")),
            ]
        )
        parameters = inspect.signature(acquire).parameters
        if "context" in parameters or "tool" in parameters or "args" in parameters:
            kwargs: dict[str, Any] = {}
            if "context" in parameters:
                kwargs["context"] = context
            if "tool" in parameters:
                kwargs["tool"] = tool
            if "args" in parameters:
                kwargs["args"] = args
            if "key" in parameters:
                kwargs["key"] = key
            decision = await self._maybe_await(acquire(**kwargs))
        elif "tenant_id" in parameters:
            decision = await self._maybe_await(
                acquire(
                    tenant_id=context.tenant_id,
                    agent_id=context.agent_id,
                    tool_name=self._tool_name(tool),
                    server_id=getattr(tool, "server_id", None),
                )
            )
        else:
            decision = await self._maybe_await(acquire(key))

        allowed = getattr(decision, "allowed", None)
        return True if allowed is None else bool(allowed)

    async def _call_mcp_tool(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        context: ToolCallContext,
        tool: Any,
    ) -> Any:
        caller = self.mcp_client.call_tool
        parameters = inspect.signature(caller).parameters
        kwargs: dict[str, Any] = {}
        if "context" in parameters:
            kwargs["context"] = context
        if "tool_spec" in parameters:
            kwargs["tool_spec"] = tool
        # 只传递 MCP Client 声明支持的扩展参数，兼容同步 client、异步 client 和测试 mock。
        return await self._maybe_await(caller(tool_name, dict(args), **kwargs))

    async def _call_with_circuit(self, tool_name: str, call_mcp: Callable[[], Awaitable[Any]]) -> Any:
        breaker = self.circuit_breaker
        getter = getattr(breaker, "get", None)
        if getter is not None:
            breaker = await self._maybe_await(getter(tool_name))

        # 兼容两种熔断器 API：一类提供 call 包装器，另一类提供 before/success/failure 三段式。
        caller = getattr(breaker, "call", None)
        if caller is not None:
            parameters = list(inspect.signature(caller).parameters)
            if parameters and parameters[0] in {"func", "callable_", "callback"}:
                return await self._maybe_await(caller(call_mcp))
            return await self._maybe_await(caller(tool_name, call_mcp))

        before_call = getattr(breaker, "before_call", None)
        record_success = getattr(breaker, "record_success", None)
        record_failure = getattr(breaker, "record_failure", None)
        if before_call is None:
            return await call_mcp()

        await self._maybe_await(before_call())
        try:
            result = await call_mcp()
        except Exception:
            if record_failure is not None:
                await self._maybe_await(record_failure())
            raise
        if record_success is not None:
            await self._maybe_await(record_success())
        return result

    async def _get_cached_idempotent_result(self, key: str) -> ToolResult | None:
        getter = getattr(self.idempotency_store, "get", None)
        if getter is None:
            return None
        cached = await self._maybe_await(getter(key))
        if cached is None:
            return None
        if isinstance(cached, ToolResult):
            return cached
        status = getattr(cached, "status", None)
        if getattr(status, "value", status) == "completed":
            result = getattr(cached, "result", None)
            if isinstance(result, ToolResult):
                return result
        return None

    async def _start_idempotency(
        self,
        key: str,
        tool: Any,
        args: Mapping[str, Any],
        context: ToolCallContext,
    ) -> bool:
        starter = getattr(self.idempotency_store, "start", None)
        if starter is None:
            return False
        # fingerprint 绑定工具名、参数和调用主体，防止同一个幂等 key 被不同请求复用。
        fingerprint = stable_json_hash(
            {"tool_name": self._tool_name(tool), "args": args, "principal": context.principal}
        )
        decision = await self._maybe_await(
            starter(
                key,
                fingerprint=fingerprint,
                metadata={"trace_id": context.trace_id, "request_id": context.request_id},
            )
        )
        if getattr(decision, "replay", False):
            return False
        return bool(getattr(decision, "accepted", False))

    async def _store_idempotent_success(self, key: str, result: ToolResult, started: bool) -> None:
        if started and hasattr(self.idempotency_store, "complete"):
            await self._maybe_await(self.idempotency_store.complete(key, result))
            return
        putter = getattr(self.idempotency_store, "put", None)
        if putter is not None:
            await self._maybe_await(putter(key, result))

    async def _store_idempotent_failure(self, key: str, exc: BaseException) -> None:
        if hasattr(self.idempotency_store, "fail"):
            await self._maybe_await(self.idempotency_store.fail(key, exc))

    @staticmethod
    def _tool_name(tool: Any) -> str:
        return str(getattr(tool, "tool_name", None) or getattr(tool, "name"))

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value
