"""Unified governed tool invocation gateway."""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any

from .audit import create_default_audit_logger
from .models import (
    DecisionEffect,
    PolicyDecision,
    ToolCallContext,
    ToolCallStatus,
    ToolResult,
    stable_json_hash,
)
from .registry import ToolNotFoundError as RegistryToolNotFoundError
from mcp_tool_harness.observability.metrics import create_default_metrics_recorder


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
        self.audit = audit if audit is not None else create_default_audit_logger()
        self.approval_center = approval_center
        self.limiter = limiter
        self.circuit_breaker = circuit_breaker
        self.idempotency_store = idempotency_store
        self.metrics = metrics if metrics is not None else create_default_metrics_recorder()
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
        idempotency_key: str | None = None
        idempotency_started = False
        audit_result: ToolResult | None = None
        audit_decision = PolicyDecision.allowed("policy not evaluated")
        audit_enabled = True

        try:
            audit_enabled = await self._resolve_audit_enabled_best_effort(context, tool)
            # 主链路第一步只做本地 schema 校验，失败时不触发 MCP/RPC 下游，避免无效流量打到业务系统。
            self._validate_input_schema(getattr(tool, "input_schema", None), normalized_args)

            # 限流放在鉴权前，优先挡住突增流量；维度由 limiter 实现决定，默认兼容 tenant/agent/tool/server。
            if self.limiter is not None:
                allowed = await self._acquire_limit(context, tool, normalized_args)
                if not allowed:
                    status = ToolCallStatus.RATE_LIMITED
                    error_code = "RATE_LIMITED"
                    audit_decision = PolicyDecision(
                        effect=DecisionEffect.DENY,
                        reason="rate limited",
                        rate_limited=True,
                        metadata={"reason_code": error_code},
                    )
                    audit_result = ToolResult.failed(
                        call_id=context.call_id,
                        trace_id=context.trace_id,
                        error_code=error_code,
                        error_message="tool invocation rate limited",
                        status=status,
                    )
                    return audit_result

            # 权限和风险判断必须在执行前完成；模型或框架传来的调用意图不能被默认信任。
            decision = await self._check_permission(context, tool, normalized_args)
            audit_decision = decision
            if decision.effect is DecisionEffect.DENY:
                status = ToolCallStatus.DENIED
                error_code = decision.reason_code or "PERMISSION_DENIED"
                audit_result = ToolResult.rejected(
                    call_id=context.call_id,
                    trace_id=context.trace_id,
                    error_code=error_code,
                    error_message=decision.reason or "tool invocation denied",
                )
                return audit_result

            # 高风险工具默认走人工审批；没有审批中心时按失败处理，避免误放行写操作。
            if decision.effect is DecisionEffect.REQUIRE_APPROVAL:
                approved = await self._request_approval(context, tool, normalized_args, decision)
                if not approved:
                    status = ToolCallStatus.DENIED
                    error_code = "APPROVAL_REJECTED"
                    audit_result = ToolResult.rejected(
                        call_id=context.call_id,
                        trace_id=context.trace_id,
                        error_code=error_code,
                        error_message="tool invocation rejected by approval center",
                    )
                    return audit_result

            # 幂等记录只包住真正的执行阶段，按工具、参数和调用主体做自动去重。
            if self.idempotency_store is not None:
                idempotency_key, idempotency_started, idempotency_result = await self._prepare_idempotency(
                    tool,
                    normalized_args,
                    context,
                )
                if idempotency_result is not None:
                    audit_result = idempotency_result
                    status = idempotency_result.status
                    error_code = idempotency_result.error_code
                    return idempotency_result

            # 真实调用统一收敛到 MCP client；RPC 工具也应先包装成 MCP/Harness tool 再进入这里。
            result = await self._execute_with_protection(tool, normalized_args, context, timeout_ms)
            audit_result = result
            status = result.status
            error_code = result.error_code
            if idempotency_key and self.idempotency_store is not None:
                if result.success:
                    await self._store_idempotent_success(idempotency_key, result, idempotency_started)
                elif idempotency_started:
                    await self._store_idempotent_failure(
                        idempotency_key,
                        result.error_code or result.error_message or "tool result failed",
                    )
            return result
        except TimeoutError as exc:
            status = ToolCallStatus.FAILED
            error_code = "TOOL_TIMEOUT"
            if idempotency_key and idempotency_started:
                await self._store_idempotent_failure(idempotency_key, exc)
            audit_result = ToolResult.failed(
                call_id=context.call_id,
                trace_id=context.trace_id,
                error_code=error_code,
                error_message=str(exc) or "tool invocation timed out",
            )
            return audit_result
        except Exception as exc:  # noqa: BLE001 - gateway must normalize downstream failures.
            if exc.__class__.__name__ == "CircuitBreakerOpenError":
                status = ToolCallStatus.CIRCUIT_OPEN
                error_code = "CIRCUIT_OPEN"
                audit_decision = PolicyDecision(
                    effect=DecisionEffect.DENY,
                    reason="circuit open",
                    circuit_open=True,
                    metadata={"reason_code": error_code},
                )
            else:
                status = ToolCallStatus.FAILED
                error_code = exc.__class__.__name__
            if idempotency_key and idempotency_started:
                await self._store_idempotent_failure(idempotency_key, exc)
            audit_result = ToolResult.failed(
                call_id=context.call_id,
                trace_id=context.trace_id,
                error_code=error_code,
                error_message=str(exc),
                status=status,
            )
            return audit_result
        finally:
            # 指标和审计不能影响主调用结果；内部会吞掉埋点异常，保障工具调用路径稳定。
            latency_ms = int((asyncio.get_running_loop().time() - started) * 1000)
            await self._record_metrics(resolved_tool_name, status, latency_ms)
            if audit_enabled:
                await self._write_audit(
                    context,
                    tool,
                    normalized_args,
                    status,
                    error_code,
                    latency_ms,
                    result=audit_result,
                    decision=audit_decision,
                )

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

    async def _resolve_circuit_config(self, tool: Any, context: ToolCallContext) -> Any | None:
        """Resolve per‑tool circuit breaker config from the attached security component.

        Returns a :class:`CircuitBreakerConfig` when the underlying
        ``resolve_circuit_config`` method is present on the security adapter,
        or ``None`` otherwise so the gateway falls back to the injected
        ``self.circuit_breaker`` as‑is.
        """
        if self.security is not None and hasattr(self.security, "resolve_circuit_config"):
            return await self._maybe_await(
                self.security.resolve_circuit_config(context, tool)
            )
        return None

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

    async def _resolve_audit_enabled_best_effort(self, context: ToolCallContext, tool: Any) -> bool:
        try:
            return await self._resolve_audit_enabled(context, tool)
        except Exception:
            return True

    async def _resolve_audit_enabled(self, context: ToolCallContext, tool: Any) -> bool:
        policy = None
        if self.security is None:
            policy = await self._resolve_registry_policy_for_audit(context, tool)
        else:
            resolver = getattr(self.security, "resolve_policy", None)
            if resolver is not None:
                policy = await self._call_policy_resolver(resolver, context, tool)
            if policy is None:
                policy = await self._resolve_registry_policy_for_audit(context, tool)
        if policy is None:
            return True
        value = self._read_mapping_or_attr(policy, "audit_enabled", True)
        return bool(value)

    async def _resolve_registry_policy_for_audit(self, context: ToolCallContext, tool: Any) -> Any:
        lister = getattr(self.registry, "list_policies", None)
        if lister is None:
            return None
        tool_name = self._tool_name(tool)
        server_id = getattr(tool, "server_id", context.server_id)
        try:
            policies = lister(server_id=server_id, tool_name=tool_name, enabled=True)
        except TypeError:
            try:
                policies = lister(server_id=server_id, tool_name=tool_name)
            except TypeError:
                policies = lister()
        resolved = await self._maybe_await(policies)
        candidates: list[tuple[int, int, Any]] = []
        for index, policy in enumerate(resolved or ()):
            if self._read_mapping_or_attr(policy, "enabled", True) is not True:
                continue
            score = self._audit_policy_specificity(policy, server_id, tool_name)
            if score is not None:
                candidates.append((score, index, policy))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))[2]

    async def _call_policy_resolver(
        self,
        resolver: Callable[..., Any],
        context: ToolCallContext,
        tool: Any,
    ) -> Any:
        try:
            parameters = inspect.signature(resolver).parameters
        except (TypeError, ValueError):
            return await self._maybe_await(resolver(context, tool))
        if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters.values()):
            return await self._maybe_await(resolver(context, tool))
        kwargs: dict[str, Any] = {}
        if "context" in parameters:
            kwargs["context"] = context
        if "tool" in parameters:
            kwargs["tool"] = tool
        if kwargs:
            return await self._maybe_await(resolver(**kwargs))
        if len(parameters) >= 2:
            return await self._maybe_await(resolver(context, tool))
        return await self._maybe_await(resolver(tool))

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

        # 从 YAML 策略中解析熔断配置；未配置则走 Gateway 注入的全局默认熔断器。
        circuit_config = await self._resolve_circuit_config(tool, context)

        # 熔断器只包住下游调用，不包住本地校验/鉴权，避免策略失败污染下游健康状态。
        if self.circuit_breaker is not None:
            protected_call: Callable[[], Awaitable[Any]] = lambda: self._call_with_circuit(
                tool_name,
                call_mcp,
                circuit_config=circuit_config,
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
        *,
        result: ToolResult | None = None,
        decision: PolicyDecision | None = None,
    ) -> None:
        if self.audit is None:
            return
        try:
            writer = getattr(self.audit, "record_call", None)
            if writer is not None:
                kwargs = {
                    "context": context,
                    "tool": tool,
                    "args": args,
                    "status": status,
                    "error_code": error_code,
                    "latency_ms": latency_ms,
                    "result": result,
                    "decision": decision,
                }
                await self._maybe_await(writer(**self._supported_kwargs(writer, kwargs)))
                return

            logger = getattr(self.audit, "log", None)
            if logger is not None:
                server_id = getattr(tool, "server_id", context.server_id or "local")
                tool_name = self._tool_name(tool)
                metadata = {
                    "schema_version": "tool_call_audit.v1",
                    "arguments": dict(args),
                    "status": status.value,
                    "error_code": error_code,
                    "latency_ms": latency_ms,
                    "request_id": context.request_id,
                    "trace_id": context.trace_id,
                    "principal": context.principal,
                    "tenant_id": context.tenant_id,
                    "server_id": server_id,
                    "tool_name": tool_name,
                    "tool_id": getattr(tool, "tool_id", None),
                    "policy_id": getattr(decision, "policy_id", None),
                    "decision_id": getattr(decision, "decision_id", None),
                    "decision_effect": getattr(getattr(decision, "effect", None), "value", None),
                    "decision_reason": getattr(decision, "reason", None),
                    "audit_enabled": True,
                    "result_success": getattr(result, "success", None),
                    "result_status": getattr(getattr(result, "status", None), "value", None),
                    "context_metadata": dict(context.metadata),
                }
                await self._maybe_await(
                    logger(
                        "tool_call",
                        actor=context.principal,
                        action=tool_name,
                        resource=f"{server_id}/{tool_name}",
                        outcome="success" if status is ToolCallStatus.SUCCEEDED else "failure",
                        correlation_id=context.trace_id,
                        request_id=context.request_id,
                        metadata=metadata,
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

    async def _call_with_circuit(
        self,
        tool_name: str,
        call_mcp: Callable[[], Awaitable[Any]],
        *,
        circuit_config: Any | None = None,
    ) -> Any:
        breaker = self.circuit_breaker
        getter = getattr(breaker, "get", None)
        if getter is not None:
            if circuit_config is not None:
                breaker = await self._maybe_await(getter(tool_name, circuit_config))
            else:
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

    async def _prepare_idempotency(
        self,
        tool: Any,
        args: Mapping[str, Any],
        context: ToolCallContext,
    ) -> tuple[str, bool, ToolResult | None]:
        key = self._idempotency_key(tool, args, context)
        starter = getattr(self.idempotency_store, "start", None)
        if starter is None:
            cached = await self._get_cached_idempotent_result(key)
            if cached is not None:
                return key, False, self._mark_idempotent_replay(key, cached, "completed_replay")
            return key, False, None

        fingerprint = self._idempotency_fingerprint(tool, args, context)
        decision = await self._maybe_await(
            starter(
                key,
                fingerprint=fingerprint,
                metadata={
                    "trace_id": context.trace_id,
                    "request_id": context.request_id,
                    "auto_key": context.idempotency_key is None,
                },
            )
        )
        if getattr(decision, "replay", False):
            record = getattr(decision, "record", None)
            result = getattr(record, "result", None)
            if isinstance(result, ToolResult):
                return key, False, self._mark_idempotent_replay(
                    key,
                    result,
                    getattr(decision, "reason", "completed_replay"),
                )
        if not getattr(decision, "accepted", False):
            return key, False, self._idempotency_rejected_result(key, decision, context)
        return key, True, None

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

    def _idempotency_key(
        self,
        tool: Any,
        args: Mapping[str, Any],
        context: ToolCallContext,
    ) -> str:
        if context.idempotency_key:
            return context.idempotency_key
        return f"auto:{stable_json_hash(self._idempotency_scope(tool, args, context))}"

    def _idempotency_fingerprint(
        self,
        tool: Any,
        args: Mapping[str, Any],
        context: ToolCallContext,
    ) -> str:
        return stable_json_hash(self._idempotency_scope(tool, args, context))

    def _idempotency_scope(
        self,
        tool: Any,
        args: Mapping[str, Any],
        context: ToolCallContext,
    ) -> dict[str, Any]:
        return {
            "server_id": str(getattr(tool, "server_id", None) or context.server_id or "local"),
            "tool_name": self._tool_name(tool),
            "tenant_id": context.tenant_id,
            "principal": context.principal,
            "args": args,
        }

    @staticmethod
    def _mark_idempotent_replay(key: str, result: ToolResult, reason: str) -> ToolResult:
        metadata = dict(result.metadata)
        metadata.update(
            {
                "cached": True,
                "idempotency_hit": True,
                "idempotency_key": key,
                "idempotency_reason": reason,
            }
        )
        return replace(result, metadata=metadata)

    def _idempotency_rejected_result(
        self,
        key: str,
        decision: Any,
        context: ToolCallContext,
    ) -> ToolResult:
        reason = str(getattr(decision, "reason", "") or "rejected")
        error_code = {
            "fingerprint_mismatch": "IDEMPOTENCY_CONFLICT",
            "already_in_progress": "IDEMPOTENCY_IN_PROGRESS",
            "failed_not_retryable": "IDEMPOTENCY_FAILED_NOT_RETRYABLE",
            "failed_retry_exhausted": "IDEMPOTENCY_RETRY_EXHAUSTED",
        }.get(reason, "IDEMPOTENCY_REJECTED")
        result = ToolResult.failed(
            call_id=context.call_id,
            trace_id=context.trace_id,
            error_code=error_code,
            error_message=f"idempotency rejected: {reason}",
            status=ToolCallStatus.FAILED,
        )
        record = getattr(decision, "record", None)
        result.metadata.update(
            {
                "idempotency_key": key,
                "idempotency_reason": reason,
                "idempotency_rejected": True,
                "idempotency_attempt_count": getattr(record, "attempt_count", None),
            }
        )
        return result

    async def _store_idempotent_success(self, key: str, result: ToolResult, started: bool) -> None:
        if started and hasattr(self.idempotency_store, "complete"):
            await self._maybe_await(self.idempotency_store.complete(key, result))
            return
        putter = getattr(self.idempotency_store, "put", None)
        if putter is not None:
            await self._maybe_await(putter(key, result))

    async def _store_idempotent_failure(self, key: str, exc: BaseException | str) -> None:
        if hasattr(self.idempotency_store, "fail"):
            await self._maybe_await(self.idempotency_store.fail(key, exc))

    @staticmethod
    def _tool_name(tool: Any) -> str:
        return str(getattr(tool, "tool_name", None) or getattr(tool, "name"))

    @staticmethod
    def _read_mapping_or_attr(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _audit_policy_specificity(
        cls,
        policy: Any,
        server_id: str | None,
        tool_name: str,
    ) -> int | None:
        policy_tool = str(cls._read_mapping_or_attr(policy, "tool_name", "*"))
        policy_server = cls._read_mapping_or_attr(policy, "server_id", None)

        tool_exact = policy_tool == tool_name
        tool_wildcard = policy_tool == "*"
        tool_pattern = not tool_exact and not tool_wildcard and fnmatch.fnmatchcase(tool_name, policy_tool)
        if not (tool_exact or tool_wildcard or tool_pattern):
            return None

        server_specific = policy_server not in (None, "", "*")
        server_exact = server_specific and server_id is not None and str(policy_server) == str(server_id)
        if server_specific and not server_exact:
            return None

        if server_exact and tool_exact:
            return 300
        if not server_specific and tool_exact:
            return 200
        if server_exact and (tool_wildcard or tool_pattern):
            return 100
        if not server_specific and (tool_wildcard or tool_pattern):
            return 0
        return None

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _supported_kwargs(callable_obj: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        try:
            parameters = inspect.signature(callable_obj).parameters
        except (TypeError, ValueError):
            return dict(kwargs)
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return dict(kwargs)
        return {key: value for key, value in kwargs.items() if key in parameters}
