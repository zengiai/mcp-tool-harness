"""验证 YAML 策略中的熔断参数能正确传递到 CircuitBreaker 实例。"""

from __future__ import annotations

from typing import Any

import pytest

from mcp_tool_harness.core import (
    PolicyAwareSecurity,
    Registry,
    RiskLevel,
    ToolCallContext,
    ToolCallStatus,
    ToolPolicy,
    ToolSpec,
)
from mcp_tool_harness.core.gateway import ToolGateway
from mcp_tool_harness.mcp import InMemoryTransport, MCPClient
from mcp_tool_harness.runtime import (
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
)


# ── 工具函数 ────────────────────────────────────────────────


def _failing_handler(_args: Any) -> Any:
    raise RuntimeError("simulated downstream failure")


# ── resolve_circuit_config 单元测试 ──────────────────────────


@pytest.mark.asyncio
async def test_resolve_circuit_config_returns_config_from_tool_policy() -> None:
    """ToolPolicy 中显式配置全部四个参数时，全部应被解析。"""
    registry = Registry()
    await registry.register_tool(
        ToolSpec(name="payment.capture", description="Capture payment"),
        policy=ToolPolicy(
            tool_name="payment.capture",
            circuit_failure_threshold=3,
            circuit_reset_timeout_seconds=60,
            circuit_success_threshold=4,
            circuit_half_open_max_calls=2,
            risk_level=RiskLevel.L2,
        ),
    )

    security = PolicyAwareSecurity(registry)
    tool = await registry.get_tool_by_identity("local", "payment.capture")
    ctx = ToolCallContext(
        request_id="req-1",
        principal="agent-a",
        tool_name="payment.capture",
    )

    config = await security.resolve_circuit_config(ctx, tool)

    assert config is not None
    assert isinstance(config, CircuitBreakerConfig)
    assert config.failure_threshold == 3
    assert config.recovery_timeout == 60.0
    assert config.success_threshold == 4
    assert config.half_open_max_calls == 2


@pytest.mark.asyncio
async def test_resolve_circuit_config_uses_defaults_when_fields_not_set() -> None:
    """ToolPolicy 存在但未显式配置熔断字段时，返回携带默认值的配置。"""
    registry = Registry()
    await registry.register_tool(
        ToolSpec(name="query.stats", description="Query stats"),
        policy=ToolPolicy(
            tool_name="query.stats",
            risk_level=RiskLevel.L0,
        ),
    )

    security = PolicyAwareSecurity(registry)
    tool = await registry.get_tool_by_identity("local", "query.stats")
    ctx = ToolCallContext(
        request_id="req-2",
        principal="agent-b",
        tool_name="query.stats",
    )

    config = await security.resolve_circuit_config(ctx, tool)

    assert config is not None
    assert config.failure_threshold == 5  # ToolPolicy 默认值
    assert config.recovery_timeout == 30.0  # ToolPolicy 默认值
    assert config.success_threshold == 2  # ToolPolicy 默认值
    assert config.half_open_max_calls == 1  # ToolPolicy 默认值


@pytest.mark.asyncio
async def test_resolve_circuit_config_returns_none_when_no_policy() -> None:
    """未注册任何策略时返回 None。"""
    registry = Registry()
    await registry.register_tool(
        ToolSpec(name="no-policy-tool", description="No policy attached"),
    )

    security = PolicyAwareSecurity(registry)
    tool = await registry.get_tool_by_identity("local", "no-policy-tool")
    ctx = ToolCallContext(
        request_id="req-3",
        principal="agent-c",
        tool_name="no-policy-tool",
    )

    config = await security.resolve_circuit_config(ctx, tool)

    assert config is None


# ── Gateway 集成测试：YAML 策略 → CircuitBreaker ─────────────


@pytest.mark.asyncio
async def test_gateway_circuit_breaker_uses_yaml_policy_threshold() -> None:
    """YAML 策略中 circuit_failure_threshold=2 应覆盖默认值 5，两次失败即熔断。"""
    registry = Registry()
    await registry.register_tool(
        ToolSpec(
            name="flaky.tool",
            description="Always fails",
            input_schema={"type": "object", "properties": {}},
        ),
        policy=ToolPolicy(
            tool_name="flaky.tool",
            circuit_failure_threshold=2,
            circuit_reset_timeout_seconds=300,
            risk_level=RiskLevel.L1,
        ),
    )

    transport = InMemoryTransport()
    transport.add_tool("flaky.tool", _failing_handler)
    mcp_client = MCPClient.with_mock(transport)

    security = PolicyAwareSecurity(registry)
    breaker_registry = CircuitBreakerRegistry()

    gateway = ToolGateway(
        registry=registry,
        security=security,
        mcp_client=mcp_client,
        circuit_breaker=breaker_registry,
    )

    ctx = ToolCallContext(
        request_id="req-flaky",
        principal="agent-d",
        tool_name="flaky.tool",
    )

    # 前两次调用：失败，但不触发熔断（threshold=2，需要 ≥2 次）
    result1 = await gateway.invoke("flaky.tool", {}, ctx)
    assert result1.status == ToolCallStatus.FAILED

    result2 = await gateway.invoke("flaky.tool", {}, ctx)
    assert result2.status == ToolCallStatus.FAILED

    # 第三次调用：熔断已打开
    result3 = await gateway.invoke("flaky.tool", {}, ctx)
    assert result3.status == ToolCallStatus.CIRCUIT_OPEN
    assert result3.error_code == "CIRCUIT_OPEN"

    # 验证熔断器实例使用了 YAML 配置的参数
    breaker = await breaker_registry.get("flaky.tool")
    snapshot = await breaker.snapshot()
    assert snapshot["state"] == "open"
    assert snapshot["config"]["failure_threshold"] == 2
    assert snapshot["config"]["recovery_timeout"] == 300.0


@pytest.mark.asyncio
async def test_gateway_circuit_breaker_defaults_when_policy_has_no_circuit_config() -> None:
    """策略中未显式配置熔断参数时，熔断器使用 ToolPolicy 默认值（5/30）。"""
    registry = Registry()
    await registry.register_tool(
        ToolSpec(
            name="default.tool",
            description="No circuit config",
            input_schema={"type": "object", "properties": {}},
        ),
        policy=ToolPolicy(
            tool_name="default.tool",
            risk_level=RiskLevel.L1,
            allowed_agents=frozenset({"agent-e"}),
        ),
    )

    transport = InMemoryTransport()
    transport.add_tool("default.tool", _failing_handler)
    mcp_client = MCPClient.with_mock(transport)

    security = PolicyAwareSecurity(registry)
    breaker_registry = CircuitBreakerRegistry()

    gateway = ToolGateway(
        registry=registry,
        security=security,
        mcp_client=mcp_client,
        circuit_breaker=breaker_registry,
    )

    ctx = ToolCallContext(
        request_id="req-default",
        principal="agent-e",
        tool_name="default.tool",
    )

    # 默认 failure_threshold=5，前 4 次失败不触发熔断
    for _ in range(4):
        result = await gateway.invoke("default.tool", {}, ctx)
        assert result.status == ToolCallStatus.FAILED

    # 第 5 次仍失败（仍未触发，因为需要 ≥5）
    result5 = await gateway.invoke("default.tool", {}, ctx)
    assert result5.status == ToolCallStatus.FAILED

    # 第 6 次：熔断打开（默认 threshold=5，第 5 次失败触发熔断，第 6 次被拒绝）
    result6 = await gateway.invoke("default.tool", {}, ctx)
    assert result6.status == ToolCallStatus.CIRCUIT_OPEN

    breaker = await breaker_registry.get("default.tool")
    snapshot = await breaker.snapshot()
    assert snapshot["config"]["failure_threshold"] == 5  # Registry 默认值


@pytest.mark.asyncio
async def test_gateway_circuit_breaker_registry_creates_per_tool_instance() -> None:
    """不同工具的熔断器互相独立，各自的 YAML 配置互不干扰。"""
    registry = Registry()

    # 工具 A：threshold=2
    await registry.register_tool(
        ToolSpec(name="tool-a", description="A", input_schema={"type": "object", "properties": {}}),
        policy=ToolPolicy(
            tool_name="tool-a",
            circuit_failure_threshold=2,
            risk_level=RiskLevel.L1,
        ),
    )
    # 工具 B：threshold=3
    await registry.register_tool(
        ToolSpec(name="tool-b", description="B", input_schema={"type": "object", "properties": {}}),
        policy=ToolPolicy(
            tool_name="tool-b",
            circuit_failure_threshold=3,
            risk_level=RiskLevel.L1,
        ),
    )

    transport = InMemoryTransport()
    transport.add_tool("tool-a", _failing_handler)
    transport.add_tool("tool-b", _failing_handler)
    mcp_client = MCPClient.with_mock(transport)

    security = PolicyAwareSecurity(registry)
    breaker_registry = CircuitBreakerRegistry()

    gateway = ToolGateway(
        registry=registry,
        security=security,
        mcp_client=mcp_client,
        circuit_breaker=breaker_registry,
    )

    ctx_a = ToolCallContext(request_id="r-a", principal="agent", tool_name="tool-a")
    ctx_b = ToolCallContext(request_id="r-b", principal="agent", tool_name="tool-b")

    # 工具 A：调用 2 次失败 → 第 3 次熔断
    await gateway.invoke("tool-a", {}, ctx_a)
    await gateway.invoke("tool-a", {}, ctx_a)
    result_a3 = await gateway.invoke("tool-a", {}, ctx_a)
    assert result_a3.status == ToolCallStatus.CIRCUIT_OPEN

    # 工具 B：仅调用 1 次失败，不应触发熔断（threshold=3）
    result_b = await gateway.invoke("tool-b", {}, ctx_b)
    assert result_b.status == ToolCallStatus.FAILED

    # 验证两个熔断器实例的配置各自独立
    snap_a = await (await breaker_registry.get("tool-a")).snapshot()
    snap_b = await (await breaker_registry.get("tool-b")).snapshot()
    assert snap_a["config"]["failure_threshold"] == 2
    assert snap_b["config"]["failure_threshold"] == 3
    assert snap_a["state"] == "open"
    assert snap_b["state"] == "closed"


@pytest.mark.asyncio
async def test_yaml_circuit_config_partial_override() -> None:
    """只配置 failure_threshold 时，recovery_timeout 保持默认。"""
    registry = Registry()
    await registry.register_tool(
        ToolSpec(name="partial.tool", description="Partial config"),
        policy=ToolPolicy(
            tool_name="partial.tool",
            circuit_failure_threshold=1,
            # 不配置 circuit_reset_timeout_seconds
            risk_level=RiskLevel.L1,
        ),
    )

    security = PolicyAwareSecurity(registry)
    tool = await registry.get_tool_by_identity("local", "partial.tool")
    ctx = ToolCallContext(request_id="r", principal="agent", tool_name="partial.tool")

    config = await security.resolve_circuit_config(ctx, tool)

    assert config is not None
    assert config.failure_threshold == 1
    assert config.recovery_timeout == 30.0  # 未配置，走默认值
    assert config.success_threshold == 2     # 未配置，走默认值
    assert config.half_open_max_calls == 1   # 未配置，走默认值
