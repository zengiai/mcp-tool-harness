from __future__ import annotations

import asyncio
from typing import Any, Mapping

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
from mcp_tool_harness.runtime import PolicyAwareRateLimiter


class _DelayedMCPClient:
    def __init__(self, client: MCPClient, delay_seconds: float) -> None:
        self.client = client
        self.delay_seconds = delay_seconds

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        await asyncio.sleep(self.delay_seconds)
        return self.client.call_tool(name, arguments)


@pytest.mark.asyncio
async def test_policy_aware_security_controls_agent_allowlist_and_risk() -> None:
    registry = Registry()
    await registry.register_tool(
        ToolSpec(name="order.cancel", description="Cancel order"),
        policy=ToolPolicy(
            tool_name="order.cancel",
            allowed_agents=frozenset({"order-agent"}),
            risk_level=RiskLevel.L1,
        ),
    )
    await registry.register_tool(
        ToolSpec(name="payment.refund", description="Refund payment"),
        policy=ToolPolicy(
            tool_name="payment.refund",
            allowed_agents=frozenset({"finance-agent"}),
            risk_level=RiskLevel.L2,
        ),
    )

    security = PolicyAwareSecurity(registry)
    tool = await registry.get_tool_by_identity("local", "order.cancel")

    denied = await security.check_permission(
        ToolCallContext(request_id="p-1", principal="crm-agent", tool_name="order.cancel"),
        tool,
        {},
    )
    assert denied.effect.value == "deny"

    allowed = await security.check_permission(
        ToolCallContext(request_id="p-2", principal="order-agent", tool_name="order.cancel"),
        tool,
        {},
    )
    assert allowed.effect.value == "allow"
    assert allowed.approval_required is False

    refund = await registry.get_tool_by_identity("local", "payment.refund")
    approval = await security.check_permission(
        ToolCallContext(request_id="p-3", principal="finance-agent", tool_name="payment.refund"),
        refund,
        {},
    )
    assert approval.effect.value == "require_approval"
    assert approval.approval_required is True


@pytest.mark.asyncio
async def test_policy_timeout_is_dynamic_and_overrides_gateway_default() -> None:
    registry = Registry(cache_ttl_seconds=0)
    await registry.register_tool(
        ToolSpec(name="inventory.rebuild", description="Rebuild inventory"),
        policy=ToolPolicy(
            tool_name="inventory.rebuild",
            allowed_agents=frozenset({"ops-agent"}),
            timeout_ms=20,
        ),
    )
    security = PolicyAwareSecurity(registry)

    transport = InMemoryTransport()
    transport.add_tool("inventory.rebuild", lambda _args: {"rebuilt": True})
    gateway = ToolGateway(
        registry=registry,
        security=security,
        mcp_client=_DelayedMCPClient(MCPClient.with_mock(transport), delay_seconds=0.1),
        default_timeout_ms=5_000,
    )

    result = await gateway.invoke(
        "inventory.rebuild",
        {},
        ToolCallContext(request_id="timeout-1", principal="ops-agent", tool_name="inventory.rebuild"),
    )
    assert result.error_code == "TOOL_TIMEOUT"
    assert "20ms" in (result.error_message or "")

    await registry.register_policy(
        ToolPolicy(
            tool_name="inventory.rebuild",
            allowed_agents=frozenset({"ops-agent"}),
            timeout_ms=250,
        )
    )
    assert await security.resolve_timeout_ms(
        ToolCallContext(request_id="timeout-2", principal="ops-agent", tool_name="inventory.rebuild"),
        await registry.get_tool_by_identity("local", "inventory.rebuild"),
    ) == 250


@pytest.mark.asyncio
async def test_policy_aware_rate_limiter_uses_multiple_dynamic_dimensions() -> None:
    registry = Registry(cache_ttl_seconds=0)
    await registry.register_tool(
        ToolSpec(name="coupon.reserve", description="Reserve coupon"),
        policy=ToolPolicy(
            tool_name="coupon.reserve",
            allowed_agents=frozenset({"*"}),
            rate_limits=(
                {
                    "dimension": "agent_tool",
                    "capacity": 2,
                    "refill_rate": 0.001,
                },
                {
                    "dimension": "custom",
                    "key_template": "tenant:{tenant_id}:order:{args.order_id}",
                    "capacity": 1,
                    "refill_rate": 0.001,
                },
            ),
        ),
    )

    transport = InMemoryTransport()
    transport.add_tool("coupon.reserve", lambda args: {"order_id": args["order_id"]})
    security = PolicyAwareSecurity(registry)
    gateway = ToolGateway(
        registry=registry,
        security=security,
        limiter=PolicyAwareRateLimiter(security=security),
        mcp_client=MCPClient.with_mock(transport),
    )

    async def invoke(request_id: str, order_id: str):
        return await gateway.invoke(
            "coupon.reserve",
            {"order_id": order_id},
            ToolCallContext(
                request_id=request_id,
                principal="agent-a",
                tool_name="coupon.reserve",
                tenant_id="tenant-a",
            ),
        )

    assert (await invoke("rl-1", "O-1")).success is True

    same_order = await invoke("rl-2", "O-1")
    assert same_order.status is ToolCallStatus.RATE_LIMITED

    assert (await invoke("rl-3", "O-2")).success is True

    agent_limit = await invoke("rl-4", "O-3")
    assert agent_limit.status is ToolCallStatus.RATE_LIMITED

    await registry.register_policy(
        ToolPolicy(
            tool_name="coupon.reserve",
            allowed_agents=frozenset({"*"}),
            rate_limits=(
                {
                    "dimension": "agent_tool",
                    "capacity": 4,
                    "refill_rate": 0.001,
                },
            ),
        )
    )
    assert (await invoke("rl-5", "O-3")).success is True
