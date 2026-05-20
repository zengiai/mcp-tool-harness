from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from mcp_tool_harness.core import (
    PolicyDecision,
    Registry,
    RiskLevel,
    ToolCallContext,
    ToolCallStatus,
    ToolPolicy,
    ToolSpec,
)
from mcp_tool_harness.core.gateway import ToolGateway
from mcp_tool_harness.mcp import InMemoryTransport, MCPClient
from mcp_tool_harness.runtime.limiter import RateLimitDecision


class _RegistryBackedPolicySecurity:
    """测试用策略解析器：动态 Registry 策略优先，代码默认策略只做兜底。"""

    def __init__(
        self,
        registry: Registry,
        *,
        code_default_policies: Mapping[str, ToolPolicy] | None = None,
    ) -> None:
        self.registry = registry
        self.code_default_policies = dict(code_default_policies or {})

    async def resolve_policy(self, tool: ToolSpec) -> ToolPolicy:
        policies = await self.registry.list_policies(
            server_id=tool.server_id,
            tool_name=tool.name,
            enabled=True,
        )
        if policies:
            return max(policies, key=lambda item: self._specificity(item, tool))
        return (
            self.code_default_policies.get(f"{tool.server_id}/{tool.name}")
            or self.code_default_policies.get(tool.name)
            or self.code_default_policies.get("*")
            or ToolPolicy(tool_name=tool.name, server_id=tool.server_id)
        )

    async def resolve_timeout_ms(self, context: ToolCallContext, tool: ToolSpec) -> int:
        del context
        return (await self.resolve_policy(tool)).timeout_ms

    async def check_permission(
        self,
        context: ToolCallContext,
        tool: ToolSpec,
        args: Mapping[str, Any],
    ) -> PolicyDecision:
        del args
        policy = await self.resolve_policy(tool)
        if not policy.enabled:
            return PolicyDecision.denied("tool disabled by dynamic policy", policy.policy_id)

        principal_candidates = {context.principal, context.agent_id}
        denied = set(policy.denied_principals)
        if "*" in denied or principal_candidates.intersection(denied):
            return PolicyDecision.denied("agent denied by tool policy", policy.policy_id)

        allowed = set(policy.allowed_principals)
        if allowed and "*" not in allowed and not principal_candidates.intersection(allowed):
            return PolicyDecision.denied("agent not in tool allowlist", policy.policy_id)

        permission_level = str(policy.metadata.get("permission_level", "")).lower()
        if policy.require_approval and permission_level not in {"l0", "l1"}:
            return PolicyDecision.require_approval("approval required by tool policy", policy.policy_id)
        return PolicyDecision.allowed("allowed by tool policy", policy.policy_id)

    @staticmethod
    def _specificity(policy: ToolPolicy, tool: ToolSpec) -> int:
        score = 0
        if policy.tool_name == tool.name:
            score += 100
        elif policy.tool_name == "*":
            score += 1
        if policy.server_id == tool.server_id:
            score += 10
        return score


class _PolicyAwareMultiDimensionLimiter:
    """按策略 metadata.rate_limits 校验多维限流，成功调用会同时消耗所有维度。"""

    def __init__(self, security: _RegistryBackedPolicySecurity) -> None:
        self.security = security
        self.counts: dict[str, int] = {}
        self.rejected_keys: list[str] = []

    async def acquire(
        self,
        *,
        context: ToolCallContext,
        tool: ToolSpec,
        args: Mapping[str, Any],
        key: str,
    ) -> RateLimitDecision:
        policy = await self.security.resolve_policy(tool)
        specs = list(policy.metadata.get("rate_limits") or ())
        if not specs and policy.rate_limit_per_minute is not None:
            specs = [{"name": "default", "key_template": key, "limit_per_minute": policy.rate_limit_per_minute}]

        resolved: list[tuple[str, int]] = [
            (self._render_key(spec, context, tool, args), int(spec["limit_per_minute"]))
            for spec in specs
        ]
        for limit_key, limit in resolved:
            if self.counts.get(limit_key, 0) >= limit:
                self.rejected_keys.append(limit_key)
                return RateLimitDecision(
                    allowed=False,
                    key=limit_key,
                    remaining=0,
                    retry_after=60,
                    capacity=limit,
                    refill_rate=limit / 60,
                )

        for limit_key, _limit in resolved:
            self.counts[limit_key] = self.counts.get(limit_key, 0) + 1

        return RateLimitDecision(
            allowed=True,
            key=",".join(limit_key for limit_key, _limit in resolved),
            remaining=1,
            retry_after=0,
            capacity=1,
            refill_rate=1,
        )

    @staticmethod
    def _render_key(
        spec: Mapping[str, Any],
        context: ToolCallContext,
        tool: ToolSpec,
        args: Mapping[str, Any],
    ) -> str:
        template = str(spec["key_template"])
        return template.format(
            tenant_id=context.tenant_id or "default",
            agent_id=context.agent_id,
            principal=context.principal,
            tool_name=tool.name,
            server_id=tool.server_id,
            campaign_id=args.get("campaign_id", ""),
        )


class _AsyncMCPClient:
    """让 core.ToolGateway 的 asyncio.wait_for 能覆盖异步下游耗时。"""

    def __init__(self, client: MCPClient, *, delay_seconds: float = 0.0) -> None:
        self.client = client
        self.delay_seconds = delay_seconds

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return self.client.call_tool(name, arguments)


@dataclass
class _RecordingApprovalCenter:
    requests: int = 0

    async def request_approval(self, *_args: Any, **_kwargs: Any) -> bool:
        self.requests += 1
        return False


@pytest.mark.asyncio
async def test_dynamic_tool_rate_limits_support_multiple_dimensions_and_custom_keys() -> None:
    registry = Registry()
    await registry.register_tool(
        ToolSpec(name="coupon.reserve", description="Reserve coupon inventory"),
        policy=ToolPolicy(
            tool_name="coupon.reserve",
            allowed_principals=frozenset({"*"}),
            metadata={
                "rate_limits": [
                    {
                        "name": "tenant_tool",
                        "key_template": "tenant:{tenant_id}:tool:{tool_name}",
                        "limit_per_minute": 4,
                    },
                    {
                        "name": "agent_tool",
                        "key_template": "agent:{agent_id}:tool:{tool_name}",
                        "limit_per_minute": 2,
                    },
                    {
                        "name": "campaign",
                        "key_template": "campaign:{campaign_id}",
                        "limit_per_minute": 1,
                    },
                ],
            },
        ),
    )

    transport = InMemoryTransport()
    transport.add_tool("coupon.reserve", lambda args: {"campaign_id": args["campaign_id"]})
    security = _RegistryBackedPolicySecurity(registry)
    limiter = _PolicyAwareMultiDimensionLimiter(security)
    gateway = ToolGateway(
        registry=registry,
        security=security,
        limiter=limiter,
        mcp_client=MCPClient.with_mock(transport),
    )

    async def invoke(request_id: str, principal: str, campaign_id: str):
        context = ToolCallContext(
            request_id=request_id,
            principal=principal,
            tool_name="coupon.reserve",
            tenant_id="tenant-a",
            metadata={"agent_id": principal},
        )
        return await gateway.invoke("coupon.reserve", {"campaign_id": campaign_id}, context)

    assert (await invoke("call-1", "agent-a", "campaign-1")).success is True

    same_campaign = await invoke("call-2", "agent-b", "campaign-1")
    assert same_campaign.status is ToolCallStatus.RATE_LIMITED
    assert same_campaign.error_code == "RATE_LIMITED"

    assert (await invoke("call-3", "agent-a", "campaign-2")).success is True

    same_agent = await invoke("call-4", "agent-a", "campaign-3")
    assert same_agent.status is ToolCallStatus.RATE_LIMITED
    assert limiter.rejected_keys == [
        "campaign:campaign-1",
        "agent:agent-a:tool:coupon.reserve",
    ]
    assert [item["method"] for item in transport.requests] == ["tools/call", "tools/call"]


@pytest.mark.asyncio
async def test_tool_policy_timeout_ms_bounds_core_gateway_execution() -> None:
    registry = Registry()
    await registry.register_tool(
        ToolSpec(name="inventory.rebuild", description="Rebuild inventory view"),
        policy=ToolPolicy(
            tool_name="inventory.rebuild",
            allowed_principals=frozenset({"*"}),
            timeout_ms=20,
        ),
    )

    transport = InMemoryTransport()
    transport.add_tool("inventory.rebuild", lambda _args: {"rebuilt": True})
    gateway = ToolGateway(
        registry=registry,
        security=_RegistryBackedPolicySecurity(registry),
        mcp_client=_AsyncMCPClient(MCPClient.with_mock(transport), delay_seconds=0.1),
        default_timeout_ms=5_000,
    )
    context = ToolCallContext(
        request_id="timeout-call-1",
        principal="ops-agent",
        tool_name="inventory.rebuild",
    )

    result = await gateway.invoke("inventory.rebuild", {}, context)

    assert result.success is False
    assert result.error_code == "TOOL_TIMEOUT"
    assert "20ms" in (result.error_message or "")
    assert transport.requests == []


@pytest.mark.asyncio
async def test_tool_permission_uses_agent_allowlist_star_and_l0_l1_without_approval() -> None:
    registry = Registry()
    await registry.register_tool(
        ToolSpec(name="order.cancel", description="Cancel an order"),
        policy=ToolPolicy(
            tool_name="order.cancel",
            allowed_principals=frozenset({"order-agent"}),
        ),
    )
    await registry.register_tool(
        ToolSpec(name="catalog.search", description="Search catalog"),
        policy=ToolPolicy(
            tool_name="catalog.search",
            risk_level=RiskLevel.LOW,
            allowed_principals=frozenset({"*"}),
            metadata={"permission_level": "l1"},
        ),
    )

    transport = InMemoryTransport()
    transport.add_tool("order.cancel", lambda _args: {"cancelled": True})
    transport.add_tool("catalog.search", lambda _args: {"items": []})
    approval_center = _RecordingApprovalCenter()
    gateway = ToolGateway(
        registry=registry,
        security=_RegistryBackedPolicySecurity(registry),
        approval_center=approval_center,
        mcp_client=MCPClient.with_mock(transport),
    )

    denied = await gateway.invoke(
        "order.cancel",
        {},
        ToolCallContext(
            request_id="allowlist-call-1",
            principal="profile-agent",
            tool_name="order.cancel",
        ),
    )
    assert denied.status is ToolCallStatus.DENIED
    assert denied.error_code == "PERMISSION_DENIED"

    allowed = await gateway.invoke(
        "catalog.search",
        {},
        ToolCallContext(
            request_id="allowlist-call-2",
            principal="any-agent",
            tool_name="catalog.search",
        ),
    )
    assert allowed.success is True
    assert allowed.output == {"items": []}
    assert approval_center.requests == 0


@pytest.mark.asyncio
async def test_dynamic_policy_overrides_code_defaults_and_can_be_updated() -> None:
    registry = Registry(cache_ttl_seconds=0)
    await registry.register_tool(ToolSpec(name="risk.score", description="Score order risk"))
    code_default = ToolPolicy(
        tool_name="risk.score",
        allowed_principals=frozenset({"*"}),
        timeout_ms=5_000,
    )
    dynamic_policy = await registry.register_policy(
        ToolPolicy(
            tool_name="risk.score",
            allowed_principals=frozenset({"risk-agent"}),
            timeout_ms=80,
        )
    )

    transport = InMemoryTransport()
    transport.add_tool("risk.score", lambda _args: {"score": 12})
    security = _RegistryBackedPolicySecurity(
        registry,
        code_default_policies={"risk.score": code_default},
    )
    gateway = ToolGateway(
        registry=registry,
        security=security,
        mcp_client=MCPClient.with_mock(transport),
    )

    denied = await gateway.invoke(
        "risk.score",
        {},
        ToolCallContext(
            request_id="dynamic-call-1",
            principal="crm-agent",
            tool_name="risk.score",
        ),
    )
    assert denied.status is ToolCallStatus.DENIED

    tool = await registry.get_tool_by_identity("local", "risk.score")
    assert (await security.resolve_policy(tool)).policy_id == dynamic_policy.policy_id
    assert await security.resolve_timeout_ms(
        ToolCallContext(
            request_id="dynamic-call-2",
            principal="risk-agent",
            tool_name="risk.score",
        ),
        tool,
    ) == 80

    await registry.register_policy(
        ToolPolicy(
            tool_name="risk.score",
            allowed_principals=frozenset({"risk-agent"}),
            timeout_ms=30,
        )
    )
    assert await security.resolve_timeout_ms(
        ToolCallContext(
            request_id="dynamic-call-3",
            principal="risk-agent",
            tool_name="risk.score",
        ),
        tool,
    ) == 30
