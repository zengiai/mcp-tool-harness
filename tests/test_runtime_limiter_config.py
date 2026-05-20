from __future__ import annotations

import pytest

from mcp_tool_harness.runtime.limiter import MultiDimensionalRateLimiter, RateLimitRule


@pytest.mark.asyncio
async def test_multi_dimensional_rules_are_all_enforced_without_partial_consume() -> None:
    limiter = MultiDimensionalRateLimiter(
        rules=[
            RateLimitRule(dimension="tenant", capacity=2, refill_rate=0.01),
            RateLimitRule(dimension="agent_tool", capacity=1, refill_rate=0.01),
        ]
    )

    first = await limiter.acquire(tenant_id="tenant-a", agent_id="agent-a", tool_name="order.create")
    second = await limiter.acquire(tenant_id="tenant-a", agent_id="agent-a", tool_name="order.create")
    third = await limiter.acquire(tenant_id="tenant-a", agent_id="agent-b", tool_name="order.create")
    fourth = await limiter.acquire(tenant_id="tenant-a", agent_id="agent-c", tool_name="order.create")

    assert first.allowed is True
    assert second.allowed is False
    assert second.dimension == "agent_tool"
    assert second.key == "agent:agent-a:tool:order.create"
    assert third.allowed is True
    assert fourth.allowed is False
    assert fourth.dimension == "tenant"
    assert fourth.key == "tenant:tenant-a"


@pytest.mark.asyncio
async def test_custom_key_rule_uses_safe_template_fields() -> None:
    limiter = MultiDimensionalRateLimiter(
        rules=[
            {
                "dimension": "custom",
                "capacity": 1,
                "refill_rate": 0.01,
                "key_template": "tenant:{tenant_id}:agent:{agent_id}:tool:{tool_name}:arg:{args.order_id}",
            }
        ]
    )

    first = await limiter.acquire(
        tenant_id="tenant-a",
        agent_id="agent-a",
        tool_name="payment.refund",
        args={"order_id": "order-1"},
    )
    second = await limiter.acquire(
        tenant_id="tenant-a",
        agent_id="agent-a",
        tool_name="payment.refund",
        args={"order_id": "order-1"},
    )
    different_order = await limiter.acquire(
        tenant_id="tenant-a",
        agent_id="agent-a",
        tool_name="payment.refund",
        args={"order_id": "order-2"},
    )

    assert first.allowed is True
    assert second.allowed is False
    assert second.dimension == "custom"
    assert second.key == "tenant:tenant-a:agent:agent-a:tool:payment.refund:arg:order-1"
    assert different_order.allowed is True


@pytest.mark.asyncio
async def test_dynamic_rule_replacement_takes_precedence_over_code_defaults() -> None:
    limiter = MultiDimensionalRateLimiter(
        default_rules=[RateLimitRule(dimension="tool", capacity=1, refill_rate=0.01)],
        rules=[RateLimitRule(dimension="tool", capacity=2, refill_rate=0.01)],
    )

    assert (await limiter.acquire(tool_name="search.query")).allowed is True
    assert (await limiter.acquire(tool_name="search.query")).allowed is True
    assert (await limiter.acquire(tool_name="search.query")).allowed is False

    await limiter.replace_rules([RateLimitRule(dimension="tool", capacity=3, refill_rate=0.01)])

    assert (await limiter.acquire(tool_name="search.query")).allowed is True
    assert (await limiter.acquire(tool_name="search.query")).allowed is True
    assert (await limiter.acquire(tool_name="search.query")).allowed is True
    rejected = await limiter.acquire(tool_name="search.query")

    assert rejected.allowed is False
    assert rejected.dimension == "tool"
    assert rejected.key == "tool:search.query"
