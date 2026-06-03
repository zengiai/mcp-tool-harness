from __future__ import annotations

import pytest

from mcp_tool_harness.config import (
    ConfigLoadError,
    YamlConfigSource,
    apply_policy_config,
    create_mcp_client,
    discover_and_register_mcp_servers,
    load_yaml_config,
)
from mcp_tool_harness.core import PolicyAwareSecurity, Registry, ToolCallContext, ToolCallStatus, ToolSpec
from mcp_tool_harness.core.gateway import ToolGateway
from mcp_tool_harness.mcp import InMemoryTransport, MCPClient
from mcp_tool_harness.runtime import PolicyAwareRateLimiter


def test_load_yaml_config_without_global_policy_keeps_static_tool_policy_defaults() -> None:
    empty_config = load_yaml_config(
        """
        tool_harness:
          version: v1
        """
    )
    assert empty_config.policies == ()

    config = load_yaml_config(
        """
        tool_harness:
          policies:
            - tool_name: query.stats
        """
    )

    policy = config.policies[0]
    assert policy.tool_name == "query.stats"
    assert policy.server_id is None
    assert policy.allowed_agents == frozenset({"*"})
    assert policy.timeout_ms is None
    assert policy.circuit_failure_threshold == 5
    assert policy.audit_enabled is True


def test_load_yaml_config_normalizes_global_tool_policy() -> None:
    config = load_yaml_config(
        """
        tool_harness:
          version: global-v1
          global_tool_policy:
            risk_level: l1
            allowed_agents: ["*"]
            timeout_ms: 1500
            rate_limits:
              - dimension: tenant_tool
                capacity: 100
                refill_rate: 2
          policies:
            - mcp_service: payment-mcp
              tool_name: "*"
              risk_level: l2
              allowed_agents: [finance-agent]
              timeout_ms: 800
        """
    )

    assert config.version == "global-v1"
    assert len(config.policies) == 2
    global_policy = config.policies[0]
    assert global_policy.tool_name == "*"
    assert global_policy.server_id is None
    assert global_policy.risk_level.value == "l1"
    assert global_policy.timeout_ms == 1500
    assert global_policy.rate_limits[0]["dimension"] == "tenant_tool"
    assert config.policies[1].server_id == "payment-mcp"


def test_global_tool_policy_rejects_service_scope() -> None:
    with pytest.raises(ConfigLoadError, match="server_id"):
        load_yaml_config(
            """
            tool_harness:
              global_tool_policy:
                server_id: payment-mcp
                timeout_ms: 1000
            """
        )


def test_load_yaml_config_normalizes_tool_policies() -> None:
    config = load_yaml_config(
        """
        tool_harness:
          version: v1
          policies:
            - tool_name: coupon.reserve
              server_id: local
              risk_level: l1
              allowed_agents: [coupon-agent, risk-agent]
              timeout_ms: 300
              rate_limits:
                - dimension: tenant_tool
                  capacity: 100
                  refill_rate: 1.6
                - dimension: custom
                  key_template: "tenant:{tenant_id}:campaign:{args.campaign_id}"
                  capacity: 5
                  refill_rate: 0.08
        """
    )

    assert config.version == "v1"
    assert len(config.policies) == 1
    policy = config.policies[0]
    assert policy.tool_name == "coupon.reserve"
    assert policy.allowed_agents == frozenset({"coupon-agent", "risk-agent"})
    assert policy.timeout_ms == 300
    assert policy.rate_limits[1]["key_template"] == "tenant:{tenant_id}:campaign:{args.campaign_id}"


def test_load_yaml_config_normalizes_mcp_servers() -> None:
    config = load_yaml_config(
        """
        tool_harness:
          mcp_servers:
            - server_id: inventory-mcp
              transport: https
              url: https://inventory.example.com/mcp
              headers:
                Authorization: Bearer ${INVENTORY_TOKEN}
              timeout_ms: 1000
            - server_id: local-risk
              transport: stdio
              command: python
              args: ["-m", "risk_mcp_server"]
              cwd: /srv/risk
        """
    )

    assert len(config.mcp_servers) == 2
    assert config.mcp_servers[0].transport == "https"
    assert config.mcp_servers[0].endpoint == "https://inventory.example.com/mcp"
    assert config.mcp_servers[0].headers["Authorization"] == "Bearer ${INVENTORY_TOKEN}"
    assert config.mcp_servers[0].timeout_seconds == 1.0
    assert config.mcp_servers[1].command_line == ("python", "-m", "risk_mcp_server")


def test_create_mcp_client_expands_https_config_headers() -> None:
    server = load_yaml_config(
        """
        tool_harness:
          mcp_servers:
            - server_id: inventory-mcp
              transport: streamable_http
              url: https://inventory.example.com/mcp
              headers:
                Authorization: Bearer ${INVENTORY_TOKEN}
              timeout_ms: 500
        """
    ).mcp_servers[0]

    client = create_mcp_client(server, environ={"INVENTORY_TOKEN": "secret-token"})

    assert client.transport.url == "https://inventory.example.com/mcp"
    assert client.transport.headers["Authorization"] == "Bearer secret-token"
    assert client.transport.default_timeout == 0.5


@pytest.mark.asyncio
async def test_yaml_config_applies_to_registry_and_gateway(tmp_path) -> None:
    config_path = tmp_path / "tool-policy.yaml"
    config_path.write_text(
        """
        tool_harness:
          policies:
            - tool_name: coupon.reserve
              allowed_agents: [coupon-agent]
              risk_level: l1
              timeout_ms: 500
              rate_limits:
                - dimension: custom
                  key_template: "tenant:{tenant_id}:campaign:{args.campaign_id}"
                  capacity: 1
                  refill_rate: 0.001
        """,
        encoding="utf-8",
    )

    registry = Registry(cache_ttl_seconds=0)
    await registry.register_tool(ToolSpec(name="coupon.reserve", description="Reserve coupon"))
    source = YamlConfigSource(config_path)
    await source.apply_to(registry)

    transport = InMemoryTransport()
    transport.add_tool("coupon.reserve", lambda args: {"campaign_id": args["campaign_id"]})
    security = PolicyAwareSecurity(registry)
    gateway = ToolGateway(
        registry=registry,
        security=security,
        limiter=PolicyAwareRateLimiter(security=security),
        mcp_client=MCPClient.with_mock(transport),
    )

    async def invoke(request_id: str, principal: str, campaign_id: str):
        return await gateway.invoke(
            "coupon.reserve",
            {"campaign_id": campaign_id},
            ToolCallContext(
                request_id=request_id,
                principal=principal,
                tool_name="coupon.reserve",
                tenant_id="tenant-a",
            ),
        )

    assert (await invoke("yaml-1", "coupon-agent", "C-1")).success is True

    limited = await invoke("yaml-2", "coupon-agent", "C-1")
    assert limited.status is ToolCallStatus.RATE_LIMITED

    denied = await invoke("yaml-3", "other-agent", "C-2")
    assert denied.status is ToolCallStatus.DENIED

    config_path.write_text(
        """
        tool_harness:
          policies:
            - tool_name: coupon.reserve
              allowed_agents: ["*"]
              risk_level: l1
              timeout_ms: 500
              rate_limits:
                - dimension: custom
                  key_template: "tenant:{tenant_id}:campaign:{args.campaign_id}"
                  capacity: 2
                  refill_rate: 0.001
        """,
        encoding="utf-8",
    )
    await apply_policy_config(registry, await source.load())

    assert (await invoke("yaml-4", "other-agent", "C-2")).success is True


@pytest.mark.asyncio
async def test_yaml_global_tool_policy_has_lower_priority_than_service_and_tool() -> None:
    config = load_yaml_config(
        """
        tool_harness:
          global_tool_policy:
            allowed_agents: [global-agent]
            timeout_ms: 1000
            metadata:
              scope: global
          policies:
            - mcp_service: payment-mcp
              tool_name: "*"
              allowed_agents: [payment-agent]
              timeout_ms: 800
              metadata:
                scope: service
            - tool_name: catalog.search
              allowed_agents: [catalog-agent]
              timeout_ms: 700
              metadata:
                scope: tool
            - server_id: payment-mcp
              tool_name: payment.refund
              allowed_agents: [refund-agent]
              timeout_ms: 600
              metadata:
                scope: service_tool
        """
    )

    registry = Registry(cache_ttl_seconds=0)
    await registry.register_tool(ToolSpec(name="inventory.query", description="Query inventory"))
    await registry.register_tool(ToolSpec(name="catalog.search", description="Search catalog"))
    await registry.register_tool(
        ToolSpec(name="payment.capture", description="Capture payment", server_id="payment-mcp")
    )
    await registry.register_tool(
        ToolSpec(name="payment.refund", description="Refund payment", server_id="payment-mcp")
    )
    await apply_policy_config(registry, config)

    security = PolicyAwareSecurity(registry)

    async def resolved_scope(server_id: str, tool_name: str) -> str:
        tool = await registry.get_tool_by_identity(server_id, tool_name)
        policy = await security.resolve_policy(
            ToolCallContext(
                request_id=f"resolve-{server_id}-{tool_name}",
                principal="agent",
                tool_name=tool_name,
                server_id=server_id,
            ),
            tool,
        )
        assert policy is not None
        return str(policy.metadata["scope"])

    assert await resolved_scope("local", "inventory.query") == "global"
    assert await resolved_scope("local", "catalog.search") == "tool"
    assert await resolved_scope("payment-mcp", "payment.capture") == "service"
    assert await resolved_scope("payment-mcp", "payment.refund") == "service_tool"

    payment_capture = await registry.get_tool_by_identity("payment-mcp", "payment.capture")
    denied_by_service_policy = await security.check_permission(
        ToolCallContext(
            request_id="priority-denied",
            principal="global-agent",
            tool_name="payment.capture",
            server_id="payment-mcp",
        ),
        payment_capture,
        {},
    )
    assert denied_by_service_policy.effect.value == "deny"

    allowed_by_service_policy = await security.check_permission(
        ToolCallContext(
            request_id="priority-allowed",
            principal="payment-agent",
            tool_name="payment.capture",
            server_id="payment-mcp",
        ),
        payment_capture,
        {},
    )
    assert allowed_by_service_policy.effect.value == "allow"


@pytest.mark.asyncio
async def test_mcp_server_config_discovers_and_registers_tools() -> None:
    config = load_yaml_config(
        """
        tool_harness:
          mcp_servers:
            - server_id: inventory-mcp
              transport: streamable_http
              url: https://inventory.example.com/mcp
              timeout_ms: 1000
        """
    )
    registry = Registry(cache_ttl_seconds=0)
    transport = InMemoryTransport()
    transport.add_tool(
        "inventory.query",
        lambda args: {"sku_id": args["sku_id"], "available": 7},
        description="Query inventory",
        input_schema={
            "type": "object",
            "properties": {"sku_id": {"type": "string"}},
            "required": ["sku_id"],
        },
    )

    bootstrap = await discover_and_register_mcp_servers(
        registry,
        config,
        client_factory=lambda _server: MCPClient.with_mock(transport),
    )

    registered = await registry.get_tool_by_identity("inventory-mcp", "inventory.query")
    assert registered.description == "Query inventory"
    assert registered.input_schema["required"] == ["sku_id"]
    assert (await registry.get_server("inventory-mcp")).endpoint == "https://inventory.example.com/mcp"

    gateway = ToolGateway(
        registry=registry,
        security=None,
        mcp_client=bootstrap.router,
    )
    result = await gateway.invoke(
        "inventory-mcp/inventory.query",
        {"sku_id": "SKU-1001"},
        ToolCallContext(
            request_id="mcp-config-call-1",
            principal="inventory-agent",
            tool_name="inventory.query",
        ),
    )

    assert result.success is True
    assert result.output == {"sku_id": "SKU-1001", "available": 7}
