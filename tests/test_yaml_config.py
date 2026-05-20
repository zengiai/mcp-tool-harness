from __future__ import annotations

import pytest

from mcp_tool_harness.config import (
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
