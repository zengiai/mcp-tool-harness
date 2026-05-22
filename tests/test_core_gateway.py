from __future__ import annotations

from typing import Any

import pytest

from mcp_tool_harness.core import Registry, ToolCallContext, ToolSpec
from mcp_tool_harness.core.gateway import ToolGateway
from mcp_tool_harness.mcp.client import InMemoryTransport, MCPClient
from mcp_tool_harness.storage import InMemoryAuditRepository


@pytest.mark.asyncio
async def test_core_gateway_invokes_registered_mcp_tool() -> None:
    registry = Registry()
    await registry.register_tool(
        ToolSpec(
            name="math.add",
            description="Add two integers",
            input_schema={
                "type": "object",
                "properties": {
                    "left": {"type": "integer"},
                    "right": {"type": "integer"},
                },
                "required": ["left", "right"],
            },
        )
    )

    transport = InMemoryTransport()
    transport.add_tool("math.add", lambda args: {"value": args["left"] + args["right"]})
    gateway = ToolGateway(
        registry=registry,
        security=None,
        mcp_client=MCPClient.with_mock(transport),
    )
    context = ToolCallContext(
        request_id="call-1",
        principal="agent-a",
        tool_name="math.add",
        trace_id="trace-1",
    )

    result = await gateway.invoke("math.add", {"left": 2, "right": 3}, context)

    assert result.success is True
    assert result.output == {"value": 5}


@pytest.mark.asyncio
async def test_core_gateway_rejects_invalid_arguments_before_mcp_call() -> None:
    registry = Registry()
    await registry.register_tool(
        ToolSpec(
            name="math.add",
            description="Add two integers",
            input_schema={
                "type": "object",
                "properties": {"left": {"type": "integer"}},
                "required": ["left"],
            },
        )
    )
    transport = InMemoryTransport()
    gateway = ToolGateway(
        registry=registry,
        security=None,
        mcp_client=MCPClient.with_mock(transport),
    )
    context = ToolCallContext(request_id="call-2", principal="agent-a", tool_name="math.add")

    result = await gateway.invoke("math.add", {}, context)

    assert result.success is False
    assert result.error_code == "ToolInputValidationError"
    assert transport.requests == []


@pytest.mark.asyncio
async def test_core_gateway_records_tool_call_to_audit_repository() -> None:
    registry = Registry()
    tool = await registry.register_tool(
        ToolSpec(
            name="math.add",
            description="Add two integers",
            input_schema={
                "type": "object",
                "properties": {
                    "left": {"type": "integer"},
                    "right": {"type": "integer"},
                },
                "required": ["left", "right"],
            },
        )
    )

    transport = InMemoryTransport()
    transport.add_tool("math.add", lambda args: {"value": args["left"] + args["right"]})
    audit = InMemoryAuditRepository()
    gateway = ToolGateway(
        registry=registry,
        security=None,
        mcp_client=MCPClient.with_mock(transport),
        audit=audit,
    )
    context = ToolCallContext(
        request_id="call-audit-1",
        principal="agent-a",
        tool_name="math.add",
        trace_id="trace-audit-1",
        metadata={
            "run_id": "run-1",
            "tool_call_id": "tool-call-1",
            "round_index": 1,
            "step_index": 1,
        },
    )

    result = await gateway.invoke("math.add", {"left": 2, "right": 3}, context)

    assert result.success is True
    records = await audit.list_records(request_id="call-audit-1")
    assert len(records) == 1
    record = records[0]
    assert record.tool_id == tool.tool_id
    assert record.context.trace_id == "trace-audit-1"
    assert record.result is not None
    assert record.result.output == {"value": 5}
    assert record.metadata["run_id"] == "run-1"
    assert record.metadata["tool_call_id"] == "tool-call-1"
    assert record.metadata["arguments"] == {"left": 2, "right": 3}


@pytest.mark.asyncio
async def test_core_gateway_does_not_hide_registry_identity_failures() -> None:
    class BrokenIdentityRegistry:
        async def get_tool_by_identity(self, *_args: Any) -> ToolSpec:
            raise RuntimeError("registry backend unavailable")

        async def get_tool(self, *_args: Any, **_kwargs: Any) -> ToolSpec:
            return ToolSpec(name="math.add", description="Fallback should not run")

    transport = InMemoryTransport()
    gateway = ToolGateway(
        registry=BrokenIdentityRegistry(),
        security=None,
        mcp_client=MCPClient.with_mock(transport),
    )
    context = ToolCallContext(request_id="call-3", principal="agent-a", tool_name="math.add")

    with pytest.raises(RuntimeError, match="registry backend unavailable"):
        await gateway.invoke("math.add", {}, context)

    assert transport.requests == []
