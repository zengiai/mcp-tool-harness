from __future__ import annotations

import json
from pathlib import Path
import asyncio
from typing import Any

import pytest

from mcp_tool_harness.core import (
    Registry,
    ToolCallContext,
    ToolPolicy,
    ToolSpec,
)
from mcp_tool_harness.core.gateway import ToolGateway
from mcp_tool_harness.mcp.client import InMemoryTransport, MCPClient
from mcp_tool_harness.runtime import InMemoryIdempotencyStore
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
async def test_core_gateway_writes_default_json_audit_file(audit_log_path: Path) -> None:
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
    gateway = ToolGateway(
        registry=registry,
        security=None,
        mcp_client=MCPClient.with_mock(transport),
    )
    context = ToolCallContext(
        request_id="call-json-audit-1",
        principal="agent-a",
        tool_name="math.add",
        tenant_id="tenant-a",
        trace_id="trace-json-audit-1",
        metadata={"run_id": "run-json-1"},
    )

    result = await gateway.invoke("math.add", {"left": 2, "right": 3}, context)

    assert result.success is True
    lines = audit_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "tool_call"
    assert event["actor"] == "agent-a"
    assert event["action"] == "math.add"
    assert event["resource"] == "local/math.add"
    assert event["outcome"] == "success"
    assert event["request_id"] == "call-json-audit-1"
    assert event["correlation_id"] == "trace-json-audit-1"
    metadata = event["metadata"]
    assert metadata["schema_version"] == "tool_call_audit.v1"
    assert metadata["request_id"] == "call-json-audit-1"
    assert metadata["trace_id"] == "trace-json-audit-1"
    assert metadata["principal"] == "agent-a"
    assert metadata["tenant_id"] == "tenant-a"
    assert metadata["server_id"] == "local"
    assert metadata["tool_name"] == "math.add"
    assert metadata["tool_id"] == tool.tool_id
    assert metadata["status"] == "succeeded"
    assert metadata["error_code"] is None
    assert metadata["result_success"] is True
    assert metadata["result_status"] == "succeeded"
    assert metadata["arguments"] == {"left": 2, "right": 3}
    assert metadata["context_metadata"] == {"run_id": "run-json-1"}


@pytest.mark.asyncio
async def test_core_gateway_writes_default_json_metrics_file(metrics_log_path: Path) -> None:
    registry = Registry()
    await registry.register_tool(ToolSpec(name="math.add", description="Add two integers"))
    transport = InMemoryTransport()
    transport.add_tool("math.add", lambda args: {"value": args["left"] + args["right"]})
    gateway = ToolGateway(
        registry=registry,
        security=None,
        mcp_client=MCPClient.with_mock(transport),
    )
    context = ToolCallContext(
        request_id="call-json-metrics-1",
        principal="agent-a",
        tool_name="math.add",
        trace_id="trace-json-metrics-1",
    )

    result = await gateway.invoke("math.add", {"left": 2, "right": 3}, context)

    assert result.success is True
    lines = metrics_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["schema_version"] == "tool_metrics.v1"
    assert event["event_type"] == "tool_call_metrics"
    assert event["metric_name"] == "tool_call"
    assert event["tool_name"] == "math.add"
    assert event["status"] == "succeeded"
    assert event["latency_ms"] >= 0
    assert event["labels"] == {"status": "succeeded", "tool_name": "math.add"}
    assert event["counters"] == {"tool_call_total": 1}
    assert event["histograms"]["tool_call_latency_ms"] >= 0


@pytest.mark.asyncio
async def test_core_gateway_respects_tool_policy_audit_disabled() -> None:
    registry = Registry()
    await registry.register_tool(
        ToolSpec(name="math.add", description="Add two integers"),
        policy=ToolPolicy(
            tool_name="math.add",
            allowed_agents=frozenset({"*"}),
            audit_enabled=False,
        ),
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
        request_id="call-audit-disabled-1",
        principal="agent-a",
        tool_name="math.add",
    )

    result = await gateway.invoke("math.add", {"left": 2, "right": 3}, context)

    assert result.success is True
    assert result.output == {"value": 5}
    assert await audit.list_records(request_id="call-audit-disabled-1") == []


@pytest.mark.asyncio
async def test_core_gateway_audit_disabled_suppresses_validation_failure_record() -> None:
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
        ),
        policy=ToolPolicy(
            tool_name="math.add",
            allowed_agents=frozenset({"*"}),
            audit_enabled=False,
        ),
    )
    transport = InMemoryTransport()
    audit = InMemoryAuditRepository()
    gateway = ToolGateway(
        registry=registry,
        security=None,
        mcp_client=MCPClient.with_mock(transport),
        audit=audit,
    )
    context = ToolCallContext(
        request_id="call-audit-disabled-validation-1",
        principal="agent-a",
        tool_name="math.add",
    )

    result = await gateway.invoke("math.add", {}, context)

    assert result.success is False
    assert result.error_code == "ToolInputValidationError"
    assert transport.requests == []
    assert await audit.list_records(request_id="call-audit-disabled-validation-1") == []


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


@pytest.mark.asyncio
async def test_core_gateway_replays_completed_result_for_same_tool_arguments() -> None:
    registry = Registry()
    await registry.register_tool(
        ToolSpec(
            name="order.create",
            description="Create order",
            input_schema={
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
            },
        )
    )
    calls: list[str] = []
    transport = InMemoryTransport()
    transport.add_tool(
        "order.create",
        lambda args: calls.append(args["sku"]) or {"sku": args["sku"], "sequence": len(calls)},
    )
    gateway = ToolGateway(
        registry=registry,
        security=None,
        mcp_client=MCPClient.with_mock(transport),
        idempotency_store=InMemoryIdempotencyStore(default_ttl=60),
    )

    first = await gateway.invoke(
        "order.create",
        {"sku": "SKU-1"},
        ToolCallContext(request_id="dedupe-1", principal="agent-a", tool_name="order.create"),
    )
    second = await gateway.invoke(
        "order.create",
        {"sku": "SKU-1"},
        ToolCallContext(request_id="dedupe-2", principal="agent-a", tool_name="order.create"),
    )

    assert first.success is True
    assert second.success is True
    assert second.output == first.output
    assert second.metadata["cached"] is True
    assert second.metadata["idempotency_reason"] == "completed_replay"
    assert calls == ["SKU-1"]


@pytest.mark.asyncio
async def test_core_gateway_rejects_explicit_key_reuse_with_different_arguments() -> None:
    registry = Registry()
    await registry.register_tool(
        ToolSpec(
            name="order.create",
            description="Create order",
            input_schema={
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
            },
        )
    )
    calls: list[str] = []
    transport = InMemoryTransport()
    transport.add_tool("order.create", lambda args: calls.append(args["sku"]) or {"sku": args["sku"]})
    gateway = ToolGateway(
        registry=registry,
        security=None,
        mcp_client=MCPClient.with_mock(transport),
        idempotency_store=InMemoryIdempotencyStore(default_ttl=60),
    )

    first = await gateway.invoke(
        "order.create",
        {"sku": "SKU-1"},
        ToolCallContext(
            request_id="conflict-1",
            principal="agent-a",
            tool_name="order.create",
            idempotency_key="manual-key-1",
        ),
    )
    second = await gateway.invoke(
        "order.create",
        {"sku": "SKU-2"},
        ToolCallContext(
            request_id="conflict-2",
            principal="agent-a",
            tool_name="order.create",
            idempotency_key="manual-key-1",
        ),
    )

    assert first.success is True
    assert second.success is False
    assert second.error_code == "IDEMPOTENCY_CONFLICT"
    assert second.metadata["idempotency_reason"] == "fingerprint_mismatch"
    assert calls == ["SKU-1"]


@pytest.mark.asyncio
async def test_core_gateway_rejects_duplicate_while_first_call_is_in_progress() -> None:
    class SlowMCPClient:
        def __init__(self) -> None:
            self.calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def call_tool(self, _name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return {"structuredContent": {"sku": arguments["sku"], "sequence": self.calls}}

    registry = Registry()
    await registry.register_tool(
        ToolSpec(
            name="order.create",
            description="Create order",
            input_schema={
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
            },
        )
    )
    client = SlowMCPClient()
    gateway = ToolGateway(
        registry=registry,
        security=None,
        mcp_client=client,
        idempotency_store=InMemoryIdempotencyStore(default_ttl=60),
    )
    first_context = ToolCallContext(
        request_id="in-progress-1",
        principal="agent-a",
        tool_name="order.create",
    )
    second_context = ToolCallContext(
        request_id="in-progress-2",
        principal="agent-a",
        tool_name="order.create",
    )

    first_task = asyncio.create_task(gateway.invoke("order.create", {"sku": "SKU-1"}, first_context))
    await client.started.wait()
    second = await gateway.invoke("order.create", {"sku": "SKU-1"}, second_context)
    client.release.set()
    first = await first_task

    assert first.success is True
    assert second.success is False
    assert second.error_code == "IDEMPOTENCY_IN_PROGRESS"
    assert client.calls == 1


@pytest.mark.asyncio
async def test_core_gateway_allows_failed_duplicate_retries_until_attempt_cap() -> None:
    class FailingMCPClient:
        def __init__(self) -> None:
            self.calls = 0

        async def call_tool(self, _name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            raise RuntimeError("temporary tool failure")

    registry = Registry()
    await registry.register_tool(
        ToolSpec(
            name="order.create",
            description="Create order",
            input_schema={
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
            },
        )
    )
    client = FailingMCPClient()
    gateway = ToolGateway(
        registry=registry,
        security=None,
        mcp_client=client,
        idempotency_store=InMemoryIdempotencyStore(default_ttl=60, max_attempts=2),
    )

    first = await gateway.invoke(
        "order.create",
        {"sku": "SKU-1"},
        ToolCallContext(request_id="retry-1", principal="agent-a", tool_name="order.create"),
    )
    second = await gateway.invoke(
        "order.create",
        {"sku": "SKU-1"},
        ToolCallContext(request_id="retry-2", principal="agent-a", tool_name="order.create"),
    )
    third = await gateway.invoke(
        "order.create",
        {"sku": "SKU-1"},
        ToolCallContext(request_id="retry-3", principal="agent-a", tool_name="order.create"),
    )

    assert first.error_code == "RuntimeError"
    assert second.error_code == "RuntimeError"
    assert third.success is False
    assert third.error_code == "IDEMPOTENCY_RETRY_EXHAUSTED"
    assert third.metadata["idempotency_attempt_count"] == 2
    assert client.calls == 2
