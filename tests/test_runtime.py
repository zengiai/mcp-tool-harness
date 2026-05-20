from __future__ import annotations

import asyncio
from logging import log

import pytest

from mcp_tool_harness.server import (
    CircuitOpenError,
    IdempotencyConflictError,
    RateLimitExceededError,
    ToolInputValidationError,
    ToolExecutionError,
    ToolGateway,
    ToolTimeoutError,
)

def test_gateway_enforces_rate_limit(time_func) -> None:
    calls = []

    def echo(text):
        calls.append(text)
        return {"text": text}

    gateway = ToolGateway(default_rate_limit_per_minute=1, time_func=time_func)
    gateway.register_tool("echo", echo)

    assert gateway.invoke("echo", {"text": "first"}, principal="agent-a").result == {"text": "first"}

    print("after first invoke, calls =", calls)

    with pytest.raises(RateLimitExceededError):
        gateway.invoke("echo", {"text": "second"}, principal="agent-a")

    print("after second invoke, calls =", calls)

    assert calls == ["first"]


def test_gateway_validates_input_schema_before_execution() -> None:
    gateway = ToolGateway(default_rate_limit_per_minute=None)
    calls = {"count": 0}

    def add(left: int, right: int) -> dict[str, int]:
        calls["count"] += 1
        return {"value": left + right}

    gateway.register_tool(
        "math.add",
        add,
        input_schema={
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
        },
    )

    with pytest.raises(ToolInputValidationError, match="missing required arguments"):
        gateway.invoke("math.add", {"left": 1})
    with pytest.raises(ToolInputValidationError, match="argument left must be integer"):
        gateway.invoke("math.add", {"left": "1", "right": 2})

    assert calls["count"] == 0


def test_gateway_enforces_async_tool_timeout() -> None:
    gateway = ToolGateway(default_rate_limit_per_minute=None)

    async def slow() -> dict[str, str]:
        await asyncio.sleep(0.05)
        return {"state": "done"}

    gateway.register_tool("slow", slow, timeout_ms=10)

    with pytest.raises(ToolTimeoutError):
        gateway.invoke("slow")


def test_gateway_opens_circuit_after_repeated_failures(time_func) -> None:
    gateway = ToolGateway(
        default_rate_limit_per_minute=None,
        circuit_failure_threshold=2,
        circuit_recovery_seconds=60,
        time_func=time_func,
    )
    attempts = {"count": 0}

    def unstable() -> None:
        attempts["count"] += 1
        raise RuntimeError("backend unavailable")

    gateway.register_tool("unstable", unstable)

    with pytest.raises(ToolExecutionError):
        gateway.invoke("unstable")
    with pytest.raises(ToolExecutionError):
        gateway.invoke("unstable")
    with pytest.raises(CircuitOpenError):
        gateway.invoke("unstable")

    assert attempts["count"] == 2


def test_gateway_allows_call_after_circuit_recovery_window(clock, time_func) -> None:
    gateway = ToolGateway(
        default_rate_limit_per_minute=None,
        circuit_failure_threshold=1,
        circuit_recovery_seconds=60,
        time_func=time_func,
    )
    fail = {"value": True}

    def sometimes_ok() -> dict[str, str]:
        if fail["value"]:
            raise RuntimeError("temporary failure")
        return {"state": "ok"}

    gateway.register_tool("sometimes.ok", sometimes_ok)

    with pytest.raises(ToolExecutionError):
        gateway.invoke("sometimes.ok")
    with pytest.raises(CircuitOpenError):
        gateway.invoke("sometimes.ok")

    clock["now"] = 61.0
    fail["value"] = False

    assert gateway.invoke("sometimes.ok").result == {"state": "ok"}


def test_gateway_reuses_successful_idempotent_result() -> None:
    gateway = ToolGateway(default_rate_limit_per_minute=None)
    counter = {"value": 0}

    def create_order(sku: str) -> dict[str, object]:
        counter["value"] += 1
        return {"sku": sku, "sequence": counter["value"]}

    gateway.register_tool("order.create", create_order, idempotent=True)

    first = gateway.invoke("order.create", {"sku": "SKU-1"}, idempotency_key="idem-1")
    second = gateway.invoke("order.create", {"sku": "SKU-1"}, idempotency_key="idem-1")

    assert first.result == {"sku": "SKU-1", "sequence": 1}
    assert second.result == first.result
    assert second.cached is True
    assert counter["value"] == 1


def test_gateway_rejects_idempotency_key_reuse_with_different_arguments() -> None:
    gateway = ToolGateway(default_rate_limit_per_minute=None)
    gateway.register_tool("order.create", lambda sku: {"sku": sku})

    gateway.invoke("order.create", {"sku": "SKU-1"}, idempotency_key="idem-1")

    with pytest.raises(IdempotencyConflictError):
        gateway.invoke("order.create", {"sku": "SKU-2"}, idempotency_key="idem-1")
