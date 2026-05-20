from __future__ import annotations

import pytest

from mcp_tool_harness.server.registry_api import ToolRegistry


def test_registry_registers_and_lists_tool_metadata() -> None:
    registry = ToolRegistry()

    def echo(text: str) -> dict[str, str]:
        return {"text": text}

    registered = registry.register(
        "echo",
        echo,
        description="Echo input text",
        input_schema={"type": "object"},
        permissions=("tool:echo",),
        idempotent=True,
        rate_limit_per_minute=10,
        tags=("test",),
    )

    assert registered.metadata.name == "echo"
    assert registry.get("echo").handler is echo
    assert registry.names() == ["echo"]
    assert registry.list()[0].to_dict() == {
        "name": "echo",
        "description": "Echo input text",
        "input_schema": {"type": "object"},
        "output_schema": {},
        "permissions": ("tool:echo",),
        "idempotent": True,
        "rate_limit_per_minute": 10,
        "timeout_ms": None,
        "tags": ("test",),
    }


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = ToolRegistry()
    registry.register("echo", lambda: "ok")

    with pytest.raises(ValueError, match="already registered"):
        registry.register("echo", lambda: "again")


def test_registry_rejects_invalid_handlers() -> None:
    registry = ToolRegistry()

    with pytest.raises(TypeError, match="callable"):
        registry.register("bad", "not-callable")  # type: ignore[arg-type]


def test_expected_core_gateway_contract_or_main_thread_followup() -> None:
    gateway_module = pytest.importorskip(
        "mcp_tool_harness.core.gateway",
        reason="core.gateway is not implemented yet; main thread should provide a compatible ToolGateway",
        exc_type=ImportError,
    )

    assert hasattr(gateway_module, "ToolGateway")
