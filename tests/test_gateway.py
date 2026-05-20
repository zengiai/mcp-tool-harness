from __future__ import annotations

import pytest

from mcp_tool_harness.server import (
    ApprovalPolicy,
    ApprovalRequiredError,
    PermissionDeniedError,
    ToolGateway,
)
from mcp_tool_harness.server import api


def test_gateway_invokes_registered_tool_successfully() -> None:
    gateway = ToolGateway(default_rate_limit_per_minute=None)
    gateway.register_tool("math.add", lambda left, right: {"value": left + right})

    response = gateway.invoke("math.add", {"left": 2, "right": 3}, principal="agent-a")

    assert response.status == "success"
    assert response.result == {"value": 5}
    assert response.cached is False


def test_gateway_rejects_denied_tool_before_execution() -> None:
    called = {"value": False}
    policy = ApprovalPolicy(denied_tools=("danger.*",))
    gateway = ToolGateway(approval_policy=policy, default_rate_limit_per_minute=None)

    def dangerous() -> str:
        called["value"] = True
        return "should-not-run"

    gateway.register_tool("danger.delete", dangerous)

    with pytest.raises(PermissionDeniedError, match="not allowed"):
        gateway.invoke("danger.delete", principal="agent-a")

    assert called["value"] is False


def test_gateway_reports_human_approval_requirement() -> None:
    policy = ApprovalPolicy(
        approval_required_tools=("payment.refund",),
        approval_id_factory=lambda: "approval-001",
    )
    gateway = ToolGateway(approval_policy=policy, default_rate_limit_per_minute=None)
    gateway.register_tool("payment.refund", lambda order_id: {"order_id": order_id})

    with pytest.raises(ApprovalRequiredError) as exc:
        gateway.invoke("payment.refund", {"order_id": "O-1"}, principal="agent-a")

    assert exc.value.approval_id == "approval-001"
    assert "requires human approval" in str(exc.value)


def test_fastapi_is_optional_and_create_app_error_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "FastAPI", None)
    monkeypatch.setattr(api, "_FASTAPI_IMPORT_ERROR", ModuleNotFoundError("No module named 'fastapi'"))

    with pytest.raises(RuntimeError, match="FastAPI is optional"):
        api.create_app()
