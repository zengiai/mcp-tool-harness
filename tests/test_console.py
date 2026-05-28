from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_tool_harness.core import Registry as CoreRegistry
from mcp_tool_harness.core import ToolSpec
from mcp_tool_harness.server import ToolGateway
from mcp_tool_harness.server.api import (
    _console_asset_media_type,
    _console_chain_detail_payload,
    _console_chains_payload,
    _console_html,
    _console_metrics_payload,
    _console_tools_payload,
    _read_console_asset,
)


@pytest.mark.asyncio
async def test_console_tools_payload_lists_current_gateway_tools() -> None:
    gateway = ToolGateway(default_rate_limit_per_minute=None)
    gateway.register_tool(
        "text.echo",
        lambda text: {"text": text},
        description="Echo text",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        tags=("read",),
        timeout_ms=100,
    )

    payload = await _console_tools_payload(gateway)

    assert payload["count"] == 1
    assert payload["tools"][0]["name"] == "text.echo"
    assert payload["tools"][0]["description"] == "Echo text"
    assert payload["tools"][0]["tags"] == ("read",)
    assert payload["tools"][0]["timeout_ms"] == 100


@pytest.mark.asyncio
async def test_console_tools_payload_can_read_core_registry() -> None:
    class GatewayWithRegistry:
        def __init__(self, registry: CoreRegistry) -> None:
            self.registry = registry

    registry = CoreRegistry()
    await registry.register_tool(ToolSpec(name="math.add", description="Add two integers"))

    payload = await _console_tools_payload(GatewayWithRegistry(registry))

    assert payload["count"] == 1
    assert payload["tools"][0]["name"] == "math.add"


def test_console_chain_payload_groups_audit_events(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    _write_jsonl(
        audit_path,
        [
            {
                "event_id": "event-1",
                "event_type": "tool_call",
                "actor": "agent-a",
                "action": "math.add",
                "resource": "local/math.add",
                "outcome": "success",
                "timestamp": 10.0,
                "correlation_id": "trace-1",
                "request_id": "request-1",
                "metadata": {
                    "status": "succeeded",
                    "tool_name": "math.add",
                    "trace_id": "trace-1",
                    "request_id": "request-1",
                },
            },
            {
                "event_id": "event-2",
                "event_type": "tool_call",
                "actor": "agent-a",
                "action": "math.multiply",
                "resource": "local/math.multiply",
                "outcome": "failure",
                "timestamp": 10.2,
                "correlation_id": "trace-1",
                "request_id": "request-2",
                "metadata": {
                    "status": "failed",
                    "tool_name": "math.multiply",
                    "trace_id": "trace-1",
                    "request_id": "request-2",
                },
            },
        ],
    )

    payload = _console_chains_payload(audit_path=audit_path)
    detail = _console_chain_detail_payload("trace-1", audit_path=audit_path)

    assert payload["count"] == 1
    chain = payload["chains"][0]
    assert chain["chain_id"] == "trace-1"
    assert chain["event_count"] == 2
    assert chain["tools"] == ["math.add", "math.multiply"]
    assert chain["status_counts"] == {"succeeded": 1, "failed": 1}
    assert detail["count"] == 2
    assert detail["events"][0]["event_id"] == "event-1"


def test_console_metrics_payload_summarizes_tool_metric_events(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {
                "schema_version": "tool_metrics.v1",
                "event_type": "tool_call_metrics",
                "metric_name": "tool_call",
                "timestamp": 10.0,
                "tool_name": "math.add",
                "status": "succeeded",
                "latency_ms": 12,
            },
            {
                "schema_version": "tool_metrics.v1",
                "event_type": "tool_call_metrics",
                "metric_name": "tool_call",
                "timestamp": 11.0,
                "tool_name": "math.add",
                "status": "failed",
                "latency_ms": 20,
            },
        ],
    )

    payload = _console_metrics_payload(metrics_path=metrics_path)

    assert payload["count"] == 2
    assert payload["summary"]["total_calls"] == 2
    assert payload["summary"]["status_counts"] == {"succeeded": 1, "failed": 1}
    assert payload["summary"]["tool_counts"] == {"math.add": 2}
    assert payload["summary"]["latency_ms"]["avg"] == 16
    assert payload["summary"]["by_tool"][0]["tool_name"] == "math.add"


def test_console_html_contains_required_tabs() -> None:
    html = _console_html()

    assert 'data-tab="tool"' in html
    assert 'data-tab="chain"' in html
    assert 'data-tab="metrics"' in html


def test_console_html_references_external_assets() -> None:
    html = _console_html()

    assert '<link rel="stylesheet" href="/console/assets/console.css">' in html
    assert '<script src="/console/assets/console.js" defer></script>' in html
    assert "<style>" not in html


def test_console_package_assets_are_loadable() -> None:
    css = _read_console_asset("console.css")
    script = _read_console_asset("console.js")

    assert ".layout" in css
    assert "loadMetrics" in script


def test_console_asset_media_types_are_allowlisted() -> None:
    assert _console_asset_media_type("console.css") == "text/css; charset=utf-8"
    assert _console_asset_media_type("console.js") == "application/javascript; charset=utf-8"
    assert _console_asset_media_type("../api.py") is None


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
