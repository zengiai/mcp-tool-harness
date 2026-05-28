from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def clock() -> dict[str, float]:
    return {"now": 0.0}


@pytest.fixture
def time_func(clock: dict[str, float]):
    return lambda: clock["now"]


@pytest.fixture(autouse=True)
def audit_log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "tool-audit.jsonl"
    monkeypatch.setenv("MCP_TOOL_HARNESS_AUDIT_LOG_PATH", str(path))
    return path


@pytest.fixture(autouse=True)
def metrics_log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "tool-metrics.jsonl"
    monkeypatch.setenv("MCP_TOOL_HARNESS_METRICS_LOG_PATH", str(path))
    return path
