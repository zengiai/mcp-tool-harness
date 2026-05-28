from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


Labels = tuple[tuple[str, str], ...]
METRICS_LOG_PATH_ENV = "MCP_TOOL_HARNESS_METRICS_LOG_PATH"
DEFAULT_METRICS_LOG_PATH = Path("logs/tool-metrics.jsonl")


def _labels(labels: Mapping[str, Any] | None = None) -> Labels:
    if not labels:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


@dataclass(frozen=True)
class MetricPoint:
    name: str
    value: float
    labels: Labels
    timestamp: float


@dataclass
class HistogramState:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "sum": self.total,
            "min": self.minimum,
            "max": self.maximum,
            "avg": None if self.count == 0 else self.total / self.count,
        }


class InMemoryMetrics:
    def __init__(self) -> None:
        self._counters: dict[tuple[str, Labels], MetricPoint] = {}
        self._gauges: dict[tuple[str, Labels], MetricPoint] = {}
        self._histograms: dict[tuple[str, Labels], HistogramState] = {}
        self._histogram_updated_at: dict[tuple[str, Labels], float] = {}
        self._lock = threading.RLock()

    def increment(self, name: str, value: float = 1.0, labels: Mapping[str, Any] | None = None) -> None:
        if value < 0:
            raise ValueError("counter increment must be non-negative")
        key = (name, _labels(labels))
        now = time.time()
        with self._lock:
            current = self._counters.get(key)
            next_value = value if current is None else current.value + value
            self._counters[key] = MetricPoint(name, next_value, key[1], now)

    def set_gauge(self, name: str, value: float, labels: Mapping[str, Any] | None = None) -> None:
        key = (name, _labels(labels))
        with self._lock:
            self._gauges[key] = MetricPoint(name, float(value), key[1], time.time())

    def observe(self, name: str, value: float, labels: Mapping[str, Any] | None = None) -> None:
        key = (name, _labels(labels))
        with self._lock:
            state = self._histograms.setdefault(key, HistogramState())
            state.observe(float(value))
            self._histogram_updated_at[key] = time.time()

    @contextmanager
    def timer(self, name: str, labels: Mapping[str, Any] | None = None):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - start, labels)

    @asynccontextmanager
    async def async_timer(self, name: str, labels: Mapping[str, Any] | None = None):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - start, labels)

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            counters = [self._point_to_dict(point) for point in self._counters.values()]
            gauges = [self._point_to_dict(point) for point in self._gauges.values()]
            histograms = [
                {
                    "name": name,
                    "labels": dict(labels),
                    "timestamp": self._histogram_updated_at.get((name, labels)),
                    **state.to_dict(),
                }
                for (name, labels), state in self._histograms.items()
            ]
        return {"counters": counters, "gauges": gauges, "histograms": histograms}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._histogram_updated_at.clear()

    @staticmethod
    def _point_to_dict(point: MetricPoint) -> dict[str, Any]:
        return {
            "name": point.name,
            "value": point.value,
            "labels": dict(point.labels),
            "timestamp": point.timestamp,
        }


class JsonLinesMetricsRecorder:
    """Append-only JSON Lines metrics recorder used as the Gateway default."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = default_metrics_log_path() if path is None else Path(path)
        self._lock = asyncio.Lock()

    async def record_tool_call(self, tool_name: str, status: str, latency_ms: int | float) -> None:
        latency = float(latency_ms)
        event = {
            "schema_version": "tool_metrics.v1",
            "event_type": "tool_call_metrics",
            "metric_name": "tool_call",
            "timestamp": time.time(),
            "tool_name": str(tool_name),
            "status": str(status),
            "latency_ms": latency,
            "labels": {
                "tool_name": str(tool_name),
                "status": str(status),
            },
            "counters": {
                "tool_call_total": 1,
            },
            "histograms": {
                "tool_call_latency_ms": latency,
            },
        }
        await self.write_event(event)

    async def write_event(self, event: Mapping[str, Any]) -> None:
        line = json.dumps(_json_safe(dict(event)), ensure_ascii=False, sort_keys=True) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def default_metrics_log_path() -> Path:
    configured = os.environ.get(METRICS_LOG_PATH_ENV)
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_METRICS_LOG_PATH


def create_default_metrics_recorder(path: str | Path | None = None) -> JsonLinesMetricsRecorder:
    return JsonLinesMetricsRecorder(path)


def read_metrics_events(path: str | Path | None = None, *, limit: int = 1_000) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    source = default_metrics_log_path() if path is None else Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except OSError:
        return []

    events: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def summarize_tool_call_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    by_tool: dict[str, dict[str, Any]] = {}
    latencies: list[float] = []

    for event in events:
        if event.get("event_type") != "tool_call_metrics":
            continue
        tool_name = str(event.get("tool_name") or "unknown")
        status = str(event.get("status") or "unknown")
        latency = _coerce_float(event.get("latency_ms"))

        status_counts[status] = status_counts.get(status, 0) + 1
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        tool_state = by_tool.setdefault(
            tool_name,
            {
                "tool_name": tool_name,
                "count": 0,
                "status_counts": {},
                "latency_ms": {
                    "count": 0,
                    "sum": 0.0,
                    "min": None,
                    "max": None,
                    "avg": None,
                },
            },
        )
        tool_state["count"] += 1
        tool_state["status_counts"][status] = tool_state["status_counts"].get(status, 0) + 1
        if latency is not None:
            latencies.append(latency)
            _observe_latency(tool_state["latency_ms"], latency)

    for tool_state in by_tool.values():
        latency_state = tool_state["latency_ms"]
        count = latency_state["count"]
        latency_state["avg"] = None if count == 0 else latency_state["sum"] / count

    return {
        "total_calls": sum(status_counts.values()),
        "status_counts": status_counts,
        "tool_counts": tool_counts,
        "latency_ms": _latency_summary(latencies),
        "by_tool": sorted(by_tool.values(), key=lambda item: (-item["count"], item["tool_name"])),
    }


def _observe_latency(state: dict[str, Any], latency: float) -> None:
    state["count"] += 1
    state["sum"] += latency
    state["min"] = latency if state["min"] is None else min(state["min"], latency)
    state["max"] = latency if state["max"] is None else max(state["max"], latency)


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "sum": 0.0, "min": None, "max": None, "avg": None}
    total = sum(values)
    return {
        "count": len(values),
        "sum": total,
        "min": min(values),
        "max": max(values),
        "avg": total / len(values),
    }


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_metrics = InMemoryMetrics()


def get_metrics() -> InMemoryMetrics:
    return _metrics


__all__ = [
    "DEFAULT_METRICS_LOG_PATH",
    "HistogramState",
    "InMemoryMetrics",
    "JsonLinesMetricsRecorder",
    "METRICS_LOG_PATH_ENV",
    "MetricPoint",
    "create_default_metrics_recorder",
    "default_metrics_log_path",
    "get_metrics",
    "read_metrics_events",
    "summarize_tool_call_metrics",
]
