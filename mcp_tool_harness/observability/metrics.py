from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, Mapping


Labels = tuple[tuple[str, str], ...]


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


_metrics = InMemoryMetrics()


def get_metrics() -> InMemoryMetrics:
    return _metrics


__all__ = ["HistogramState", "InMemoryMetrics", "MetricPoint", "get_metrics"]
