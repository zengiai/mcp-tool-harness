from __future__ import annotations

import contextvars
import secrets
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Mapping


def _hex(bytes_count: int) -> str:
    return secrets.token_hex(bytes_count)


@dataclass(frozen=True)
class SpanContext:
    trace_id: str
    span_id: str
    parent_span_id: str | None = None


@dataclass
class Span:
    name: str
    context: SpanContext
    attributes: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: str = "ok"
    error: str | None = None

    def finish(self, *, status: str = "ok", error: BaseException | str | None = None) -> None:
        self.end_time = time.time()
        self.status = status
        if error is not None:
            self.error = str(error)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "attributes": dict(self.attributes),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": None if self.end_time is None else self.end_time - self.start_time,
            "status": self.status,
            "error": self.error,
        }


_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar("mcp_current_span", default=None)


class InMemorySpanExporter:
    def __init__(self, max_spans: int = 10_000) -> None:
        if max_spans <= 0:
            raise ValueError("max_spans must be positive")
        self.max_spans = max_spans
        self._spans: list[Span] = []

    def export(self, span: Span) -> None:
        self._spans.append(span)
        overflow = len(self._spans) - self.max_spans
        if overflow > 0:
            del self._spans[:overflow]

    def snapshot(self) -> list[dict[str, Any]]:
        return [span.to_dict() for span in self._spans]

    def clear(self) -> None:
        self._spans.clear()


class Tracer:
    def __init__(self, exporter: InMemorySpanExporter | None = None) -> None:
        self.exporter = exporter or InMemorySpanExporter()

    def current_span(self) -> Span | None:
        return _current_span.get()

    def start_span(self, name: str, attributes: Mapping[str, Any] | None = None) -> Span:
        parent = self.current_span()
        context = SpanContext(
            trace_id=parent.context.trace_id if parent else _hex(16),
            span_id=_hex(8),
            parent_span_id=parent.context.span_id if parent else None,
        )
        return Span(name=name, context=context, attributes=dict(attributes or {}))

    @contextmanager
    def span(self, name: str, attributes: Mapping[str, Any] | None = None):
        span = self.start_span(name, attributes)
        token = _current_span.set(span)
        try:
            yield span
            span.finish(status="ok")
        except Exception as exc:
            span.finish(status="error", error=exc)
            raise
        finally:
            _current_span.reset(token)
            self.exporter.export(span)

    @asynccontextmanager
    async def async_span(self, name: str, attributes: Mapping[str, Any] | None = None):
        span = self.start_span(name, attributes)
        token = _current_span.set(span)
        try:
            yield span
            span.finish(status="ok")
        except Exception as exc:
            span.finish(status="error", error=exc)
            raise
        finally:
            _current_span.reset(token)
            self.exporter.export(span)


_tracer = Tracer()


def get_tracer() -> Tracer:
    return _tracer


def current_span() -> Span | None:
    return _tracer.current_span()


__all__ = [
    "InMemorySpanExporter",
    "Span",
    "SpanContext",
    "Tracer",
    "current_span",
    "get_tracer",
]
