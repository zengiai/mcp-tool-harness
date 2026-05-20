from __future__ import annotations

import asyncio
import enum
import json
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - optional integration surface.
    from .models import ToolCall, ToolResult  # noqa: F401


class AuditOutcome(str, enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    PENDING_APPROVAL = "pending_approval"


def _json_safe(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    actor: str
    action: str
    resource: str
    outcome: AuditOutcome | str
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    correlation_id: str | None = None
    request_id: str | None = None
    risk_level: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "outcome": _json_safe(self.outcome),
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "risk_level": self.risk_level,
            "metadata": _json_safe(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


class AuditSink(Protocol):
    async def write(self, event: AuditEvent) -> None:
        ...


class InMemoryAuditSink:
    def __init__(self, max_events: int = 10_000) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self.max_events = max_events
        self._events: list[AuditEvent] = []
        self._lock = asyncio.Lock()

    async def write(self, event: AuditEvent) -> None:
        async with self._lock:
            self._events.append(event)
            overflow = len(self._events) - self.max_events
            if overflow > 0:
                del self._events[:overflow]

    async def snapshot(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [event.to_dict() for event in self._events]

    async def clear(self) -> None:
        async with self._lock:
            self._events.clear()


class JsonLinesAuditSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def write(self, event: AuditEvent) -> None:
        line = event.to_json() + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)


class AsyncAuditLogger:
    def __init__(
        self,
        sinks: list[AuditSink] | None = None,
        *,
        max_queue_size: int = 10_000,
        drop_when_full: bool = False,
    ) -> None:
        self.sinks = sinks or [InMemoryAuditSink()]
        self.drop_when_full = drop_when_full
        self._queue: asyncio.Queue[AuditEvent | None] = asyncio.Queue(maxsize=max_queue_size)
        self._worker: asyncio.Task[None] | None = None
        self._dropped = 0
        self._started = False
        self._stop_lock = asyncio.Lock()

    @property
    def dropped_events(self) -> int:
        return self._dropped

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._started = True
            self._worker = asyncio.create_task(self._run(), name="mcp-audit-logger")

    async def stop(self, *, drain: bool = True) -> None:
        async with self._stop_lock:
            if self._worker is None:
                return
            if drain:
                await self._queue.join()
            await self._queue.put(None)
            await self._worker
            self._worker = None
            self._started = False

    async def emit(self, event: AuditEvent) -> bool:
        if not self._started:
            await self.start()
        if self.drop_when_full and self._queue.full():
            self._dropped += 1
            return False
        await self._queue.put(event)
        return True

    async def log(
        self,
        event_type: str,
        *,
        actor: str,
        action: str,
        resource: str,
        outcome: AuditOutcome | str,
        correlation_id: str | None = None,
        request_id: str | None = None,
        risk_level: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        event = AuditEvent(
            event_type=event_type,
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            correlation_id=correlation_id,
            request_id=request_id,
            risk_level=risk_level,
            metadata=metadata or {},
        )
        return await self.emit(event)

    async def flush(self) -> None:
        await self._queue.join()

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                if event is None:
                    return
                for sink in self.sinks:
                    await sink.write(event)
            finally:
                self._queue.task_done()


_default_sink = InMemoryAuditSink()
_default_logger = AsyncAuditLogger(sinks=[_default_sink])


def get_audit_logger() -> AsyncAuditLogger:
    return _default_logger


def get_memory_audit_sink() -> InMemoryAuditSink:
    return _default_sink


__all__ = [
    "AsyncAuditLogger",
    "AuditEvent",
    "AuditOutcome",
    "AuditSink",
    "InMemoryAuditSink",
    "JsonLinesAuditSink",
    "get_audit_logger",
    "get_memory_audit_sink",
]
