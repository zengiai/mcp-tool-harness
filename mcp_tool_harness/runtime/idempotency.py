from __future__ import annotations

import asyncio
import enum
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping


class IdempotencyStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DuplicateRequestError(RuntimeError):
    pass


@dataclass
class IdempotencyRecord:
    key: str
    status: IdempotencyStatus
    fingerprint: str | None = None
    result: Any = None
    error: str | None = None
    attempt_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    expires_at: float | None = None

    def expired(self) -> bool:
        return self.expires_at is not None and time.monotonic() >= self.expires_at


@dataclass(frozen=True)
class IdempotencyDecision:
    accepted: bool
    replay: bool
    in_progress: bool
    record: IdempotencyRecord | None = None
    reason: str = ""


class InMemoryIdempotencyStore:
    def __init__(
        self,
        *,
        default_ttl: float = 300.0,
        retry_failed: bool = True,
        max_attempts: int = 3,
    ) -> None:
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.default_ttl = default_ttl
        self.retry_failed = retry_failed
        self.max_attempts = max_attempts
        self._records: dict[str, IdempotencyRecord] = {}
        self._lock = asyncio.Lock()

    async def start(
        self,
        key: str,
        *,
        ttl: float | None = None,
        fingerprint: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> IdempotencyDecision:
        ttl = ttl if ttl is not None else self.default_ttl
        if ttl <= 0:
            raise ValueError("ttl must be positive")

        async with self._lock:
            existing = self._records.get(key)
            if existing is not None and existing.expired():
                self._records.pop(key, None)
                existing = None

            attempt_count = 1
            created_at: float | None = None
            if existing is not None:
                if existing.fingerprint and fingerprint and existing.fingerprint != fingerprint:
                    return IdempotencyDecision(False, False, False, existing, "fingerprint_mismatch")
                if existing.status == IdempotencyStatus.COMPLETED:
                    return IdempotencyDecision(False, True, False, existing, "completed_replay")
                if existing.status == IdempotencyStatus.IN_PROGRESS:
                    return IdempotencyDecision(False, False, True, existing, "already_in_progress")
                if existing.status == IdempotencyStatus.FAILED and not self.retry_failed:
                    return IdempotencyDecision(False, False, False, existing, "failed_not_retryable")
                if existing.status == IdempotencyStatus.FAILED:
                    if existing.attempt_count >= self.max_attempts:
                        return IdempotencyDecision(
                            False,
                            False,
                            False,
                            existing,
                            "failed_retry_exhausted",
                        )
                    attempt_count = existing.attempt_count + 1
                    created_at = existing.created_at

            now = time.monotonic()
            record = IdempotencyRecord(
                key=key,
                status=IdempotencyStatus.IN_PROGRESS,
                fingerprint=fingerprint,
                attempt_count=attempt_count,
                metadata=metadata or {},
                created_at=created_at or now,
                updated_at=now,
                expires_at=now + ttl,
            )
            self._records[key] = record
            return IdempotencyDecision(True, False, False, record, "accepted")

    async def complete(self, key: str, result: Any = None) -> IdempotencyRecord:
        async with self._lock:
            record = self._require_record(key)
            record.status = IdempotencyStatus.COMPLETED
            record.result = result
            record.error = None
            record.updated_at = time.monotonic()
            return record

    async def fail(self, key: str, error: BaseException | str) -> IdempotencyRecord:
        async with self._lock:
            record = self._require_record(key)
            record.status = IdempotencyStatus.FAILED
            record.error = str(error)
            record.updated_at = time.monotonic()
            return record

    async def get(self, key: str) -> IdempotencyRecord | None:
        async with self._lock:
            record = self._records.get(key)
            if record is not None and record.expired():
                self._records.pop(key, None)
                return None
            return record

    async def remove(self, key: str) -> bool:
        async with self._lock:
            return self._records.pop(key, None) is not None

    async def cleanup(self) -> int:
        async with self._lock:
            expired = [key for key, record in self._records.items() if record.expired()]
            for key in expired:
                self._records.pop(key, None)
            return len(expired)

    async def execute(
        self,
        key: str,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        *args: Any,
        ttl: float | None = None,
        fingerprint: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        decision = await self.start(key, ttl=ttl, fingerprint=fingerprint, metadata=metadata)
        if decision.replay and decision.record is not None:
            return decision.record.result
        if not decision.accepted:
            raise DuplicateRequestError(f"Idempotency key '{key}' rejected: {decision.reason}")

        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            await self.complete(key, result)
            return result
        except Exception as exc:
            await self.fail(key, exc)
            raise

    async def snapshot(self) -> dict[str, dict[str, Any]]:
        async with self._lock:
            return {
                key: {
                    "key": record.key,
                    "status": record.status.value,
                    "fingerprint": record.fingerprint,
                    "error": record.error,
                    "attempt_count": record.attempt_count,
                    "metadata": dict(record.metadata),
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                    "expires_at": record.expires_at,
                }
                for key, record in self._records.items()
                if not record.expired()
            }

    def _require_record(self, key: str) -> IdempotencyRecord:
        record = self._records.get(key)
        if record is None or record.expired():
            raise KeyError(f"Idempotency key '{key}' does not exist")
        return record


__all__ = [
    "DuplicateRequestError",
    "IdempotencyDecision",
    "IdempotencyRecord",
    "IdempotencyStatus",
    "InMemoryIdempotencyStore",
]
