from __future__ import annotations

import asyncio
import functools
import inspect
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


class OperationTimeoutError(TimeoutError):
    pass


async def run_with_timeout(awaitable: Awaitable[Any], timeout: float | None, *, operation: str = "operation") -> Any:
    if timeout is None:
        return await awaitable
    if timeout <= 0:
        raise OperationTimeoutError(f"{operation} timed out before it started")
    try:
        return await asyncio.wait_for(awaitable, timeout)
    except asyncio.TimeoutError as exc:
        raise OperationTimeoutError(f"{operation} timed out after {timeout:.3f}s") from exc


@asynccontextmanager
async def timeout_scope(timeout: float | None, *, operation: str = "operation"):
    if timeout is None:
        yield
        return
    if timeout <= 0:
        raise OperationTimeoutError(f"{operation} timed out before it started")
    try:
        async with asyncio.timeout(timeout):
            yield
    except TimeoutError as exc:
        raise OperationTimeoutError(f"{operation} timed out after {timeout:.3f}s") from exc


@dataclass(frozen=True)
class TimeoutBudget:
    timeout: float | None
    started_at: float = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.timeout is not None and self.timeout < 0:
            raise ValueError("timeout must be non-negative")
        if self.started_at is None:
            object.__setattr__(self, "started_at", time.monotonic())

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def remaining(self) -> float | None:
        if self.timeout is None:
            return None
        return max(0.0, self.timeout - self.elapsed())

    def expired(self) -> bool:
        remaining = self.remaining()
        return remaining is not None and remaining <= 0

    def child(self, timeout: float | None = None) -> "TimeoutBudget":
        parent_remaining = self.remaining()
        if parent_remaining is None:
            return TimeoutBudget(timeout)
        if timeout is None:
            return TimeoutBudget(parent_remaining)
        return TimeoutBudget(min(parent_remaining, timeout))

    async def wait_for(self, awaitable: Awaitable[Any], *, operation: str = "operation") -> Any:
        return await run_with_timeout(awaitable, self.remaining(), operation=operation)


def with_timeout(timeout: float | None, *, operation: str | None = None):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        name = operation or getattr(func, "__name__", "operation")

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await run_with_timeout(func(*args, **kwargs), timeout, operation=name)

            return async_wrapper

        @functools.wraps(func)
        async def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await run_with_timeout(asyncio.to_thread(func, *args, **kwargs), timeout, operation=name)

        return sync_wrapper

    return decorator


__all__ = [
    "OperationTimeoutError",
    "TimeoutBudget",
    "run_with_timeout",
    "timeout_scope",
    "with_timeout",
]
