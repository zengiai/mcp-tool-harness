from __future__ import annotations

import asyncio
import enum
import inspect
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


class CircuitState(str, enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(RuntimeError):
    pass


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 2
    half_open_max_calls: int = 1

    def __post_init__(self) -> None:
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
        if self.half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be positive")


class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def allow_request(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            if self._state == CircuitState.OPEN:
                if self._opened_at is not None and now - self._opened_at >= self.config.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    self._half_open_calls = 0
                else:
                    return False
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    return False
                self._half_open_calls += 1
            return True

    async def before_call(self) -> None:
        if not await self.allow_request():
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")

    async def record_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls = max(0, self._half_open_calls - 1)
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._close()
                return
            self._failure_count = 0

    async def record_failure(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls = max(0, self._half_open_calls - 1)
                self._open()
                return
            if self._state == CircuitState.OPEN:
                return
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._open()

    async def call(
        self,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        await self.before_call()
        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                if timeout is not None:
                    result = await asyncio.wait_for(result, timeout)
                else:
                    result = await result
            await self.record_success()
            return result
        except Exception:
            await self.record_failure()
            raise

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "half_open_calls": self._half_open_calls,
                "opened_at": self._opened_at,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                    "half_open_max_calls": self.config.half_open_max_calls,
                },
            }

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0

    def _close(self) -> None:
        self._state = CircuitState.CLOSED
        self._opened_at = None
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0


class CircuitBreakerRegistry:
    def __init__(self, default_config: CircuitBreakerConfig | None = None) -> None:
        self.default_config = default_config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get(self, name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
        async with self._lock:
            breaker = self._breakers.get(name)
            if breaker is None:
                breaker = CircuitBreaker(name, config or self.default_config)
                self._breakers[name] = breaker
            return breaker

    async def snapshot(self) -> dict[str, dict[str, Any]]:
        async with self._lock:
            items = list(self._breakers.items())
        return {name: await breaker.snapshot() for name, breaker in items}


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitBreakerRegistry",
    "CircuitState",
]
