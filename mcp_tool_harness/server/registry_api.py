"""Tool registration API for the MCP Tool Harness server layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping


ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class ToolMetadata:
    """Metadata exposed to MCP and framework adapters."""

    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    permissions: tuple[str, ...] = field(default_factory=tuple)
    idempotent: bool = False
    rate_limit_per_minute: int | None = None
    timeout_ms: int | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegisteredTool:
    metadata: ToolMetadata
    handler: ToolHandler


class ToolRegistry:
    """In-memory tool registry.

    This class is intentionally dependency-free. A storage-backed registry can
    implement the same methods later without changing the server API.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        handler: ToolHandler,
        *,
        description: str = "",
        input_schema: Mapping[str, Any] | None = None,
        output_schema: Mapping[str, Any] | None = None,
        permissions: Iterable[str] | None = None,
        idempotent: bool = False,
        rate_limit_per_minute: int | None = None,
        timeout_ms: int | None = None,
        tags: Iterable[str] | None = None,
        replace: bool = False,
    ) -> RegisteredTool:
        """Register a callable tool and return its registry record."""

        if not name or not isinstance(name, str):
            raise ValueError("tool name must be a non-empty string")
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        if name in self._tools and not replace:
            raise ValueError(f"tool '{name}' is already registered")
        if rate_limit_per_minute is not None and rate_limit_per_minute < 1:
            raise ValueError("rate_limit_per_minute must be positive when provided")
        if timeout_ms is not None and timeout_ms < 1:
            raise ValueError("timeout_ms must be positive when provided")

        metadata = ToolMetadata(
            name=name,
            description=description,
            input_schema=dict(input_schema or {}),
            output_schema=dict(output_schema or {}),
            permissions=tuple(permissions or ()),
            idempotent=idempotent,
            rate_limit_per_minute=rate_limit_per_minute,
            timeout_ms=timeout_ms,
            tags=tuple(tags or ()),
        )
        registered = RegisteredTool(metadata=metadata, handler=handler)
        self._tools[name] = registered
        return registered

    def unregister(self, name: str) -> None:
        try:
            del self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool '{name}' is not registered") from exc

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool '{name}' is not registered") from exc

    def list(self) -> list[ToolMetadata]:
        return [registered.metadata for registered in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def clear(self) -> None:
        self._tools.clear()
