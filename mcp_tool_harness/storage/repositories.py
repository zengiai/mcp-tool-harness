"""Pure standard-library in-memory repositories.

The repositories expose async methods so callers can swap in persistent
implementations later without changing registry or adapter code.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import TypeVar

from mcp_tool_harness.core.models import (
    ApprovalStatus,
    ApprovalTask,
    ToolCallRecord,
    ToolPolicy,
    ToolServer,
    ToolSpec,
)


T = TypeVar("T")


def _clone(value: T) -> T:
    return deepcopy(value)


class InMemoryToolServerRepository:
    def __init__(self) -> None:
        self._items: dict[str, ToolServer] = {}
        self._lock = asyncio.Lock()

    async def save(self, server: ToolServer) -> ToolServer:
        async with self._lock:
            self._items[server.server_id] = _clone(server)
            return _clone(server)

    async def get(self, server_id: str) -> ToolServer | None:
        async with self._lock:
            item = self._items.get(server_id)
            return _clone(item) if item is not None else None

    async def list_servers(self) -> list[ToolServer]:
        async with self._lock:
            return [_clone(item) for item in self._items.values()]

    async def delete(self, server_id: str) -> bool:
        async with self._lock:
            return self._items.pop(server_id, None) is not None


class InMemoryToolRepository:
    def __init__(self) -> None:
        self._items: dict[str, ToolSpec] = {}
        self._identity_index: dict[tuple[str, str, str], str] = {}
        self._lock = asyncio.Lock()

    async def save(self, tool: ToolSpec) -> ToolSpec:
        async with self._lock:
            stored = _clone(tool)
            previous = self._items.get(stored.tool_id)
            if previous is not None:
                self._identity_index.pop(previous.identity, None)
            self._items[stored.tool_id] = stored
            self._identity_index[stored.identity] = stored.tool_id
            return _clone(stored)

    async def get(self, tool_id: str) -> ToolSpec | None:
        async with self._lock:
            item = self._items.get(tool_id)
            return _clone(item) if item is not None else None

    async def get_by_identity(
        self,
        server_id: str,
        name: str,
        version: str = "1.0.0",
    ) -> ToolSpec | None:
        async with self._lock:
            tool_id = self._identity_index.get((server_id, name, version))
            if tool_id is None:
                return None
            item = self._items.get(tool_id)
            return _clone(item) if item is not None else None

    async def list_tools(
        self,
        server_id: str | None = None,
        enabled: bool | None = None,
    ) -> list[ToolSpec]:
        async with self._lock:
            values = self._items.values()
            if server_id is not None:
                values = [item for item in values if item.server_id == server_id]
            if enabled is not None:
                values = [item for item in values if item.enabled is enabled]
            return [_clone(item) for item in values]

    async def delete(self, tool_id: str) -> bool:
        async with self._lock:
            item = self._items.pop(tool_id, None)
            if item is None:
                return False
            self._identity_index.pop(item.identity, None)
            return True


class InMemoryPolicyRepository:
    def __init__(self) -> None:
        self._items: dict[str, ToolPolicy] = {}
        self._lock = asyncio.Lock()

    async def save(self, policy: ToolPolicy) -> ToolPolicy:
        async with self._lock:
            self._items[policy.policy_id] = _clone(policy)
            return _clone(policy)

    async def get(self, policy_id: str) -> ToolPolicy | None:
        async with self._lock:
            item = self._items.get(policy_id)
            return _clone(item) if item is not None else None

    async def list_policies(
        self,
        server_id: str | None = None,
        tool_name: str | None = None,
        enabled: bool | None = None,
    ) -> list[ToolPolicy]:
        async with self._lock:
            values = self._items.values()
            if server_id is not None:
                values = [
                    item
                    for item in values
                    if item.server_id is None or item.server_id == server_id
                ]
            if tool_name is not None:
                values = [
                    item
                    for item in values
                    if item.tool_name == "*" or item.tool_name == tool_name
                ]
            if enabled is not None:
                values = [item for item in values if item.enabled is enabled]
            return [_clone(item) for item in values]

    async def delete(self, policy_id: str) -> bool:
        async with self._lock:
            return self._items.pop(policy_id, None) is not None


class InMemoryAuditRepository:
    def __init__(self) -> None:
        self._items: dict[str, ToolCallRecord] = {}
        self._order: list[str] = []
        self._lock = asyncio.Lock()

    async def append(self, record: ToolCallRecord) -> ToolCallRecord:
        async with self._lock:
            stored = _clone(record)
            if stored.record_id not in self._items:
                self._order.append(stored.record_id)
            self._items[stored.record_id] = stored
            return _clone(stored)

    async def get(self, record_id: str) -> ToolCallRecord | None:
        async with self._lock:
            item = self._items.get(record_id)
            return _clone(item) if item is not None else None

    async def list_records(
        self,
        request_id: str | None = None,
        tool_id: str | None = None,
        principal: str | None = None,
        limit: int = 100,
    ) -> list[ToolCallRecord]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        async with self._lock:
            records = [self._items[item_id] for item_id in self._order]
            if request_id is not None:
                records = [
                    item for item in records if item.context.request_id == request_id
                ]
            if tool_id is not None:
                records = [item for item in records if item.tool_id == tool_id]
            if principal is not None:
                records = [
                    item for item in records if item.context.principal == principal
                ]
            return [_clone(item) for item in records[-limit:]]


class InMemoryApprovalRepository:
    def __init__(self) -> None:
        self._items: dict[str, ApprovalTask] = {}
        self._order: list[str] = []
        self._lock = asyncio.Lock()

    async def save(self, task: ApprovalTask) -> ApprovalTask:
        async with self._lock:
            stored = _clone(task)
            if stored.approval_id not in self._items:
                self._order.append(stored.approval_id)
            self._items[stored.approval_id] = stored
            return _clone(stored)

    async def get(self, approval_id: str) -> ApprovalTask | None:
        async with self._lock:
            item = self._items.get(approval_id)
            return _clone(item) if item is not None else None

    async def list_tasks(
        self,
        status: ApprovalStatus | None = None,
        requested_by: str | None = None,
        limit: int = 100,
    ) -> list[ApprovalTask]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        async with self._lock:
            tasks = [self._items[item_id] for item_id in self._order]
            if status is not None:
                tasks = [item for item in tasks if item.status == status]
            if requested_by is not None:
                tasks = [item for item in tasks if item.requested_by == requested_by]
            return [_clone(item) for item in tasks[-limit:]]
