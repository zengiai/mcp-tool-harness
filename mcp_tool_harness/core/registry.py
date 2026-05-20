"""Registry for MCP tool servers, tool specs, and governance policy metadata."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from typing import Any

from mcp_tool_harness.core.models import ToolPolicy, ToolServer, ToolSpec
from mcp_tool_harness.storage.repositories import (
    InMemoryPolicyRepository,
    InMemoryToolRepository,
    InMemoryToolServerRepository,
)


class RegistryError(RuntimeError):
    """Base registry error."""


class ToolRegistrationConflict(RegistryError):
    """Raised when a tool identity is registered with an incompatible schema."""


class ToolNotFoundError(RegistryError):
    """Raised when a tool cannot be resolved from the registry."""


@dataclass(slots=True)
class _CacheEntry:
    value: Any
    expires_at: float


class Registry:
    """Local registry with async APIs and TTL-protected in-process cache.

    设计说明：
    - Registry 只管理工具元数据、服务端元数据和治理策略，不执行工具。
    - 写入路径落到 repository，读取路径先查本地 TTL 缓存，降低高频调用时的元数据查询成本。
    - 对外保持 async API，后续替换为 DB/Redis 实现时不影响 Gateway 和 Adapter。
    """

    def __init__(
        self,
        tool_repository: InMemoryToolRepository | None = None,
        server_repository: InMemoryToolServerRepository | None = None,
        policy_repository: InMemoryPolicyRepository | None = None,
        cache_ttl_seconds: float = 30.0,
    ) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be zero or positive")
        self._tools = tool_repository or InMemoryToolRepository()
        self._servers = server_repository or InMemoryToolServerRepository()
        self._policies = policy_repository or InMemoryPolicyRepository()
        self._cache_ttl_seconds = cache_ttl_seconds
        self._tool_cache: dict[str, _CacheEntry] = {}
        self._server_cache: dict[str, _CacheEntry] = {}
        self._policy_cache: dict[str, _CacheEntry] = {}

    async def register_server(self, server: ToolServer) -> ToolServer:
        existing = await self._servers.get(server.server_id)
        if existing and existing.endpoint != server.endpoint:
            # 同一个 server_id 不能静默漂移到另一个 endpoint，否则审计和风险策略会失真。
            raise RegistryError(
                f"server_id {server.server_id!r} already points to a different endpoint"
            )
        stored = existing or await self._servers.save(server)
        self._put_cache(self._server_cache, self._server_key(stored.server_id), stored)
        return deepcopy(stored)

    async def get_server(self, server_id: str) -> ToolServer | None:
        key = self._server_key(server_id)
        cached = self._get_cache(self._server_cache, key)
        if cached is not None:
            return cached
        server = await self._servers.get(server_id)
        if server is not None:
            self._put_cache(self._server_cache, key, server)
        return server

    async def register_policy(self, policy: ToolPolicy) -> ToolPolicy:
        stored = await self._policies.save(policy)
        # 策略更新后立即刷新本地缓存；生产环境可扩展为消息广播或 Redis pub/sub 失效。
        self._put_cache(self._policy_cache, self._policy_key(stored.policy_id), stored)
        return deepcopy(stored)

    async def get_policy(self, policy_id: str) -> ToolPolicy | None:
        key = self._policy_key(policy_id)
        cached = self._get_cache(self._policy_cache, key)
        if cached is not None:
            return cached
        policy = await self._policies.get(policy_id)
        if policy is not None:
            self._put_cache(self._policy_cache, key, policy)
        return policy

    async def register_tool(
        self,
        spec: ToolSpec,
        policy: ToolPolicy | None = None,
    ) -> ToolSpec:
        # 工具身份由 server_id + name + version 决定；schema_hash 用来判断是否为幂等重复注册。
        existing = await self._tools.get_by_identity(*spec.identity)
        if existing is not None:
            if existing.schema_hash != spec.schema_hash:
                # 同身份不同 schema 属于不兼容变更，必须显式升版本，不能覆盖旧工具。
                raise ToolRegistrationConflict(
                    "tool identity already exists with a different schema_hash"
                )
            self._cache_tool(existing)
            if policy is not None:
                await self.register_policy(policy)
            return deepcopy(existing)

        stored = await self._tools.save(spec)
        if policy is not None:
            await self.register_policy(policy)
        self._cache_tool(stored)
        return deepcopy(stored)

    async def get_tool(self, tool_id: str) -> ToolSpec:
        # tool_id 查询主要服务内部跳转；Agent 框架侧通常走 get_tool_by_identity。
        key = self._tool_id_key(tool_id)
        cached = self._get_cache(self._tool_cache, key)
        if cached is not None:
            return cached
        tool = await self._tools.get(tool_id)
        if tool is None:
            raise ToolNotFoundError(f"tool_id {tool_id!r} is not registered")
        self._cache_tool(tool)
        return tool

    async def get_tool_by_identity(
        self,
        server_id: str,
        name: str,
        version: str = "1.0.0",
    ) -> ToolSpec:
        # 高频调用路径按身份查工具，缓存命中时不访问 repository。
        key = self._tool_identity_cache_key(server_id, name, version)
        cached = self._get_cache(self._tool_cache, key)
        if cached is not None:
            return cached
        tool = await self._tools.get_by_identity(server_id, name, version)
        if tool is None:
            raise ToolNotFoundError(
                f"tool {server_id!r}/{name!r}@{version!r} is not registered"
            )
        self._cache_tool(tool)
        return tool

    async def list_tools(
        self,
        server_id: str | None = None,
        enabled: bool | None = None,
    ) -> list[ToolSpec]:
        return await self._tools.list_tools(server_id=server_id, enabled=enabled)

    async def list_policies(
        self,
        server_id: str | None = None,
        tool_name: str | None = None,
        enabled: bool | None = None,
    ) -> list[ToolPolicy]:
        return await self._policies.list_policies(
            server_id=server_id,
            tool_name=tool_name,
            enabled=enabled,
        )

    async def unregister_tool(self, tool_id: str) -> bool:
        tool = await self._tools.get(tool_id)
        deleted = await self._tools.delete(tool_id)
        if deleted:
            # 删除时同时清理 tool_id 和 identity 两套缓存索引，避免读到已下线工具。
            self._tool_cache.pop(self._tool_id_key(tool_id), None)
            if tool is not None:
                self._tool_cache.pop(self._tool_identity_cache_key(*tool.identity), None)
        return deleted

    def invalidate_tool_cache(self, tool: ToolSpec | None = None) -> None:
        if tool is None:
            self._tool_cache.clear()
            return
        self._tool_cache.pop(self._tool_id_key(tool.tool_id), None)
        self._tool_cache.pop(self._tool_identity_cache_key(*tool.identity), None)

    def invalidate_all(self) -> None:
        self._tool_cache.clear()
        self._server_cache.clear()
        self._policy_cache.clear()

    def _cache_tool(self, tool: ToolSpec) -> None:
        # 同一份 ToolSpec 建两种索引：按 tool_id 和按 server/name/version。
        self._put_cache(self._tool_cache, self._tool_id_key(tool.tool_id), tool)
        self._put_cache(self._tool_cache, self._tool_identity_cache_key(*tool.identity), tool)

    def _put_cache(self, cache: dict[str, _CacheEntry], key: str, value: Any) -> None:
        if self._cache_ttl_seconds == 0:
            cache.pop(key, None)
            return
        # deepcopy 防止调用方修改返回对象后污染 Registry 内部缓存。
        cache[key] = _CacheEntry(
            value=deepcopy(value),
            expires_at=monotonic() + self._cache_ttl_seconds,
        )

    def _get_cache(self, cache: dict[str, _CacheEntry], key: str) -> Any | None:
        entry = cache.get(key)
        if entry is None:
            return None
        if entry.expires_at <= monotonic():
            # 懒失效：读取时发现过期再清理，避免额外后台线程。
            cache.pop(key, None)
            return None
        return deepcopy(entry.value)

    @staticmethod
    def _tool_id_key(tool_id: str) -> str:
        return f"tool:id:{tool_id}"

    @staticmethod
    def _tool_identity_cache_key(server_id: str, name: str, version: str) -> str:
        return f"tool:identity:{server_id}:{name}:{version}"

    @staticmethod
    def _server_key(server_id: str) -> str:
        return f"server:{server_id}"

    @staticmethod
    def _policy_key(policy_id: str) -> str:
        return f"policy:{policy_id}"
