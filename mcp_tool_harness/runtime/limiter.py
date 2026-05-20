from __future__ import annotations

import asyncio
import re
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


RATE_LIMIT_DIMENSIONS = frozenset(
    {
        "tool",
        "agent",
        "tenant",
        "agent_tool",
        "tenant_tool",
        "server_tool",
        "custom",
    }
)

_TEMPLATE_FIELD_PATTERN = re.compile(r"{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)}")


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    key: str
    remaining: float
    retry_after: float
    capacity: float
    refill_rate: float
    dimension: str | None = None
    rule_id: str | None = None
    decisions: tuple["RateLimitDecision", ...] = field(default_factory=tuple)
    rejected_decision: "RateLimitDecision | None" = None


@dataclass(frozen=True)
class RateLimitRule:
    dimension: str
    capacity: float
    refill_rate: float
    key_template: str | None = None
    rule_id: str = ""
    match: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dimension = str(self.dimension)
        if dimension not in RATE_LIMIT_DIMENSIONS:
            raise ValueError(f"unsupported rate limit dimension: {dimension}")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.refill_rate <= 0:
            raise ValueError("refill_rate must be positive")
        if dimension == "custom" and not self.key_template:
            raise ValueError("custom rate limit rule requires key_template")

        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "capacity", float(self.capacity))
        object.__setattr__(self, "refill_rate", float(self.refill_rate))
        object.__setattr__(self, "match", dict(self.match or {}))
        if not self.rule_id:
            object.__setattr__(self, "rule_id", _default_rule_id(self))


@dataclass(frozen=True)
class RateLimitConfig:
    rules: tuple[RateLimitRule, ...] = field(default_factory=tuple)

    def __init__(self, rules: Iterable[RateLimitRule | Mapping[str, Any]] = ()) -> None:
        object.__setattr__(self, "rules", _normalize_rules(rules))


@dataclass(frozen=True)
class _ResolvedRateLimitRule:
    rule: RateLimitRule
    key: str

    @property
    def bucket_id(self) -> str:
        return (
            f"{self.rule.rule_id}:{self.rule.dimension}:"
            f"{self.rule.capacity:g}:{self.rule.refill_rate:g}:{self.key}"
        )


class TokenBucket:
    def __init__(
        self,
        *,
        capacity: float,
        refill_rate: float,
        initial_tokens: float | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be positive")
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._tokens = min(float(initial_tokens if initial_tokens is not None else capacity), self.capacity)
        self._updated_at = time.monotonic()
        self.last_seen_at = self._updated_at
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        cost: float = 1.0,
        *,
        wait: bool = False,
        timeout: float | None = None,
        key: str = "",
    ) -> RateLimitDecision:
        if cost <= 0:
            raise ValueError("cost must be positive")
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            async with self._lock:
                self._refill()
                self.last_seen_at = time.monotonic()
                if self._tokens >= cost:
                    self._tokens -= cost
                    return RateLimitDecision(
                        allowed=True,
                        key=key,
                        remaining=self._tokens,
                        retry_after=0.0,
                        capacity=self.capacity,
                        refill_rate=self.refill_rate,
                    )
                retry_after = (cost - self._tokens) / self.refill_rate
                remaining = self._tokens

            if not wait:
                return RateLimitDecision(
                    allowed=False,
                    key=key,
                    remaining=remaining,
                    retry_after=retry_after,
                    capacity=self.capacity,
                    refill_rate=self.refill_rate,
                )

            now = time.monotonic()
            if deadline is not None and now + retry_after > deadline:
                return RateLimitDecision(
                    allowed=False,
                    key=key,
                    remaining=remaining,
                    retry_after=retry_after,
                    capacity=self.capacity,
                    refill_rate=self.refill_rate,
                )
            await asyncio.sleep(retry_after)

    async def snapshot(self) -> dict[str, float]:
        async with self._lock:
            self._refill()
            return {
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "tokens": self._tokens,
                "updated_at": self._updated_at,
                "last_seen_at": self.last_seen_at,
            }

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self._updated_at)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._updated_at = now


class RateLimiter:
    def __init__(
        self,
        *,
        default_capacity: float = 60.0,
        default_refill_rate: float = 1.0,
        overrides: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.default_capacity = default_capacity
        self.default_refill_rate = default_refill_rate
        self._overrides = dict(overrides or {})
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        key: str,
        *,
        cost: float = 1.0,
        wait: bool = False,
        timeout: float | None = None,
    ) -> RateLimitDecision:
        bucket = await self.bucket_for(key)
        return await bucket.acquire(cost, wait=wait, timeout=timeout, key=key)

    async def allow(self, key: str, *, cost: float = 1.0) -> bool:
        return (await self.acquire(key, cost=cost)).allowed

    async def bucket_for(self, key: str) -> TokenBucket:
        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                capacity, refill_rate = self._overrides.get(key, (self.default_capacity, self.default_refill_rate))
                bucket = TokenBucket(capacity=capacity, refill_rate=refill_rate)
                self._buckets[key] = bucket
            return bucket

    async def configure(self, key: str, *, capacity: float, refill_rate: float) -> None:
        async with self._lock:
            self._overrides[key] = (capacity, refill_rate)
            self._buckets[key] = TokenBucket(capacity=capacity, refill_rate=refill_rate)

    async def cleanup_idle(self, max_idle_seconds: float) -> int:
        now = time.monotonic()
        async with self._lock:
            stale = [
                key
                for key, bucket in self._buckets.items()
                if now - bucket.last_seen_at >= max_idle_seconds
            ]
            for key in stale:
                self._buckets.pop(key, None)
            return len(stale)

    async def snapshot(self) -> dict[str, dict[str, float]]:
        async with self._lock:
            items = list(self._buckets.items())
        return {key: await bucket.snapshot() for key, bucket in items}


class MultiDimensionalRateLimiter:
    """Composable runtime limiter for tenant/agent/tool/custom isolation.

    Thread-safety说明：
    - 规则更新由实例级 async lock 串行化。
    - 单次多维扣减按 bucket_id 排序后同时持有相关 bucket 锁，先全量检查，再统一扣减。

    事务边界说明：
    - 限流只维护进程内 token 状态，不承担跨进程强一致事务。
    - 动态替换规则会清空旧 bucket，确保新配置对后续调用立即生效。

    缓存策略说明：
    - 每个规则维度和实际 key 对应一个内存 TokenBucket。
    - 可通过 cleanup_idle 清理长时间未访问的 bucket，避免无限增长。
    """

    def __init__(
        self,
        *,
        rules: Iterable[RateLimitRule | Mapping[str, Any]] | RateLimitConfig | None = None,
        default_rules: Iterable[RateLimitRule | Mapping[str, Any]] | RateLimitConfig | None = None,
    ) -> None:
        self._default_rules = _normalize_config_rules(default_rules)
        self._configured_rules = None if rules is None else _normalize_config_rules(rules)
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        key: str | None = None,
        *,
        cost: float = 1.0,
        wait: bool = False,
        timeout: float | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        tool_name: str | None = None,
        server_id: str | None = None,
        args: Mapping[str, Any] | None = None,
        arguments: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        rules: Iterable[RateLimitRule | Mapping[str, Any]] | RateLimitConfig | None = None,
        **attributes: Any,
    ) -> RateLimitDecision:
        if cost <= 0:
            raise ValueError("cost must be positive")

        values = _rate_limit_values(
            key=key,
            tenant_id=tenant_id,
            agent_id=agent_id,
            tool_name=tool_name,
            server_id=server_id,
            args=args if args is not None else arguments,
            metadata=metadata,
            attributes=attributes,
        )
        active_rules = _normalize_config_rules(rules) if rules is not None else await self._active_rules()
        resolved = _resolve_rules(active_rules, values)
        if not resolved:
            return RateLimitDecision(
                allowed=True,
                key=str(key or values.get("tool_name") or ""),
                remaining=float("inf"),
                retry_after=0.0,
                capacity=float("inf"),
                refill_rate=float("inf"),
                dimension=None,
            )

        buckets = await self._buckets_for(resolved)
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            decision = await self._try_acquire_all(resolved, buckets, cost)
            if decision.allowed:
                return decision
            if not wait:
                return decision

            retry_after = max(item.retry_after for item in decision.decisions if not item.allowed)
            now = time.monotonic()
            if deadline is not None and now + retry_after > deadline:
                return decision
            await asyncio.sleep(retry_after)

    async def allow(
        self,
        key: str | None = None,
        *,
        cost: float = 1.0,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        tool_name: str | None = None,
        server_id: str | None = None,
        args: Mapping[str, Any] | None = None,
        arguments: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        rules: Iterable[RateLimitRule | Mapping[str, Any]] | RateLimitConfig | None = None,
        **attributes: Any,
    ) -> bool:
        return (
            await self.acquire(
                key,
                cost=cost,
                tenant_id=tenant_id,
                agent_id=agent_id,
                tool_name=tool_name,
                server_id=server_id,
                args=args,
                arguments=arguments,
                metadata=metadata,
                rules=rules,
                **attributes,
            )
        ).allowed

    async def configure_rules(
        self,
        rules: Iterable[RateLimitRule | Mapping[str, Any]] | RateLimitConfig,
    ) -> None:
        await self.replace_rules(rules)

    async def replace_rules(
        self,
        rules: Iterable[RateLimitRule | Mapping[str, Any]] | RateLimitConfig,
    ) -> None:
        normalized = _normalize_config_rules(rules)
        async with self._lock:
            self._configured_rules = normalized
            self._buckets.clear()

    async def clear_rules(self) -> None:
        async with self._lock:
            self._configured_rules = ()
            self._buckets.clear()

    async def cleanup_idle(self, max_idle_seconds: float) -> int:
        now = time.monotonic()
        async with self._lock:
            stale = [
                key
                for key, bucket in self._buckets.items()
                if now - bucket.last_seen_at >= max_idle_seconds
            ]
            for key in stale:
                self._buckets.pop(key, None)
            return len(stale)

    async def snapshot(self) -> dict[str, dict[str, float]]:
        async with self._lock:
            items = list(self._buckets.items())
        return {key: await bucket.snapshot() for key, bucket in items}

    async def _active_rules(self) -> tuple[RateLimitRule, ...]:
        async with self._lock:
            if self._configured_rules is not None:
                return self._configured_rules
            return self._default_rules

    async def _buckets_for(
        self,
        resolved: Sequence[_ResolvedRateLimitRule],
    ) -> dict[str, TokenBucket]:
        async with self._lock:
            buckets: dict[str, TokenBucket] = {}
            for item in resolved:
                bucket_id = item.bucket_id
                bucket = self._buckets.get(bucket_id)
                if bucket is None:
                    bucket = TokenBucket(capacity=item.rule.capacity, refill_rate=item.rule.refill_rate)
                    self._buckets[bucket_id] = bucket
                buckets[bucket_id] = bucket
            return buckets

    async def _try_acquire_all(
        self,
        resolved: Sequence[_ResolvedRateLimitRule],
        buckets: Mapping[str, TokenBucket],
        cost: float,
    ) -> RateLimitDecision:
        ordered = sorted(((item.bucket_id, item) for item in resolved), key=lambda pair: pair[0])
        decisions_by_bucket: dict[str, RateLimitDecision] = {}

        async with AsyncExitStack() as stack:
            for bucket_id, _ in ordered:
                await stack.enter_async_context(buckets[bucket_id]._lock)

            for bucket_id, item in ordered:
                bucket = buckets[bucket_id]
                bucket._refill()
                bucket.last_seen_at = time.monotonic()
                if bucket._tokens >= cost:
                    decisions_by_bucket[bucket_id] = RateLimitDecision(
                        allowed=True,
                        key=item.key,
                        remaining=bucket._tokens - cost,
                        retry_after=0.0,
                        capacity=bucket.capacity,
                        refill_rate=bucket.refill_rate,
                        dimension=item.rule.dimension,
                        rule_id=item.rule.rule_id,
                    )
                else:
                    retry_after = (cost - bucket._tokens) / bucket.refill_rate
                    decisions_by_bucket[bucket_id] = RateLimitDecision(
                        allowed=False,
                        key=item.key,
                        remaining=bucket._tokens,
                        retry_after=retry_after,
                        capacity=bucket.capacity,
                        refill_rate=bucket.refill_rate,
                        dimension=item.rule.dimension,
                        rule_id=item.rule.rule_id,
                    )

            decisions = tuple(decisions_by_bucket[bucket_id] for bucket_id, _ in ordered)
            rejected = next((item for item in decisions if not item.allowed), None)
            if rejected is not None:
                return _summarize_decisions(False, decisions, rejected)

            for bucket_id, _ in ordered:
                buckets[bucket_id]._tokens -= cost

        return _summarize_decisions(True, decisions, None)


def _summarize_decisions(
    allowed: bool,
    decisions: tuple[RateLimitDecision, ...],
    rejected: RateLimitDecision | None,
) -> RateLimitDecision:
    if rejected is not None:
        return RateLimitDecision(
            allowed=False,
            key=rejected.key,
            remaining=rejected.remaining,
            retry_after=rejected.retry_after,
            capacity=rejected.capacity,
            refill_rate=rejected.refill_rate,
            dimension=rejected.dimension,
            rule_id=rejected.rule_id,
            decisions=decisions,
            rejected_decision=rejected,
        )

    tightest = min(decisions, key=lambda item: item.remaining)
    return RateLimitDecision(
        allowed=allowed,
        key=tightest.key,
        remaining=tightest.remaining,
        retry_after=0.0,
        capacity=tightest.capacity,
        refill_rate=tightest.refill_rate,
        dimension=tightest.dimension,
        rule_id=tightest.rule_id,
        decisions=decisions,
        rejected_decision=None,
    )


def _normalize_config_rules(
    rules: Iterable[RateLimitRule | Mapping[str, Any]] | RateLimitConfig | None,
) -> tuple[RateLimitRule, ...]:
    if rules is None:
        return ()
    if isinstance(rules, RateLimitConfig):
        return rules.rules
    return _normalize_rules(rules)


def _normalize_rules(rules: Iterable[RateLimitRule | Mapping[str, Any]]) -> tuple[RateLimitRule, ...]:
    return tuple(_normalize_rule(rule) for rule in rules)


def _normalize_rule(rule: RateLimitRule | Mapping[str, Any]) -> RateLimitRule:
    if isinstance(rule, RateLimitRule):
        return rule

    if isinstance(rule, Mapping):
        data = dict(rule)
        rule_id = data.get("rule_id", data.get("id", ""))
        dimension = data.get("dimension", data.get("scope", data.get("name")))
        capacity = data.get("capacity", data.get("burst", data.get("limit_per_minute")))
        refill_rate = data.get("refill_rate", data.get("tokens_per_second", data.get("rate")))
        if refill_rate is None and data.get("limit_per_minute") is not None:
            refill_rate = float(data["limit_per_minute"]) / 60.0
        key_template = data.get("key_template", data.get("template"))
        match = data.get("match", data.get("matches", {}))
    else:
        rule_id = getattr(rule, "rule_id", getattr(rule, "id", ""))
        dimension = getattr(rule, "dimension", getattr(rule, "scope", None))
        capacity = getattr(rule, "capacity", getattr(rule, "burst", None))
        refill_rate = getattr(rule, "refill_rate", getattr(rule, "tokens_per_second", getattr(rule, "rate", None)))
        key_template = getattr(rule, "key_template", getattr(rule, "template", None))
        match = getattr(rule, "match", getattr(rule, "matches", {}))

    if dimension is None:
        raise ValueError("rate limit rule requires dimension")
    if capacity is None:
        raise ValueError("rate limit rule requires capacity")
    if refill_rate is None:
        raise ValueError("rate limit rule requires refill_rate")

    return RateLimitRule(
        dimension=str(dimension),
        capacity=float(capacity),
        refill_rate=float(refill_rate),
        key_template=None if key_template is None else str(key_template),
        rule_id=str(rule_id or ""),
        match=dict(match or {}),
    )


class PolicyAwareRateLimiter:
    """Rate limiter that resolves the latest ToolPolicy for every call.

    配置优先级：
    1. ToolPolicy.rate_limits
    2. ToolPolicy.metadata["rate_limits"]
    3. ToolPolicy.rate_limit_per_minute
    4. default_rules
    """

    def __init__(
        self,
        *,
        security: Any | None = None,
        registry: Any | None = None,
        default_rules: Iterable[RateLimitRule | Mapping[str, Any]] | RateLimitConfig | None = None,
    ) -> None:
        if security is None and registry is None:
            raise ValueError("security or registry is required")
        self.security = security
        self.registry = registry
        self._limiter = MultiDimensionalRateLimiter(default_rules=default_rules)

    async def acquire(
        self,
        key: str | None = None,
        *,
        context: Any | None = None,
        tool: Any | None = None,
        args: Mapping[str, Any] | None = None,
        cost: float = 1.0,
        wait: bool = False,
        timeout: float | None = None,
        **attributes: Any,
    ) -> RateLimitDecision:
        policy = await self._resolve_policy(context, tool)
        rules = self._rules_from_policy(policy)
        return await self._limiter.acquire(
            key,
            cost=cost,
            wait=wait,
            timeout=timeout,
            tenant_id=_read_attr(context, "tenant_id", "default"),
            agent_id=_read_attr(context, "agent_id", _read_attr(context, "principal", None)),
            tool_name=_read_attr(tool, "tool_name", _read_attr(tool, "name", _read_attr(context, "tool_name", key))),
            server_id=_read_attr(tool, "server_id", _read_attr(context, "server_id", "local")),
            args=args or {},
            metadata=_read_attr(context, "metadata", {}) or {},
            rules=rules if rules else None,
            **attributes,
        )

    async def configure_rules(
        self,
        rules: Iterable[RateLimitRule | Mapping[str, Any]] | RateLimitConfig,
    ) -> None:
        await self._limiter.configure_rules(rules)

    async def replace_rules(
        self,
        rules: Iterable[RateLimitRule | Mapping[str, Any]] | RateLimitConfig,
    ) -> None:
        await self._limiter.replace_rules(rules)

    async def clear_rules(self) -> None:
        await self._limiter.clear_rules()

    async def snapshot(self) -> dict[str, dict[str, float]]:
        return await self._limiter.snapshot()

    async def _resolve_policy(self, context: Any, tool: Any) -> Any | None:
        resolver = getattr(self.security, "resolve_policy", None)
        if resolver is not None:
            return await _maybe_await(resolver(context, tool))

        if self.registry is None:
            return None
        lister = getattr(self.registry, "list_policies", None)
        if lister is None:
            return None
        tool_name = _read_attr(tool, "tool_name", _read_attr(tool, "name", _read_attr(context, "tool_name", "")))
        server_id = _read_attr(tool, "server_id", _read_attr(context, "server_id", None))
        try:
            policies = await _maybe_await(lister(server_id=server_id, tool_name=tool_name, enabled=True))
        except TypeError:
            policies = await _maybe_await(lister())
        return next(iter(policies or ()), None)

    @staticmethod
    def _rules_from_policy(policy: Any | None) -> tuple[RateLimitRule, ...]:
        if policy is None:
            return ()
        configured = _read_attr(policy, "rate_limits", ()) or _lookup_path(
            {"metadata": _read_attr(policy, "metadata", {}) or {}},
            "metadata.rate_limits",
        )
        if configured:
            return _normalize_config_rules(configured)
        per_minute = _read_attr(policy, "rate_limit_per_minute", None)
        if per_minute is None:
            return ()
        return _normalize_config_rules(
            (
                {
                    "dimension": "agent_tool",
                    "capacity": float(per_minute),
                    "refill_rate": float(per_minute) / 60.0,
                    "rule_id": f"policy:{_read_attr(policy, 'policy_id', 'default')}:agent_tool",
                },
            )
        )


def _read_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _default_rule_id(rule: RateLimitRule) -> str:
    match_part = ",".join(f"{key}={rule.match[key]}" for key in sorted(rule.match))
    template_part = rule.key_template or ""
    return f"{rule.dimension}:{template_part}:{match_part}"


def _rate_limit_values(
    *,
    key: str | None,
    tenant_id: str | None,
    agent_id: str | None,
    tool_name: str | None,
    server_id: str | None,
    args: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    attributes: Mapping[str, Any],
) -> dict[str, Any]:
    tool = tool_name if tool_name is not None else key
    values: dict[str, Any] = {
        "key": key,
        "tenant_id": tenant_id or "default",
        "agent_id": agent_id,
        "tool_name": tool,
        "server_id": server_id or "local",
        "args": dict(args or {}),
        "arguments": dict(args or {}),
        "metadata": dict(metadata or {}),
    }
    values.update(attributes)
    return values


def _resolve_rules(
    rules: Sequence[RateLimitRule],
    values: Mapping[str, Any],
) -> tuple[_ResolvedRateLimitRule, ...]:
    resolved: list[_ResolvedRateLimitRule] = []
    seen: set[str] = set()
    for rule in rules:
        if not _matches_rule(rule, values):
            continue
        key = _rule_key(rule, values)
        if key is None:
            continue
        item = _ResolvedRateLimitRule(rule=rule, key=key)
        if item.bucket_id in seen:
            continue
        seen.add(item.bucket_id)
        resolved.append(item)
    return tuple(resolved)


def _matches_rule(rule: RateLimitRule, values: Mapping[str, Any]) -> bool:
    aliases = {
        "tool": "tool_name",
        "server": "server_id",
        "tenant": "tenant_id",
        "agent": "agent_id",
    }
    for raw_field, expected in rule.match.items():
        field = aliases.get(str(raw_field), str(raw_field))
        actual = values.get(field)
        if expected in (None, "*"):
            continue
        if isinstance(expected, (frozenset, list, set, tuple)):
            if actual not in expected:
                return False
            continue
        if actual != expected:
            return False
    return True


def _rule_key(rule: RateLimitRule, values: Mapping[str, Any]) -> str | None:
    tool_name = values.get("tool_name")
    agent_id = values.get("agent_id")
    tenant_id = values.get("tenant_id")
    server_id = values.get("server_id")

    if rule.dimension == "tool":
        return None if tool_name is None else f"tool:{tool_name}"
    if rule.dimension == "agent":
        return None if agent_id is None else f"agent:{agent_id}"
    if rule.dimension == "tenant":
        return f"tenant:{tenant_id or 'default'}"
    if rule.dimension == "agent_tool":
        if agent_id is None or tool_name is None:
            return None
        return f"agent:{agent_id}:tool:{tool_name}"
    if rule.dimension == "tenant_tool":
        if tool_name is None:
            return None
        return f"tenant:{tenant_id or 'default'}:tool:{tool_name}"
    if rule.dimension == "server_tool":
        if tool_name is None:
            return None
        return f"server:{server_id or 'local'}:tool:{tool_name}"
    if rule.dimension == "custom":
        return _render_key_template(str(rule.key_template), values)
    return None


def _render_key_template(template: str, values: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        value = _lookup_path(values, match.group(1))
        return "" if value is None else str(value)

    return _TEMPLATE_FIELD_PATTERN.sub(replace, template)


def _lookup_path(values: Mapping[str, Any], path: str) -> Any:
    current: Any = values
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
    return current


__all__ = [
    "MultiDimensionalRateLimiter",
    "PolicyAwareRateLimiter",
    "RATE_LIMIT_DIMENSIONS",
    "RateLimitConfig",
    "RateLimitDecision",
    "RateLimitRule",
    "RateLimiter",
    "TokenBucket",
]
