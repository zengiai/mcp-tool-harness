from __future__ import annotations

import enum
import fnmatch
import inspect
import re
import time
from dataclasses import dataclass, field, is_dataclass, asdict
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

from .models import PolicyDecision

if TYPE_CHECKING:  # pragma: no cover - optional integration surface.
    from .models import Principal, ToolCall, ToolCallContext, ToolDefinition, ToolPolicy  # noqa: F401


class PolicyEffect(str, enum.Enum):
    ALLOW = "allow"
    DENY = "deny"


class RiskLevel(str, enum.Enum):
    L0 = "l0"
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_RISK_ORDER = {
    RiskLevel.L0: 10,
    RiskLevel.LOW: 10,
    RiskLevel.L1: 40,
    RiskLevel.MEDIUM: 40,
    RiskLevel.L2: 70,
    RiskLevel.HIGH: 70,
    RiskLevel.L3: 90,
    RiskLevel.CRITICAL: 90,
}

_RISK_ALIASES = {
    "0": RiskLevel.L0,
    "l0": RiskLevel.L0,
    "low": RiskLevel.LOW,
    "1": RiskLevel.L1,
    "l1": RiskLevel.L1,
    "medium": RiskLevel.MEDIUM,
    "2": RiskLevel.L2,
    "l2": RiskLevel.L2,
    "high": RiskLevel.HIGH,
    "3": RiskLevel.L3,
    "l3": RiskLevel.L3,
    "critical": RiskLevel.CRITICAL,
}

_RISK_BUCKETS = {
    "0": 0,
    "l0": 0,
    "low": 0,
    "1": 1,
    "l1": 1,
    "medium": 1,
    "2": 2,
    "l2": 2,
    "high": 2,
    "3": 3,
    "l3": 3,
    "critical": 3,
}


def _enum_value(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    return value


def _normalize(value: Any) -> str:
    value = _enum_value(value)
    return str(value).strip().lower()


def _iter_values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, enum.Enum):
        return (value.value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _read(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _resolve_path(obj: Any, path: str, default: Any = None) -> Any:
    current = obj
    for part in path.split("."):
        current = _read(current, part, default)
        if current is default:
            return default
    return current


def _to_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _resource_name(resource: Any) -> str:
    for key in ("name", "tool_name", "resource", "id", "identifier"):
        found = _read(resource, key)
        if found is not None:
            return str(_enum_value(found))
    return str(_enum_value(resource))


def _principal_id(principal: Any) -> str:
    for key in ("id", "principal_id", "user_id", "subject", "name"):
        found = _read(principal, key)
        if found is not None:
            return str(_enum_value(found))
    return "anonymous"


def _principal_roles(principal: Any) -> set[str]:
    roles = _read(principal, "roles")
    if roles is None:
        roles = _read(principal, "role")
    return {_normalize(role) for role in _iter_values(roles)}


def _match_pattern(pattern: Any, value: Any) -> bool:
    pattern_text = _normalize(pattern)
    value_text = _normalize(value)
    return pattern_text == "*" or fnmatch.fnmatchcase(value_text, pattern_text)


def _match_any(patterns: Iterable[Any], value: Any) -> bool:
    candidates = tuple(patterns)
    if not candidates:
        return True
    return any(_match_pattern(pattern, value) for pattern in candidates)


def _coerce_risk_level(value: Any) -> RiskLevel | None:
    if value is None:
        return None
    if isinstance(value, RiskLevel):
        return value
    normalized = _normalize(value)
    if normalized.startswith("risklevel."):
        normalized = normalized.removeprefix("risklevel.")
    if normalized in _RISK_ALIASES:
        return _RISK_ALIASES[normalized]
    for level in RiskLevel:
        if normalized == level.value:
            return level
    return None


def _risk_level_from_score(score: int) -> RiskLevel:
    if score >= _RISK_ORDER[RiskLevel.CRITICAL]:
        return RiskLevel.CRITICAL
    if score >= _RISK_ORDER[RiskLevel.HIGH]:
        return RiskLevel.HIGH
    if score >= _RISK_ORDER[RiskLevel.MEDIUM]:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


@dataclass(frozen=True)
class AttributeCondition:
    attribute: str
    operator: str = "eq"
    value: Any = None
    source: str = "context"

    def evaluate(self, principal: Any, resource: Any, context: Mapping[str, Any] | None = None) -> bool:
        source_obj: Any
        context = context or {}
        if self.source == "principal":
            source_obj = principal
        elif self.source == "resource":
            source_obj = resource
        else:
            source_obj = context

        actual = _resolve_path(source_obj, self.attribute)
        op = self.operator.lower()

        if op in {"exists", "present"}:
            return actual is not None
        if op in {"missing", "not_exists"}:
            return actual is None
        if op == "eq":
            return actual == self.value
        if op == "ne":
            return actual != self.value
        if op == "in":
            return actual in _iter_values(self.value)
        if op == "not_in":
            return actual not in _iter_values(self.value)
        if op == "contains":
            return self.value in _iter_values(actual)
        if op == "regex":
            return actual is not None and re.search(str(self.value), str(actual)) is not None
        if op in {"gt", "gte", "lt", "lte"}:
            if actual is None:
                return False
            try:
                left = float(actual)
                right = float(self.value)
            except (TypeError, ValueError):
                return False
            if op == "gt":
                return left > right
            if op == "gte":
                return left >= right
            if op == "lt":
                return left < right
            return left <= right
        raise ValueError(f"Unsupported ABAC operator: {self.operator}")


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    effect: PolicyEffect = PolicyEffect.ALLOW
    actions: Sequence[str] = ("*",)
    resources: Sequence[str] = ("*",)
    roles: Sequence[str] = ()
    conditions: Sequence[AttributeCondition] = ()
    priority: int = 0
    description: str = ""

    def matches(self, principal: Any, action: Any, resource: Any, context: Mapping[str, Any] | None = None) -> bool:
        if not _match_any(self.actions, action):
            return False
        if not _match_any(self.resources, _resource_name(resource)):
            return False
        if self.roles:
            required = {_normalize(role) for role in self.roles}
            if required.isdisjoint(_principal_roles(principal)):
                return False
        return all(condition.evaluate(principal, resource, context) for condition in self.conditions)


@dataclass(frozen=True)
class RolePermission:
    role: str
    actions: Sequence[str] = ("*",)
    resources: Sequence[str] = ("*",)

    def allows(self, role: str, action: Any, resource: Any) -> bool:
        return (
            _normalize(role) == _normalize(self.role)
            and _match_any(self.actions, action)
            and _match_any(self.resources, _resource_name(resource))
        )


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    score: int
    approval_required: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SecurityDecision:
    allowed: bool
    approval_required: bool = False
    approved: bool = False
    reason: str = ""
    principal_id: str = "anonymous"
    matched_rule_ids: tuple[str, ...] = ()
    risk: RiskAssessment | None = None


@dataclass
class ApprovalPolicy:
    approval_risk_level: RiskLevel = RiskLevel.HIGH
    score_threshold: int = 70
    sensitive_actions: Sequence[str] = (
        "delete*",
        "exec*",
        "shell*",
        "write_file",
        "filesystem.write",
        "payment*",
        "refund*",
        "deploy*",
        "admin*",
    )
    sensitive_resources: Sequence[str] = ()
    trusted_roles: Sequence[str] = ("admin", "security")

    def assess(self, principal: Any, action: Any, resource: Any, context: Mapping[str, Any] | None = None) -> RiskAssessment:
        context = context or {}
        score = 0
        reasons: list[str] = []

        explicit_score = _read(context, "risk_score")
        if explicit_score is not None:
            try:
                score = max(score, int(explicit_score))
                reasons.append("context_risk_score")
            except (TypeError, ValueError):
                pass

        explicit_level = _coerce_risk_level(_read(context, "risk_level"))
        if explicit_level is not None:
            score = max(score, _RISK_ORDER[explicit_level])
            reasons.append(f"context_risk_level:{explicit_level.value}")

        if _match_any(self.sensitive_actions, action):
            score = max(score, 75)
            reasons.append("sensitive_action")

        if self.sensitive_resources and _match_any(self.sensitive_resources, _resource_name(resource)):
            score = max(score, 75)
            reasons.append("sensitive_resource")

        if not _principal_roles(principal):
            score = max(score, 35)
            reasons.append("missing_roles")

        context_attrs = _to_mapping(context)
        if context_attrs.get("external_network") or context_attrs.get("network_access"):
            score = max(score, 70)
            reasons.append("network_access")
        if context_attrs.get("writes_filesystem") or context_attrs.get("mutates_state"):
            score = max(score, 70)
            reasons.append("state_mutation")

        trusted = not set(map(_normalize, self.trusted_roles)).isdisjoint(_principal_roles(principal))
        if trusted and score > 0:
            score = max(0, score - 10)
            reasons.append("trusted_role_adjustment")

        level = _risk_level_from_score(score)
        approval_required = (
            bool(context_attrs.get("approval_required"))
            or score >= self.score_threshold
            or _RISK_ORDER[level] >= _RISK_ORDER[self.approval_risk_level]
        )
        return RiskAssessment(level=level, score=score, approval_required=approval_required, reasons=tuple(reasons))


@dataclass
class RBACPolicy:
    permissions: Sequence[RolePermission] = ()

    def allows(self, principal: Any, action: Any, resource: Any) -> tuple[bool, tuple[str, ...]]:
        matched: list[str] = []
        for role in _principal_roles(principal):
            for permission in self.permissions:
                if permission.allows(role, action, resource):
                    matched.append(f"role:{permission.role}")
        return bool(matched), tuple(matched)


@dataclass
class SecurityPolicyEngine:
    rules: Sequence[PolicyRule] = ()
    rbac: RBACPolicy = field(default_factory=RBACPolicy)
    approval_policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    default_allow: bool = False

    def evaluate(
        self,
        principal: Any,
        action: Any,
        resource: Any,
        context: Mapping[str, Any] | None = None,
    ) -> SecurityDecision:
        context = context or {}
        principal_id = _principal_id(principal)
        matched: list[str] = []

        for rule in sorted(self.rules, key=lambda item: item.priority, reverse=True):
            if rule.matches(principal, action, resource, context):
                matched.append(rule.rule_id)
                if rule.effect == PolicyEffect.DENY:
                    risk = self.approval_policy.assess(principal, action, resource, context)
                    return SecurityDecision(
                        allowed=False,
                        reason=f"Denied by policy rule {rule.rule_id}",
                        principal_id=principal_id,
                        matched_rule_ids=tuple(matched),
                        risk=risk,
                    )

        allow_rule_ids = [
            rule.rule_id
            for rule in sorted(self.rules, key=lambda item: item.priority, reverse=True)
            if rule.effect == PolicyEffect.ALLOW and rule.matches(principal, action, resource, context)
        ]
        if allow_rule_ids:
            matched.extend(rule_id for rule_id in allow_rule_ids if rule_id not in matched)

        rbac_allowed, rbac_matches = self.rbac.allows(principal, action, resource)
        matched.extend(rule_id for rule_id in rbac_matches if rule_id not in matched)

        allowed_by_policy = bool(allow_rule_ids) or rbac_allowed or self.default_allow
        risk = self.approval_policy.assess(principal, action, resource, context)
        approved = _is_approved(context)

        if not allowed_by_policy:
            return SecurityDecision(
                allowed=False,
                reason="No matching allow policy",
                principal_id=principal_id,
                matched_rule_ids=tuple(matched),
                risk=risk,
            )

        if risk.approval_required and not approved:
            return SecurityDecision(
                allowed=False,
                approval_required=True,
                approved=False,
                reason="Approval required before execution",
                principal_id=principal_id,
                matched_rule_ids=tuple(matched),
                risk=risk,
            )

        return SecurityDecision(
            allowed=True,
            approval_required=risk.approval_required,
            approved=approved,
            reason="Allowed",
            principal_id=principal_id,
            matched_rule_ids=tuple(matched),
            risk=risk,
        )


def _is_approved(context: Mapping[str, Any]) -> bool:
    status = _normalize(_read(context, "approval_status", ""))
    return bool(_read(context, "approved", False)) or status in {"approved", "granted", "allow"}


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _tool_name_for_policy(context: Any, tool: Any) -> str:
    return str(
        _read(context, "tool_name", None)
        or _read(tool, "tool_name", None)
        or _read(tool, "name", None)
        or _resolve_path(tool, "metadata.name", "")
    )


def _server_id_for_policy(context: Any, tool: Any) -> str | None:
    found = (
        _read(context, "server_id", None)
        or _read(tool, "server_id", None)
        or _resolve_path(tool, "metadata.server_id", None)
    )
    return None if found is None else str(found)


def _agent_id_for_policy(context: Any) -> str:
    found = _read(context, "agent_id", None)
    if found is None:
        metadata = _read(context, "metadata", {}) or {}
        found = _read(metadata, "agent_id", None)
    if found is None:
        found = _read(context, "principal", None)
    return str(found or "anonymous")


def _policy_values(policy: Any, *names: str) -> tuple[str, ...]:
    values: list[str] = []
    for name in names:
        for item in _iter_values(_read(policy, name, ())):
            text = str(_enum_value(item)).strip()
            if text:
                values.append(text)
    return tuple(values)


def _policy_id(policy: Any) -> str | None:
    found = _read(policy, "policy_id", None)
    return None if found is None else str(found)


def _risk_bucket(value: Any) -> int:
    normalized = _normalize(value)
    if normalized.startswith("risklevel."):
        normalized = normalized.removeprefix("risklevel.")
    return _RISK_BUCKETS.get(normalized, 0)


def _policy_requires_approval(policy: Any) -> bool:
    return bool(_read(policy, "require_approval", False)) or _risk_bucket(_read(policy, "risk_level")) >= 2


def _agent_matches(patterns: Sequence[str], agent_id: str) -> bool:
    return any(_match_pattern(pattern, agent_id) for pattern in patterns)


def _policy_specificity(policy: Any, server_id: str | None, tool_name: str) -> int | None:
    policy_tool = _read(policy, "tool_name", "*")
    policy_server = _read(policy, "server_id", None)

    tool_exact = _normalize(policy_tool) == _normalize(tool_name)
    tool_wildcard = _normalize(policy_tool) == "*"
    tool_pattern = not tool_exact and not tool_wildcard and _match_pattern(policy_tool, tool_name)
    if not (tool_exact or tool_wildcard or tool_pattern):
        return None

    server_specific = policy_server not in (None, "", "*")
    server_exact = server_specific and server_id is not None and _normalize(policy_server) == _normalize(server_id)
    if server_specific and not server_exact:
        return None

    if server_exact and tool_exact:
        return 300
    if not server_specific and tool_exact:
        return 200
    if server_exact and (tool_wildcard or tool_pattern):
        return 100
    if not server_specific and (tool_wildcard or tool_pattern):
        return 0
    return None


@dataclass(slots=True)
class PolicyAwareSecurity:
    """Security adapter that evaluates the latest ToolPolicy records from a registry.

    线程安全说明：本类不缓存策略、不维护可变运行时状态；并发调用只读取 registry
    返回的快照。registry 本身的并发一致性由其实现负责。
    """

    registry: Any
    default_allow: bool = False

    async def check_permission(self, context: "ToolCallContext", tool: Any, args: Mapping[str, Any] | None = None) -> PolicyDecision:
        policy = await self._resolve_policy(context, tool)
        if policy is None:
            if self.default_allow:
                return PolicyDecision.allowed("no matching policy")
            return PolicyDecision.denied("no matching policy")

        agent_id = _agent_id_for_policy(context)
        denied_agents = _policy_values(policy, "denied_agents", "denied_principals")
        if denied_agents and _agent_matches(denied_agents, agent_id):
            return PolicyDecision.denied("agent denied by policy", _policy_id(policy))

        allowed_agents = _policy_values(policy, "allowed_agents", "allowed_principals")
        if not allowed_agents or not _agent_matches(allowed_agents, agent_id):
            return PolicyDecision.denied("agent not allowed by policy", _policy_id(policy))

        if _policy_requires_approval(policy):
            return PolicyDecision.require_approval("approval required by policy", _policy_id(policy))
        return PolicyDecision.allowed("allowed by policy", _policy_id(policy))

    async def resolve_timeout_ms(self, context: "ToolCallContext", tool: Any) -> int | None:
        policy = await self.resolve_policy(context, tool)
        if policy is None:
            return None
        timeout_ms = _read(policy, "timeout_ms", None)
        if timeout_ms is None:
            return None
        try:
            timeout = int(timeout_ms)
        except (TypeError, ValueError):
            return None
        return timeout if timeout > 0 else None

    async def resolve_policy(self, context: "ToolCallContext", tool: Any) -> "ToolPolicy | None":
        return await self._resolve_policy(context, tool)

    async def _resolve_policy(self, context: "ToolCallContext", tool: Any) -> "ToolPolicy | None":
        tool_name = _tool_name_for_policy(context, tool)
        server_id = _server_id_for_policy(context, tool)
        policies = await self._list_policies(server_id, tool_name)

        candidates: list[tuple[int, int, Any]] = []
        for index, policy in enumerate(policies):
            if _read(policy, "enabled", True) is not True:
                continue
            specificity = _policy_specificity(policy, server_id, tool_name)
            if specificity is not None:
                candidates.append((specificity, index, policy))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))[2]

    async def _list_policies(self, server_id: str | None, tool_name: str) -> tuple[Any, ...]:
        lister = getattr(self.registry, "list_policies", None)
        if lister is None:
            return ()
        try:
            policies = lister(server_id=server_id, tool_name=tool_name, enabled=True)
        except TypeError:
            try:
                policies = lister(server_id=server_id, tool_name=tool_name)
            except TypeError:
                policies = lister()
        resolved = await _maybe_await(policies)
        return tuple(resolved or ())


def assess_risk(
    principal: Any,
    action: Any,
    resource: Any,
    context: Mapping[str, Any] | None = None,
    approval_policy: ApprovalPolicy | None = None,
) -> RiskAssessment:
    return (approval_policy or ApprovalPolicy()).assess(principal, action, resource, context)


def evaluate_policy(
    principal: Any,
    action: Any,
    resource: Any,
    context: Mapping[str, Any] | None = None,
    *,
    rules: Sequence[PolicyRule] = (),
    permissions: Sequence[RolePermission] = (),
    default_allow: bool = False,
    approval_policy: ApprovalPolicy | None = None,
) -> SecurityDecision:
    engine = SecurityPolicyEngine(
        rules=rules,
        rbac=RBACPolicy(permissions=permissions),
        approval_policy=approval_policy or ApprovalPolicy(),
        default_allow=default_allow,
    )
    return engine.evaluate(principal, action, resource, context)


def audit_metadata(decision: SecurityDecision) -> dict[str, Any]:
    risk = decision.risk
    return {
        "allowed": decision.allowed,
        "approval_required": decision.approval_required,
        "approved": decision.approved,
        "reason": decision.reason,
        "principal_id": decision.principal_id,
        "matched_rule_ids": list(decision.matched_rule_ids),
        "risk": None
        if risk is None
        else {
            "level": risk.level.value,
            "score": risk.score,
            "approval_required": risk.approval_required,
            "reasons": list(risk.reasons),
        },
        "evaluated_at": time.time(),
    }


__all__ = [
    "ApprovalPolicy",
    "AttributeCondition",
    "PolicyEffect",
    "PolicyRule",
    "PolicyAwareSecurity",
    "RBACPolicy",
    "RiskAssessment",
    "RiskLevel",
    "RolePermission",
    "SecurityDecision",
    "SecurityPolicyEngine",
    "assess_risk",
    "audit_metadata",
    "evaluate_policy",
]
