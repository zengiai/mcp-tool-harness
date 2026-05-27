"""Core domain models for MCP Tool Harness.

The module intentionally depends only on Python's standard library so it can be
used by adapters, tests, and bootstrap code without dependency ordering risks.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _wire_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sort_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return _wire_datetime(value)
    if is_dataclass(value):
        return {
            item.name: _canonicalize(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize(item) for item in value]
        return sorted(normalized, key=_sort_key)
    if isinstance(value, bytes):
        return value.hex()
    return value


def stable_json_dumps(value: Any) -> str:
    """Serialize a value into deterministic JSON."""

    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_json_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible or dataclass values."""

    payload = stable_json_dumps(value).encode("utf-8")
    return sha256(payload).hexdigest()


def schema_hash(input_schema: Mapping[str, Any], output_schema: Mapping[str, Any] | None = None) -> str:
    """Return a stable hash for a tool input/output schema pair."""

    payload: dict[str, Any] = {"input": input_schema}
    if output_schema is not None:
        payload["output"] = output_schema
    return stable_json_hash(payload)


def tool_identity_key(server_id: str, name: str, version: str) -> tuple[str, str, str]:
    return server_id, name, version


def _generated_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{stable_json_hash(payload)[:32]}"


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


class JsonModelMixin:
    """Small helper shared by dataclass models."""

    def to_dict(self) -> dict[str, Any]:
        return _canonicalize(self)

    def to_json(self) -> str:
        return stable_json_dumps(self)

    def stable_hash(self) -> str:
        return stable_json_hash(self)


class ToolProtocol(str, Enum):
    MCP = "mcp"


class ToolServerStatus(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    UNHEALTHY = "unhealthy"


class AuthType(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    CUSTOM = "custom"


class RiskLevel(str, Enum):
    L0 = "l0"
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def _coerce_risk_level(value: Any) -> RiskLevel:
    if isinstance(value, RiskLevel):
        return value
    normalized = str(value).strip().lower()
    if normalized.startswith("risklevel."):
        normalized = normalized.removeprefix("risklevel.")
    aliases = {
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
    if normalized in aliases:
        return aliases[normalized]
    for level in RiskLevel:
        if normalized == level.name.lower() or normalized == level.value.lower():
            return level
    raise ValueError("risk_level must be one of l0/l1/l2/l3 or LOW/MEDIUM/HIGH/CRITICAL")


class DecisionEffect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ToolCallStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    DENIED = "denied"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class ToolServer(JsonModelMixin):
    server_id: str
    endpoint: str
    name: str = ""
    protocol: ToolProtocol = ToolProtocol.MCP
    version: str = "1.0.0"
    status: ToolServerStatus = ToolServerStatus.ACTIVE
    auth_type: AuthType = AuthType.NONE
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_text(self.server_id, "server_id")
        _require_text(self.endpoint, "endpoint")
        if not self.name:
            self.name = self.server_id
        self.capabilities = tuple(self.capabilities)


@dataclass(slots=True)
class ToolSpec(JsonModelMixin):
    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    server_id: str = "local"
    version: str = "1.0.0"
    enabled: bool = True
    policy_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_id: str = ""
    schema_hash: str = ""
    registered_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.description, "description")
        _require_text(self.server_id, "server_id")
        _require_text(self.version, "version")
        self.schema_hash = schema_hash(self.input_schema, self.output_schema)
        if not self.tool_id:
            self.tool_id = _generated_id(
                "tool",
                {
                    "server_id": self.server_id,
                    "name": self.name,
                    "version": self.version,
                },
            )

    @property
    def identity(self) -> tuple[str, str, str]:
        return tool_identity_key(self.server_id, self.name, self.version)

    @property
    def tool_name(self) -> str:
        """Compatibility alias used by gateway and adapter layers."""

        return self.name


@dataclass(slots=True)
class ToolPolicy(JsonModelMixin):
    tool_name: str = "*"
    server_id: str | None = None
    risk_level: RiskLevel = RiskLevel.L0
    enabled: bool = True
    require_approval: bool = False
    allowed_principals: frozenset[str] = field(default_factory=frozenset)
    denied_principals: frozenset[str] = field(default_factory=frozenset)
    rate_limit_per_minute: int | None = None
    rate_limits: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    timeout_ms: int | None = None
    circuit_failure_threshold: int = 5
    circuit_reset_timeout_seconds: int = 30
    circuit_success_threshold: int = 2
    circuit_half_open_max_calls: int = 1
    audit_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    policy_id: str = ""
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    allowed_agents: frozenset[str] = field(default_factory=lambda: frozenset({"*"}))
    denied_agents: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        _require_text(self.tool_name, "tool_name")
        if self.server_id is not None:
            _require_text(self.server_id, "server_id")
        self.risk_level = _coerce_risk_level(self.risk_level)
        if self.rate_limit_per_minute is not None and self.rate_limit_per_minute <= 0:
            raise ValueError("rate_limit_per_minute must be positive")
        self.rate_limits = tuple(dict(item) for item in self.rate_limits)
        if self.timeout_ms is not None and self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if self.circuit_failure_threshold <= 0:
            raise ValueError("circuit_failure_threshold must be positive")
        if self.circuit_reset_timeout_seconds <= 0:
            raise ValueError("circuit_reset_timeout_seconds must be positive")
        if self.circuit_success_threshold <= 0:
            raise ValueError("circuit_success_threshold must be positive")
        if self.circuit_half_open_max_calls <= 0:
            raise ValueError("circuit_half_open_max_calls must be positive")
        self.allowed_principals = frozenset(self.allowed_principals)
        self.denied_principals = frozenset(self.denied_principals)
        self.allowed_agents = frozenset(self.allowed_agents)
        self.denied_agents = frozenset(self.denied_agents)
        if not self.policy_id:
            self.policy_id = _generated_id(
                "policy",
                {
                    "server_id": self.server_id or "*",
                    "tool_name": self.tool_name,
                },
            )


@dataclass(slots=True)
class ToolCallContext(JsonModelMixin):
    request_id: str
    principal: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    server_id: str | None = None
    tenant_id: str | None = None
    trace_id: str | None = None
    idempotency_key: str | None = None
    auth_claims: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    deadline_at: datetime | None = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_text(self.request_id, "request_id")
        _require_text(self.principal, "principal")
        _require_text(self.tool_name, "tool_name")
        if self.server_id is not None:
            _require_text(self.server_id, "server_id")

    @property
    def call_id(self) -> str:
        """Compatibility alias for designs that call the request id a call id."""

        return self.request_id

    @property
    def agent_id(self) -> str:
        """Return the agent identity used by runtime isolation keys."""

        return str(self.metadata.get("agent_id") or self.principal)


@dataclass(slots=True)
class ToolResult(JsonModelMixin):
    request_id: str
    success: bool
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None
    status: ToolCallStatus = ToolCallStatus.SUCCEEDED
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.request_id, "request_id")
        if not self.success and self.status == ToolCallStatus.SUCCEEDED:
            self.status = ToolCallStatus.FAILED
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must be greater than or equal to started_at")

    @property
    def latency_ms(self) -> float:
        return (self.finished_at - self.started_at).total_seconds() * 1000

    @property
    def data(self) -> Any:
        """Compatibility alias for callers that expect result data."""

        return self.output

    @classmethod
    def success_result(
        cls,
        *,
        call_id: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
        data: Any = None,
        output: Any = None,
    ) -> "ToolResult":
        result = cls(
            request_id=request_id or call_id or f"request_{uuid.uuid4().hex}",
            success=True,
            output=data if output is None else output,
            status=ToolCallStatus.SUCCEEDED,
        )
        if trace_id is not None:
            result.metadata["trace_id"] = trace_id
        return result

    @classmethod
    def failed(
        cls,
        *,
        call_id: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
        error_code: str = "TOOL_FAILED",
        error_message: str = "tool failed",
        status: ToolCallStatus = ToolCallStatus.FAILED,
    ) -> "ToolResult":
        result = cls(
            request_id=request_id or call_id or f"request_{uuid.uuid4().hex}",
            success=False,
            error_code=error_code,
            error_message=error_message,
            status=status,
        )
        if trace_id is not None:
            result.metadata["trace_id"] = trace_id
        return result

    @classmethod
    def rejected(
        cls,
        *,
        call_id: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
        error_code: str = "REJECTED",
        error_message: str = "tool invocation rejected",
    ) -> "ToolResult":
        return cls.failed(
            call_id=call_id,
            request_id=request_id,
            trace_id=trace_id,
            error_code=error_code,
            error_message=error_message,
            status=ToolCallStatus.DENIED,
        )


@dataclass(slots=True)
class PolicyDecision(JsonModelMixin):
    effect: DecisionEffect
    reason: str
    policy_id: str | None = None
    approval_required: bool = False
    rate_limited: bool = False
    circuit_open: bool = False
    ttl_seconds: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    decision_id: str = field(default_factory=lambda: f"decision_{uuid.uuid4().hex}")
    decided_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_text(self.reason, "reason")
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if self.effect == DecisionEffect.REQUIRE_APPROVAL:
            self.approval_required = True
        if self.effect == DecisionEffect.DENY and self.approval_required:
            raise ValueError("denied decisions cannot require approval")

    @classmethod
    def allow(cls, reason: str = "allowed", policy_id: str | None = None) -> "PolicyDecision":
        return cls(effect=DecisionEffect.ALLOW, reason=reason, policy_id=policy_id)

    @classmethod
    def deny(cls, reason: str, policy_id: str | None = None) -> "PolicyDecision":
        return cls(effect=DecisionEffect.DENY, reason=reason, policy_id=policy_id)

    @classmethod
    def require_approval(cls, reason: str, policy_id: str | None = None) -> "PolicyDecision":
        return cls(
            effect=DecisionEffect.REQUIRE_APPROVAL,
            reason=reason,
            policy_id=policy_id,
            approval_required=True,
        )

    @property
    def decision(self) -> DecisionEffect:
        """Compatibility alias for older gateway code."""

        return self.effect

    @property
    def reason_code(self) -> str | None:
        return self.metadata.get("reason_code")

    @classmethod
    def allowed(cls, reason: str = "allowed", policy_id: str | None = None) -> "PolicyDecision":
        return cls.allow(reason=reason, policy_id=policy_id)

    @classmethod
    def denied(cls, reason: str, policy_id: str | None = None) -> "PolicyDecision":
        return cls.deny(reason=reason, policy_id=policy_id)


@dataclass(slots=True)
class ToolCallRecord(JsonModelMixin):
    context: ToolCallContext
    decision: PolicyDecision
    result: ToolResult | None = None
    tool_id: str | None = None
    server_id: str | None = None
    status: ToolCallStatus | None = None
    record_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex}")
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.server_id is None:
            self.server_id = self.context.server_id
        if self.status is None:
            self.status = self._infer_status()

    def _infer_status(self) -> ToolCallStatus:
        if self.result is not None:
            return self.result.status
        if self.decision.rate_limited:
            return ToolCallStatus.RATE_LIMITED
        if self.decision.circuit_open:
            return ToolCallStatus.CIRCUIT_OPEN
        if self.decision.effect == DecisionEffect.DENY:
            return ToolCallStatus.DENIED
        if self.decision.effect == DecisionEffect.REQUIRE_APPROVAL:
            return ToolCallStatus.PENDING_APPROVAL
        return ToolCallStatus.SUCCEEDED


@dataclass(slots=True)
class AgentRunRecord(JsonModelMixin):
    run_id: str
    request_id: str
    trace_id: str
    agent_id: str
    provider: str = ""
    model: str = ""
    prompt_hash: str = ""
    status: str = "succeeded"
    final_answer: str = ""
    tool_call_count: int = 0
    error: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        _require_text(self.request_id, "request_id")
        _require_text(self.trace_id, "trace_id")
        _require_text(self.agent_id, "agent_id")
        if self.tool_call_count < 0:
            raise ValueError("tool_call_count must be zero or positive")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must be greater than or equal to started_at")


@dataclass(slots=True)
class AgentToolCallRecord(JsonModelMixin):
    run_id: str
    request_id: str
    trace_id: str
    tool_call_id: str
    round_index: int
    step_index: int
    model_tool_name: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    status: ToolCallStatus = ToolCallStatus.SUCCEEDED
    result: Any = None
    error: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    cached: bool = False
    server_id: str | None = None
    tool_version: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    record_id: str = field(default_factory=lambda: f"agent_tool_call_{uuid.uuid4().hex}")

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        _require_text(self.request_id, "request_id")
        _require_text(self.trace_id, "trace_id")
        _require_text(self.tool_call_id, "tool_call_id")
        _require_text(self.model_tool_name, "model_tool_name")
        _require_text(self.tool_name, "tool_name")
        if self.round_index < 1:
            raise ValueError("round_index must be positive")
        if self.step_index < 1:
            raise ValueError("step_index must be positive")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must be greater than or equal to started_at")


@dataclass(slots=True)
class ApprovalTask(JsonModelMixin):
    context: ToolCallContext
    reason: str
    decision: PolicyDecision | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_by: str | None = None
    approver: str | None = None
    expires_at: datetime | None = None
    decided_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    approval_id: str = field(default_factory=lambda: f"approval_{uuid.uuid4().hex}")
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_text(self.reason, "reason")
        if self.requested_by is None:
            self.requested_by = self.context.principal

    def approve(self, approver: str) -> None:
        _require_text(approver, "approver")
        self.status = ApprovalStatus.APPROVED
        self.approver = approver
        self.decided_at = utc_now()

    def reject(self, approver: str) -> None:
        _require_text(approver, "approver")
        self.status = ApprovalStatus.REJECTED
        self.approver = approver
        self.decided_at = utc_now()
