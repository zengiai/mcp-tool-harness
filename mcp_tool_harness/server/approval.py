"""Approval and permission policy primitives for tool invocation."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4


ALLOW = "allow"
DENY = "deny"
REQUIRES_APPROVAL = "requires_approval"


@dataclass(frozen=True)
class ApprovalDecision:
    """Result of evaluating whether a tool call can proceed."""

    status: str
    reason: str = ""
    approval_id: str | None = None

    @property
    def allowed(self) -> bool:
        return self.status == ALLOW


class ApprovalPolicy:
    """In-process approval policy.

    The policy intentionally stays small: it supports exact names and shell-style
    patterns so the SDK is runnable before an external IAM or approval service is
    attached.
    """

    def __init__(
        self,
        *,
        denied_tools: Iterable[str] | None = None,
        approval_required_tools: Iterable[str] | None = None,
        allowed_tools: Iterable[str] | None = None,
        approval_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._denied_tools = tuple(denied_tools or ())
        self._approval_required_tools = tuple(approval_required_tools or ())
        self._allowed_tools = tuple(allowed_tools) if allowed_tools is not None else None
        self._approval_id_factory = approval_id_factory or (lambda: f"approval-{uuid4().hex}")

    def evaluate(
        self,
        *,
        tool_name: str,
        principal: str | Mapping[str, Any] | None = None,
        arguments: Mapping[str, Any] | None = None,
    ) -> ApprovalDecision:
        """Return the approval decision for one invocation."""

        del arguments  # Reserved for attribute-based policies.
        subject = _principal_subject(principal)

        if _matches(tool_name, self._denied_tools):
            return ApprovalDecision(
                status=DENY,
                reason=f"principal '{subject}' is not allowed to call tool '{tool_name}'",
            )

        if self._allowed_tools is not None and not _matches(tool_name, self._allowed_tools):
            return ApprovalDecision(
                status=DENY,
                reason=f"tool '{tool_name}' is outside the configured allow list",
            )

        if _matches(tool_name, self._approval_required_tools):
            return ApprovalDecision(
                status=REQUIRES_APPROVAL,
                reason=f"tool '{tool_name}' requires human approval",
                approval_id=self._approval_id_factory(),
            )

        return ApprovalDecision(status=ALLOW, reason="allowed")


def _matches(tool_name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatchcase(tool_name, pattern) for pattern in patterns)


def _principal_subject(principal: str | Mapping[str, Any] | None) -> str:
    if principal is None:
        return "anonymous"
    if isinstance(principal, str):
        return principal
    subject = principal.get("subject") or principal.get("user") or principal.get("id")
    return str(subject or "anonymous")
