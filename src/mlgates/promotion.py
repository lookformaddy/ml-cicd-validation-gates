"""Promotion policy: the semantic stage machine, approvals, and rollback.

Stages: experimental -> staging -> candidate -> production.
Transitions are only legal along the arrows, and each step demands its gates:
- experimental -> staging:   data + model gates have no FAIL (WARNs ok)
- staging -> candidate:       regression check passes (no significant regression)
- candidate -> production:    signed human approval for this exact run
- production -> staging:      rollback only (post-deploy regression trigger)

Aggregation is fail-closed: anything not explicitly allowed is rejected, and
every decision carries its reasons so the audit trail is complete.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Stage(str, Enum):
    EXPERIMENTAL = "experimental"
    STAGING = "staging"
    CANDIDATE = "candidate"
    PRODUCTION = "production"


_ALLOWED_TRANSITIONS: dict[Stage, set[Stage]] = {
    Stage.EXPERIMENTAL: {Stage.STAGING},
    Stage.STAGING: {Stage.CANDIDATE},
    Stage.CANDIDATE: {Stage.PRODUCTION},
    Stage.PRODUCTION: {Stage.STAGING},  # rollback only
}


def allowed_next(current: Stage) -> set[Stage]:
    return set(_ALLOWED_TRANSITIONS.get(current, set()))


@dataclass
class Approval:
    """Signed-off human approval record for candidate -> production."""

    run_id: str
    approver: str
    comment: str
    at: str
    signature: str

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "approver": self.approver,
            "comment": self.comment,
            "at": self.at,
            "signature": self.signature,
        }


def sign_approval(run_id: str, approver: str, comment: str = "") -> Approval:
    payload = f"{run_id}:{approver}:{comment}"
    signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return Approval(
        run_id=run_id,
        approver=approver,
        comment=comment,
        at=datetime.now(timezone.utc).isoformat(),
        signature=signature,
    )


def verify_approval(approval: Approval, run_id: str) -> bool:
    expected = sign_approval(run_id, approval.approver, approval.comment).signature
    return approval.run_id == run_id and approval.signature == expected


@dataclass
class PromotionDecision:
    approved: bool
    from_stage: Stage
    to_stage: Stage
    reasons: list[str] = field(default_factory=list)
    approval: Approval | None = None

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "reasons": self.reasons,
            "approval": self.approval.to_dict() if self.approval else None,
        }


@dataclass
class RollbackRecord:
    run_id: str
    from_stage: Stage
    to_stage: Stage
    reason: str
    triggered_by: str
    at: str

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "reason": self.reason,
            "triggered_by": self.triggered_by,
            "at": self.at,
        }


class PromotionDecider:
    """Fail-closed promotion decisions across the stage machine."""

    def __init__(self, require_human_approval: bool = True) -> None:
        self.require_human_approval = require_human_approval

    def decide(
        self,
        current: Stage,
        *,
        data_blocked: bool,
        model_blocked: bool,
        regression_blocked: bool,
        regression_reason: str = "",
        approval: Approval | None = None,
        run_id: str = "",
    ) -> PromotionDecision:
        reasons: list[str] = []
        nxt = next(iter(allowed_next(current)), None)
        if nxt is None:
            return PromotionDecision(False, current, current, ["no legal transition"])

        if current is Stage.EXPERIMENTAL:
            if data_blocked:
                reasons.append("data validation gates FAIL")
            if model_blocked:
                reasons.append("model validation gates FAIL")
            if reasons:
                return PromotionDecision(False, current, nxt, reasons)
            return PromotionDecision(
                True, current, nxt,
                ["data and model gates pass (no FAIL)"],
            )

        if current is Stage.STAGING:
            if data_blocked or model_blocked:
                return PromotionDecision(
                    False, current, nxt,
                    ["upstream gates FAIL; cannot advance to candidate"],
                )
            if regression_blocked:
                return PromotionDecision(
                    False, current, nxt,
                    [f"regression check blocks promotion: {regression_reason}"],
                )
            return PromotionDecision(
                True, current, nxt,
                ["no significant regression vs production; promoted to candidate"],
            )

        if current is Stage.CANDIDATE:
            if data_blocked or model_blocked or regression_blocked:
                return PromotionDecision(
                    False, current, nxt, ["gates FAIL; cannot go to production"]
                )
            if self.require_human_approval:
                if approval is None:
                    return PromotionDecision(
                        False, current, nxt,
                        ["human approval required for production; none provided"],
                    )
                if not verify_approval(approval, run_id):
                    return PromotionDecision(
                        False, current, nxt,
                        ["approval signature invalid or for a different run"],
                    )
                return PromotionDecision(
                    True, current, nxt,
                    [f"approved by {approval.approver} (signed {approval.signature[:12]}...)"],
                    approval=approval,
                )
            return PromotionDecision(
                True, current, nxt, ["human approval not required by policy"]
            )

        # PRODUCTION -> STAGING is rollback-only; handled by trigger_rollback.
        return PromotionDecision(
            False, current, current,
            ["production can only move via rollback (use trigger_rollback)"],
        )

    def trigger_rollback(
        self, current: Stage, run_id: str, reason: str, triggered_by: str
    ) -> RollbackRecord:
        if current is not Stage.PRODUCTION:
            raise ValueError(
                f"rollback is only legal from production (current: {current.value})"
            )
        return RollbackRecord(
            run_id=run_id,
            from_stage=Stage.PRODUCTION,
            to_stage=Stage.STAGING,
            reason=reason,
            triggered_by=triggered_by,
            at=datetime.now(timezone.utc).isoformat(),
        )
