"""Promotion: fail-closed decision matrix, approvals, and rollback wiring."""
from __future__ import annotations

import pytest

from mlgates.promotion import (
    PromotionDecider,
    Stage,
    allowed_next,
    sign_approval,
    verify_approval,
)


@pytest.fixture()
def decider():
    return PromotionDecider(require_human_approval=True)


def _ok_kwargs(**over):
    base = dict(
        data_blocked=False, model_blocked=False, regression_blocked=False,
        regression_reason="", approval=None, run_id="run-1",
    )
    base.update(over)
    return base


def test_stage_machine_shape():
    assert allowed_next(Stage.EXPERIMENTAL) == {Stage.STAGING}
    assert allowed_next(Stage.STAGING) == {Stage.CANDIDATE}
    assert allowed_next(Stage.CANDIDATE) == {Stage.PRODUCTION}
    assert allowed_next(Stage.PRODUCTION) == {Stage.STAGING}  # rollback only


def test_experimental_to_staging_when_gates_pass(decider):
    d = decider.decide(Stage.EXPERIMENTAL, **_ok_kwargs())
    assert d.approved and d.to_stage is Stage.STAGING


def test_fail_closed_on_data_gate(decider):
    d = decider.decide(Stage.EXPERIMENTAL, **_ok_kwargs(data_blocked=True))
    assert not d.approved
    assert any("data" in r for r in d.reasons)


def test_fail_closed_on_model_gate(decider):
    d = decider.decide(Stage.EXPERIMENTAL, **_ok_kwargs(model_blocked=True))
    assert not d.approved


def test_staging_to_candidate_requires_clean_regression(decider):
    d = decider.decide(
        Stage.STAGING,
        **_ok_kwargs(regression_blocked=True, regression_reason="p=0.01"),
    )
    assert not d.approved
    assert any("regression" in r for r in d.reasons)


def test_staging_to_candidate_when_clean(decider):
    d = decider.decide(Stage.STAGING, **_ok_kwargs())
    assert d.approved and d.to_stage is Stage.CANDIDATE


def test_production_requires_human_approval(decider):
    d = decider.decide(Stage.CANDIDATE, **_ok_kwargs())
    assert not d.approved
    assert any("approval" in r for r in d.reasons)


def test_production_with_valid_signed_approval(decider):
    approval = sign_approval("run-1", "cto@example.com", "looks good")
    d = decider.decide(Stage.CANDIDATE, **_ok_kwargs(approval=approval))
    assert d.approved and d.to_stage is Stage.PRODUCTION
    assert d.approval is approval


def test_approval_for_other_run_rejected(decider):
    approval = sign_approval("run-OTHER", "cto@example.com", "looks good")
    d = decider.decide(Stage.CANDIDATE, **_ok_kwargs(approval=approval))
    assert not d.approved


def test_tampered_approval_rejected(decider):
    approval = sign_approval("run-1", "cto@example.com", "looks good")
    approval.approver = "mallory@example.com"
    assert not verify_approval(approval, "run-1")
    d = decider.decide(Stage.CANDIDATE, **_ok_kwargs(approval=approval))
    assert not d.approved


def test_approval_not_required_when_policy_disabled():
    decider = PromotionDecider(require_human_approval=False)
    d = decider.decide(Stage.CANDIDATE, **_ok_kwargs())
    assert d.approved and d.to_stage is Stage.PRODUCTION


def test_rollback_from_production(decider):
    record = decider.trigger_rollback(
        Stage.PRODUCTION, "run-9", "post-deploy error-rate spike", "oncall"
    )
    assert record.from_stage is Stage.PRODUCTION
    assert record.to_stage is Stage.STAGING
    assert record.reason == "post-deploy error-rate spike"
    d = record.to_dict()
    assert d["triggered_by"] == "oncall"


def test_rollback_from_non_production_rejected(decider):
    with pytest.raises(ValueError):
        decider.trigger_rollback(Stage.STAGING, "run-9", "x", "oncall")


def test_production_cannot_advance_via_decide(decider):
    approval = sign_approval("run-1", "cto@example.com")
    d = decider.decide(Stage.PRODUCTION, **_ok_kwargs(approval=approval))
    assert not d.approved  # must go through trigger_rollback


def test_decision_serializes(decider):
    d = decider.decide(Stage.EXPERIMENTAL, **_ok_kwargs())
    s = d.to_dict()
    assert s["from_stage"] == "experimental"
    assert s["to_stage"] == "staging"
    assert s["approved"] is True
