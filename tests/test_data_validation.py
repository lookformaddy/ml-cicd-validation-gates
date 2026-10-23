"""Data-validation gates: every failure mode plus the clean path."""
from __future__ import annotations

import copy

import pytest

from mlgates.data_gen import generate_churn_dataset
from mlgates.data_validation import run_data_gates
from mlgates.gates import GateStatus


@pytest.fixture()
def policy(cfg):
    return cfg.data


def _gate(report, name):
    return next(r for r in report.results if r.gate == name)


def test_clean_data_all_pass(clean_rows, policy):
    report = run_data_gates(clean_rows, policy)
    assert not report.blocked
    assert report.overall is GateStatus.PASS
    assert len(report.results) == 5


def test_missing_column_fails(clean_rows, policy):
    rows = [{k: v for k, v in r.items() if k != "region"} for r in clean_rows]
    report = run_data_gates(rows, policy)
    gate = _gate(report, "schema_drift")
    assert gate.status is GateStatus.FAIL
    assert gate.evidence["missing"] == ["region"]
    assert report.blocked


def test_new_column_warns_not_blocks(clean_rows, policy):
    rows = [dict(r, loyalty_tier="gold") for r in clean_rows]
    report = run_data_gates(rows, policy)
    gate = _gate(report, "schema_drift")
    assert gate.status is GateStatus.WARN
    assert gate.evidence["new"] == ["loyalty_tier"]
    assert not report.blocked  # WARNs never block


def test_dtype_change_fails(clean_rows, policy):
    rows = [dict(r, tenure_months=str(r["tenure_months"])) for r in clean_rows]
    report = run_data_gates(rows, policy)
    gate = _gate(report, "schema_drift")
    assert gate.status is GateStatus.FAIL
    assert gate.evidence["dtype_changed"] == ["tenure_months"]


def test_range_violation_fails(clean_rows, policy):
    rows = copy.deepcopy(clean_rows)
    n_bad = int(len(rows) * 0.05)
    for r in rows[:n_bad]:
        r["monthly_spend"] = 99999.0
    report = run_data_gates(rows, policy)
    gate = _gate(report, "range_checks")
    assert gate.status is GateStatus.FAIL
    assert gate.evidence["violations"]["monthly_spend"]["count"] == n_bad


def test_null_rate_exceeds_threshold_fails(clean_rows, policy):
    rows = copy.deepcopy(clean_rows)
    n_null = int(len(rows) * 0.10)
    for r in rows[:n_null]:
        r["region"] = None
    report = run_data_gates(rows, policy)
    gate = _gate(report, "null_rate")
    assert gate.status is GateStatus.FAIL
    assert gate.evidence["worst_column"] == "region"
    assert gate.evidence["worst_rate"] == pytest.approx(0.10)


def test_null_rate_elevated_warns(clean_rows, policy):
    rows = copy.deepcopy(clean_rows)
    n_null = int(len(rows) * 0.03)  # above 50% of the 5% threshold, below it
    for r in rows[:n_null]:
        r["region"] = None
    report = run_data_gates(rows, policy)
    assert _gate(report, "null_rate").status is GateStatus.WARN


def test_class_imbalance_fails(policy):
    rows = generate_churn_dataset(200, seed=3)
    rows = [dict(r, is_churned=0) for r in rows]
    report = run_data_gates(rows, policy)
    gate = _gate(report, "class_balance")
    assert gate.status is GateStatus.FAIL
    assert gate.evidence["minority_fraction"] == pytest.approx(0.0)


def test_duplicate_rate_fails(clean_rows, policy):
    rows = list(clean_rows) + clean_rows[: int(len(clean_rows) * 0.05)]
    report = run_data_gates(rows, policy)
    gate = _gate(report, "duplicate_rate")
    assert gate.status is GateStatus.FAIL
    assert gate.evidence["duplicate_rate"] > policy.max_duplicate_rate


def test_gate_evidence_is_machine_readable(clean_rows, policy):
    report = run_data_gates(clean_rows, policy)
    d = report.to_dict()
    assert d["stage"] == "data_validation"
    assert d["overall"] == "pass"
    assert all("evidence" in r for r in d["results"])
