"""Model-validation gates: floors, slice gates, latency and size budgets."""
from __future__ import annotations

import pytest

from mlgates.config import ModelGatePolicy
from mlgates.data_gen import split_rows, to_xy
from mlgates.gates import GateStatus
from mlgates.model_validation import (
    baseline_delta_gate,
    evaluate,
    latency_budget_gate,
    metric_floor_gate,
    model_size_gate,
    run_model_gates,
    slice_gate,
)
from mlgates.trainer import LogisticRegressionTrainer, MajorityBaselineTrainer


@pytest.fixture()
def trained(cfg, clean_rows):
    train_rows, holdout_rows = split_rows(clean_rows, cfg.holdout_fraction, cfg.seed)
    X_train, y_train = to_xy(train_rows)
    X_hold, y_hold = to_xy(holdout_rows)
    candidate = LogisticRegressionTrainer(epochs=cfg.trainer_epochs).fit(X_train, y_train)
    naive = MajorityBaselineTrainer().fit(X_train, y_train)
    return {
        "candidate": candidate,
        "naive": naive,
        "rows": holdout_rows,
        "X": X_hold,
        "y": y_hold,
        "metrics": evaluate(candidate, X_hold, y_hold),
        "naive_metrics": evaluate(naive, X_hold, y_hold),
    }


def test_metric_floors_pass(trained, cfg):
    gate = metric_floor_gate(trained["metrics"], cfg.model)
    assert gate.status is GateStatus.PASS


def test_metric_floor_breach_fails(trained):
    policy = ModelGatePolicy(metric_floors={"accuracy": 0.999, "auc": 0.999})
    gate = metric_floor_gate(trained["metrics"], policy)
    assert gate.status is GateStatus.FAIL
    assert "breaches" in gate.evidence


def test_baseline_delta_pass(trained, cfg):
    gate = baseline_delta_gate(trained["metrics"], trained["naive_metrics"], cfg.model)
    assert gate.status is GateStatus.PASS
    assert gate.evidence["deltas"]["accuracy"] >= 0


def test_baseline_delta_too_small_fails(trained):
    policy = ModelGatePolicy(min_delta_vs_baseline=0.50)
    gate = baseline_delta_gate(trained["metrics"], trained["naive_metrics"], policy)
    assert gate.status is GateStatus.FAIL


def test_slice_gate_passes_when_no_regression():
    cand = {"region": {"west": {"accuracy": 0.80, "n": 100.0}}}
    base = {"region": {"west": {"accuracy": 0.81, "n": 100.0}}}
    gate = slice_gate(cand, base, ModelGatePolicy(max_slice_regression=0.03))
    assert gate.status is GateStatus.PASS


def test_slice_gate_blocks_regression():
    cand = {"region": {"west": {"accuracy": 0.70, "n": 100.0}}}
    base = {"region": {"west": {"accuracy": 0.80, "n": 100.0}}}
    gate = slice_gate(cand, base, ModelGatePolicy(max_slice_regression=0.03))
    assert gate.status is GateStatus.FAIL
    assert "region=west" in gate.evidence["regressions"]
    assert gate.evidence["regressions"]["region=west"]["drop"] == pytest.approx(-0.10)


def test_latency_budget_pass(trained, cfg):
    gate = latency_budget_gate(trained["candidate"], trained["X"], cfg.model)
    assert gate.status is GateStatus.PASS
    assert gate.evidence["p99_ms"] >= 0


def test_latency_budget_fail_on_zero_budget(trained):
    policy = ModelGatePolicy(max_p99_latency_ms=0.0)
    gate = latency_budget_gate(trained["candidate"], trained["X"], policy)
    assert gate.status is GateStatus.FAIL


def test_model_size_pass(trained, cfg):
    gate = model_size_gate(trained["candidate"], cfg.model)
    assert gate.status is GateStatus.PASS
    assert gate.evidence["n_parameters"] == len(trained["X"][0]) + 1


def test_model_size_fail_on_tiny_budget(trained):
    policy = ModelGatePolicy(max_model_bytes=10)
    gate = model_size_gate(trained["candidate"], policy)
    assert gate.status is GateStatus.FAIL


def test_run_model_gates_end_to_end(trained, cfg, clean_rows):
    from mlgates.model_validation import slice_metrics

    naive_slices = {
        col: slice_metrics(
            trained["naive"], trained["rows"], trained["X"], trained["y"],
            col, cfg.model.min_slice_samples,
        )
        for col in cfg.slice_columns
    }
    report, metrics, slices = run_model_gates(
        trained["candidate"], trained["rows"], trained["X"], trained["y"],
        trained["naive"], trained["naive_metrics"], naive_slices,
        cfg.model, cfg.slice_columns,
    )
    assert not report.blocked
    assert report.overall is GateStatus.PASS
    assert len(report.results) == 5
    assert set(slices) == set(cfg.slice_columns)
    assert metrics["accuracy"] > 0.65
