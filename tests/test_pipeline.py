"""End-to-end pipeline: happy path, blocked path, determinism, approvals."""
from __future__ import annotations

import pytest

from mlgates.config import ModelGatePolicy, PipelineConfig
from mlgates.manifest import verify_manifest
from mlgates.pipeline import SRC_ROOT, run_pipeline
from mlgates.promotion import Stage, sign_approval
from mlgates.data_gen import generate_churn_dataset


def test_happy_path_reaches_candidate(cfg):
    result = run_pipeline(cfg)
    assert not result.blocked
    assert not result.data_report.blocked
    assert not result.model_report.blocked
    assert not result.regression.blocked
    assert result.furthest_stage is Stage.CANDIDATE
    # production still gated on human approval
    assert result.progression[-1].approved is False
    assert result.progression[-1].to_stage is Stage.PRODUCTION


def test_blocked_on_impossible_metric_floor(cfg):
    base = cfg
    model_policy = ModelGatePolicy(
        metric_floors={"accuracy": 0.999, "auc": 0.999},
        min_delta_vs_baseline=base.model.min_delta_vs_baseline,
        max_slice_regression=base.model.max_slice_regression,
        min_slice_samples=base.model.min_slice_samples,
        max_p99_latency_ms=base.model.max_p99_latency_ms,
        max_model_bytes=base.model.max_model_bytes,
    )
    bad = PipelineConfig(
        seed=base.seed, n_rows=base.n_rows, holdout_fraction=base.holdout_fraction,
        trainer=base.trainer, trainer_epochs=base.trainer_epochs,
        trainer_lr=base.trainer_lr, production_epochs=base.production_epochs,
        slice_columns=base.slice_columns, data=base.data, model=model_policy,
        regression=base.regression, promotion=base.promotion,
    )
    result = run_pipeline(bad)
    assert result.blocked
    assert result.model_report.blocked
    assert result.furthest_stage is Stage.EXPERIMENTAL
    assert result.progression[0].approved is False


def test_pipeline_is_deterministic(cfg):
    r1 = run_pipeline(cfg)
    r2 = run_pipeline(cfg)
    assert r1.manifest.data_hash == r2.manifest.data_hash
    assert r1.candidate_metrics["accuracy"] == pytest.approx(
        r2.candidate_metrics["accuracy"]
    )
    assert r1.manifest.reproduce_command() == r2.manifest.reproduce_command()


def test_manifest_verifies_against_inputs(cfg):
    result = run_pipeline(cfg)
    rows = generate_churn_dataset(cfg.n_rows, cfg.seed)
    check = verify_manifest(result.manifest, rows, SRC_ROOT)
    assert check["data_hash_match"] is True
    assert check["code_hash_match"] is True


def test_signed_approval_promotes_to_production(cfg):
    run_id = "run-fixed-1234"
    approval = sign_approval(run_id, "vp-eng@example.com", "ship it")
    result = run_pipeline(cfg, approval=approval, run_id=run_id)
    assert result.furthest_stage is Stage.PRODUCTION
    final = result.progression[-1]
    assert final.approved and final.to_stage is Stage.PRODUCTION
    assert final.approval.approver == "vp-eng@example.com"


def test_summary_json_shape(cfg):
    result = run_pipeline(cfg)
    s = result.summary()
    assert s["run_id"] == result.run_id
    assert "manifest" in s and "reproduce_command" in s["manifest"]
    assert s["data_gates"]["overall"] == "pass"
    assert isinstance(s["progression"], list) and len(s["progression"]) >= 3
