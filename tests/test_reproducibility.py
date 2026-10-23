"""Reproducibility: hashes are stable, sensitive to change, and verifiable."""
from __future__ import annotations

import copy

import pytest

from mlgates.data_gen import generate_churn_dataset
from mlgates.manifest import (
    build_manifest,
    code_hash,
    data_hash,
    model_hash,
    verify_manifest,
)
from mlgates.pipeline import SRC_ROOT
from mlgates.trainer import LogisticRegressionTrainer, MajorityBaselineTrainer
from mlgates.data_gen import to_xy


def _model(seed_rows=11, trainer=None):
    rows = generate_churn_dataset(200, seed_rows)
    X, y = to_xy(rows)
    trainer = trainer or LogisticRegressionTrainer(epochs=10)
    return rows, trainer.fit(X, y)


def test_data_hash_deterministic():
    rows = generate_churn_dataset(300, 5)
    assert data_hash(rows) == data_hash(generate_churn_dataset(300, 5))


def test_data_hash_sensitive_to_change():
    rows = generate_churn_dataset(300, 5)
    mutated = copy.deepcopy(rows)
    mutated[0]["tenure_months"] += 1
    assert data_hash(rows) != data_hash(mutated)


def test_code_hash_stable():
    assert code_hash(SRC_ROOT) == code_hash(SRC_ROOT)
    assert len(code_hash(SRC_ROOT)) == 64


def test_model_hash_differs_between_models():
    rows1, m1 = _model(seed_rows=11)
    _, m2 = _model(seed_rows=12)
    assert model_hash(m1) != model_hash(m2)


def test_manifest_reproduce_command_embeds_run_params():
    rows, model = _model()
    manifest = build_manifest(
        run_id="run-abc123", seed=11, n_rows=200, holdout_fraction=0.25,
        trainer="logreg", trainer_epochs=10, config={"seed": 11},
        rows=rows, model=model, src_root=SRC_ROOT,
    )
    cmd = manifest.reproduce_command()
    assert "--seed 11" in cmd
    assert "--rows 200" in cmd
    assert "--trainer logreg" in cmd
    assert "--epochs 10" in cmd
    assert manifest.data_hash[:12] in cmd


def test_verify_manifest_passes_on_identical_inputs():
    rows, model = _model()
    manifest = build_manifest(
        run_id="run-x", seed=11, n_rows=200, holdout_fraction=0.25,
        trainer="logreg", trainer_epochs=10, config={},
        rows=rows, model=model, src_root=SRC_ROOT,
    )
    check = verify_manifest(manifest, rows, SRC_ROOT)
    assert check == {"data_hash_match": True, "code_hash_match": True}


def test_verify_manifest_catches_data_change():
    rows, model = _model()
    manifest = build_manifest(
        run_id="run-x", seed=11, n_rows=200, holdout_fraction=0.25,
        trainer="logreg", trainer_epochs=10, config={},
        rows=rows, model=model, src_root=SRC_ROOT,
    )
    mutated = copy.deepcopy(rows)
    mutated[3]["region"] = "south"
    check = verify_manifest(manifest, mutated, SRC_ROOT)
    assert check["data_hash_match"] is False
    assert check["code_hash_match"] is True
