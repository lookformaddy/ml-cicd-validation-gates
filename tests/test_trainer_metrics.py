"""Trainer determinism, correctness, and metric math."""
from __future__ import annotations

import pytest

from mlgates.data_gen import featurize, generate_churn_dataset, to_xy
from mlgates.model_validation import (
    accuracy,
    auc_score,
    evaluate,
    precision_recall_f1,
)
from mlgates.trainer import (
    LogisticRegressionTrainer,
    MajorityBaselineTrainer,
    Model,
    get_trainer,
)


@pytest.fixture()
def xy():
    rows = generate_churn_dataset(400, seed=11)
    return to_xy(rows)


def test_logreg_is_deterministic(xy):
    X, y = xy
    t = LogisticRegressionTrainer(epochs=20)
    m1 = t.fit(X, y)
    m2 = t.fit(X, y)
    assert m1.weights == m2.weights
    assert m1.bias == m2.bias


def test_logreg_learns_signal(xy):
    X, y = xy
    model = LogisticRegressionTrainer(epochs=60).fit(X, y)
    assert accuracy(y, model.predict(X)) > 0.70


def test_predict_proba_in_unit_interval(xy):
    X, y = xy
    model = LogisticRegressionTrainer(epochs=10).fit(X, y)
    for p in model.predict_proba(X):
        assert 0.0 <= p <= 1.0


def test_majority_baseline_predicts_majority(xy):
    X, y = xy
    model = MajorityBaselineTrainer().fit(X, y)
    majority = 1 if sum(y) * 2 >= len(y) else 0
    assert set(model.predict(X)) == {majority}


def test_model_serialization_round_trip(xy):
    X, y = xy
    model = LogisticRegressionTrainer(epochs=10).fit(X, y)
    clone = Model.from_dict(model.to_dict())
    assert clone.predict(X) == model.predict(X)
    assert clone.n_parameters() == model.n_parameters() == len(X[0]) + 1


def test_get_trainer_unknown():
    with pytest.raises(ValueError):
        get_trainer("xgboost")


def test_auc_perfect_ranking():
    assert auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_auc_worst_ranking():
    assert auc_score([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_auc_constant_scores():
    assert auc_score([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auc_single_class():
    assert auc_score([1, 1, 1], [0.9, 0.8, 0.7]) == pytest.approx(0.5)


def test_precision_recall_f1_known():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 0, 1, 1]
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)


def test_evaluate_returns_all_metrics(xy):
    X, y = xy
    model = LogisticRegressionTrainer(epochs=20).fit(X, y)
    metrics = evaluate(model, X, y)
    assert set(metrics) == {"accuracy", "auc", "precision", "recall", "f1"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
