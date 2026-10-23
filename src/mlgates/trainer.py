"""Pluggable trainers. Ships a deterministic pure-stdlib logistic regression.

The Trainer protocol is the extension point: swap in any estimator that maps
feature matrices to a Model. The bundled LogisticRegressionTrainer uses batch
gradient descent with zero-init weights and in-trainer standardization, so it
is bit-for-bit deterministic — a requirement for the reproducibility gates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol


class Trainer(Protocol):
    name: str

    def fit(self, X: list[list[float]], y: list[int]) -> "Model": ...


@dataclass
class Model:
    """A trained linear classifier with everything needed to score it."""

    weights: list[float]
    bias: float
    feature_names: list[str]
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    trainer_name: str = "unknown"

    def _standardize(self, x: list[float]) -> list[float]:
        if not self.means:
            return list(x)
        return [
            (v - m) / s if s > 0 else 0.0
            for v, m, s in zip(x, self.means, self.stds)
        ]

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        out = []
        for x in X:
            z = self.bias + sum(
                w * v for w, v in zip(self.weights, self._standardize(x))
            )
            out.append(1.0 / (1.0 + math.exp(-z)))
        return out

    def predict(self, X: list[list[float]]) -> list[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]

    def n_parameters(self) -> int:
        return len(self.weights) + 1

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "feature_names": self.feature_names,
            "means": self.means,
            "stds": self.stds,
            "trainer_name": self.trainer_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        return cls(
            weights=list(d["weights"]),
            bias=float(d["bias"]),
            feature_names=list(d["feature_names"]),
            means=list(d.get("means", [])),
            stds=list(d.get("stds", [])),
            trainer_name=str(d.get("trainer_name", "unknown")),
        )


class LogisticRegressionTrainer:
    """Deterministic batch-gradient-descent logistic regression (pure stdlib)."""

    name = "logreg"

    def __init__(self, lr: float = 0.5, epochs: int = 300, l2: float = 1e-4) -> None:
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2

    def fit(self, X: list[list[float]], y: list[int]) -> Model:
        n = len(X)
        n_feat = len(X[0])
        means = [sum(col) / n for col in zip(*X)]
        stds = []
        for j, col in enumerate(zip(*X)):
            var = sum((v - means[j]) ** 2 for v in col) / n
            stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
        Xs = [
            [(v - m) / s for v, m, s in zip(row, means, stds)] for row in X
        ]
        weights = [0.0] * n_feat
        bias = 0.0
        for _ in range(self.epochs):
            grad_w = [0.0] * n_feat
            grad_b = 0.0
            for xs, target in zip(Xs, y):
                z = bias + sum(w * v for w, v in zip(weights, xs))
                # numerically stable sigmoid
                p = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
                err = p - target
                grad_b += err
                for j in range(n_feat):
                    grad_w[j] += err * xs[j]
            for j in range(n_feat):
                weights[j] -= self.lr * (grad_w[j] / n + self.l2 * weights[j])
            bias -= self.lr * grad_b / n
        return Model(
            weights=weights,
            bias=bias,
            feature_names=[f"f{j}" for j in range(n_feat)],
            means=means,
            stds=stds,
            trainer_name=self.name,
        )


class MajorityBaselineTrainer:
    """Naive baseline: always predicts the training majority class."""

    name = "majority"

    def fit(self, X: list[list[float]], y: list[int]) -> Model:
        majority = 1 if sum(y) * 2 >= len(y) else 0
        n_feat = len(X[0])
        # Encode the majority decision as a constant-logit linear model.
        bias = 10.0 if majority == 1 else -10.0
        return Model(
            weights=[0.0] * n_feat,
            bias=bias,
            feature_names=[f"f{j}" for j in range(n_feat)],
            trainer_name=self.name,
        )


def get_trainer(name: str, lr: float = 0.5, epochs: int = 300) -> Trainer:
    if name == "logreg":
        return LogisticRegressionTrainer(lr=lr, epochs=epochs)
    if name == "majority":
        return MajorityBaselineTrainer()
    raise ValueError(f"unknown trainer: {name!r} (expected 'logreg' or 'majority')")
