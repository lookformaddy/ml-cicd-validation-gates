"""Model validation gates: metric floors, slice gates, latency and size budgets.

Also hosts the pure-stdlib metric implementations (accuracy, AUC, precision,
recall, F1) and slice evaluation used by both the gates and the pipeline.
"""
from __future__ import annotations

import json
import time

from .config import ModelGatePolicy
from .data_gen import Row
from .gates import GateReport, GateResult, GateStatus
from .trainer import Model


# ---------------------------------------------------------------- metrics ---

def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def auc_score(y_true: list[int], y_score: list[float]) -> float:
    """Mann-Whitney U AUC with average ranks for ties (pure stdlib)."""
    pairs = sorted(zip(y_score, y_true))
    n = len(pairs)
    n_pos = sum(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based ranks i+1 .. j
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def precision_recall_f1(
    y_true: list[int], y_pred: list[int]
) -> tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(model: Model, X: list[list[float]], y: list[int]) -> dict[str, float]:
    preds = model.predict(X)
    probas = model.predict_proba(X)
    precision, recall, f1 = precision_recall_f1(y, preds)
    return {
        "accuracy": accuracy(y, preds),
        "auc": auc_score(y, probas),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def slice_metrics(
    model: Model,
    rows: list[Row],
    X: list[list[float]],
    y: list[int],
    slice_column: str,
    min_samples: int,
) -> dict[str, dict[str, float]]:
    """Per-slice accuracy (+n) for each value of `slice_column` with enough rows."""
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(str(row[slice_column]), []).append(i)
    out: dict[str, dict[str, float]] = {}
    preds = model.predict(X)
    for value, idx in sorted(groups.items()):
        if len(idx) < min_samples:
            continue
        ys = [y[i] for i in idx]
        ps = [preds[i] for i in idx]
        out[value] = {"accuracy": accuracy(ys, ps), "n": float(len(idx))}
    return out


# ------------------------------------------------------------------ gates ---

def metric_floor_gate(
    metrics: dict[str, float], policy: ModelGatePolicy
) -> GateResult:
    breaches = {
        m: {"value": metrics.get(m, 0.0), "floor": floor}
        for m, floor in policy.metric_floors.items()
        if metrics.get(m, 0.0) < floor
    }
    if breaches:
        worst = min(breaches, key=lambda m: breaches[m]["value"] - breaches[m]["floor"])
        return GateResult(
            "metric_floors", GateStatus.FAIL,
            f"{worst}={breaches[worst]['value']:.4f} below floor {breaches[worst]['floor']:.4f}",
            {"breaches": breaches, "metrics": metrics},
        )
    return GateResult(
        "metric_floors", GateStatus.PASS,
        "all metrics above floors", {"metrics": metrics, "floors": policy.metric_floors},
    )


def baseline_delta_gate(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    policy: ModelGatePolicy,
) -> GateResult:
    regressions = {}
    for m, floor in policy.metric_floors.items():
        delta = metrics.get(m, 0.0) - baseline_metrics.get(m, 0.0)
        if delta < policy.min_delta_vs_baseline:
            regressions[m] = {
                "candidate": metrics.get(m, 0.0),
                "baseline": baseline_metrics.get(m, 0.0),
                "delta": delta,
                "required_delta": policy.min_delta_vs_baseline,
            }
    if regressions:
        return GateResult(
            "baseline_delta", GateStatus.FAIL,
            f"candidate fails to beat baseline by {policy.min_delta_vs_baseline} "
            f"on {sorted(regressions)}",
            {"regressions": regressions},
        )
    deltas = {
        m: metrics.get(m, 0.0) - baseline_metrics.get(m, 0.0)
        for m in policy.metric_floors
    }
    return GateResult(
        "baseline_delta", GateStatus.PASS,
        "candidate beats baseline on all floored metrics", {"deltas": deltas},
    )


def slice_gate(
    candidate_slices: dict[str, dict[str, dict[str, float]]],
    baseline_slices: dict[str, dict[str, dict[str, float]]],
    policy: ModelGatePolicy,
) -> GateResult:
    """No protected slice may regress vs the baseline beyond the tolerance."""
    regressions: dict[str, dict] = {}
    for col, slices in candidate_slices.items():
        for value, stats in slices.items():
            base = baseline_slices.get(col, {}).get(value)
            if base is None:
                continue
            drop = stats["accuracy"] - base["accuracy"]
            if drop < -policy.max_slice_regression:
                regressions[f"{col}={value}"] = {
                    "candidate_accuracy": stats["accuracy"],
                    "baseline_accuracy": base["accuracy"],
                    "drop": drop,
                    "n": stats["n"],
                }
    if regressions:
        return GateResult(
            "slice_gates", GateStatus.FAIL,
            f"slice regression beyond tolerance: {sorted(regressions)}",
            {"regressions": regressions},
        )
    return GateResult(
        "slice_gates", GateStatus.PASS,
        "no protected slice regresses beyond tolerance",
        {"slices_evaluated": {c: len(s) for c, s in candidate_slices.items()}},
    )


def _p99(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(0.99 * len(ordered)))
    return ordered[idx]


def latency_budget_gate(
    model: Model, X: list[list[float]], policy: ModelGatePolicy
) -> GateResult:
    """Empirical p99 single-row inference latency vs the serving budget."""
    if not X:
        return GateResult("latency_budget", GateStatus.PASS, "no rows to time", {})
    sample = X[: min(len(X), 2000)]
    latencies_ms: list[float] = []
    for x in sample:
        t0 = time.perf_counter()
        model.predict([x])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    p99 = _p99(latencies_ms)
    evidence = {
        "p99_ms": p99,
        "budget_ms": policy.max_p99_latency_ms,
        "rows_timed": len(sample),
    }
    if p99 > policy.max_p99_latency_ms:
        return GateResult(
            "latency_budget", GateStatus.FAIL,
            f"p99 latency {p99:.3f}ms exceeds budget {policy.max_p99_latency_ms:.3f}ms",
            evidence,
        )
    return GateResult(
        "latency_budget", GateStatus.PASS,
        f"p99 latency {p99:.3f}ms within budget", evidence,
    )


def model_size_gate(model: Model, policy: ModelGatePolicy) -> GateResult:
    size_bytes = len(json.dumps(model.to_dict()).encode("utf-8"))
    evidence = {
        "model_bytes": size_bytes,
        "budget_bytes": policy.max_model_bytes,
        "n_parameters": model.n_parameters(),
    }
    if size_bytes > policy.max_model_bytes:
        return GateResult(
            "model_size", GateStatus.FAIL,
            f"model artifact {size_bytes}B exceeds budget {policy.max_model_bytes}B",
            evidence,
        )
    return GateResult(
        "model_size", GateStatus.PASS,
        f"model artifact {size_bytes}B within budget", evidence,
    )


def run_model_gates(
    model: Model,
    rows: list[Row],
    X: list[list[float]],
    y: list[int],
    baseline_model: Model,
    baseline_metrics: dict[str, float],
    baseline_slices: dict[str, dict[str, dict[str, float]]],
    policy: ModelGatePolicy,
    slice_columns: tuple[str, ...],
) -> tuple[GateReport, dict[str, float], dict[str, dict[str, dict[str, float]]]]:
    """Run all model gates. Returns (report, candidate_metrics, candidate_slices)."""
    metrics = evaluate(model, X, y)
    candidate_slices = {
        col: slice_metrics(model, rows, X, y, col, policy.min_slice_samples)
        for col in slice_columns
    }
    report = GateReport(stage="model_validation")
    report.add(metric_floor_gate(metrics, policy))
    report.add(baseline_delta_gate(metrics, baseline_metrics, policy))
    report.add(slice_gate(candidate_slices, baseline_slices, policy))
    report.add(latency_budget_gate(model, X, policy))
    report.add(model_size_gate(model, policy))
    return report, metrics, candidate_slices
