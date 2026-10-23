"""Data validation gates: schema drift, ranges, nulls, class balance, duplicates.

Each gate returns a GateResult with evidence. Severity policy:
- missing column / dtype change -> FAIL (contract broken)
- new column               -> WARN (contract drift, tolerated)
- null rate / duplicates / range violations / class imbalance -> FAIL past the
  configured threshold, WARN when elevated but under it.
"""
from __future__ import annotations

import json
import math
from typing import Any

from .config import DataGatePolicy
from .data_gen import Row
from .gates import GateReport, GateResult, GateStatus


def _severity_to_status(severity: str) -> GateStatus:
    return GateStatus.FAIL if severity == "fail" else GateStatus.WARN


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _actual_schema(rows: list[Row]) -> dict[str, str]:
    schema: dict[str, str] = {}
    for row in rows:
        for col, val in row.items():
            if col not in schema and val is not None and not (
                isinstance(val, float) and math.isnan(val)
            ):
                schema[col] = _infer_type(val)
    return schema


def schema_drift_gate(rows: list[Row], policy: DataGatePolicy) -> GateResult:
    actual = _actual_schema(rows)
    expected = policy.expected_schema
    missing = sorted(set(expected) - set(actual))
    new = sorted(set(actual) - set(expected))
    dtype_changed = sorted(
        c for c in set(expected) & set(actual) if expected[c] != actual[c]
    )
    if missing:
        status = _severity_to_status(policy.missing_column_severity)
        return GateResult(
            "schema_drift", status,
            f"missing columns: {missing}",
            {"missing": missing, "new": new, "dtype_changed": dtype_changed},
        )
    if dtype_changed:
        status = _severity_to_status(policy.dtype_change_severity)
        return GateResult(
            "schema_drift", status,
            f"dtype changed: {dtype_changed}",
            {"missing": [], "new": new, "dtype_changed": dtype_changed},
        )
    if new:
        status = _severity_to_status(policy.new_column_severity)
        return GateResult(
            "schema_drift", status,
            f"new columns not in contract: {new}",
            {"missing": [], "new": new, "dtype_changed": []},
        )
    return GateResult(
        "schema_drift", GateStatus.PASS, "schema matches contract",
        {"columns": sorted(actual)},
    )


def range_gate(rows: list[Row], policy: DataGatePolicy) -> GateResult:
    violations: dict[str, dict[str, Any]] = {}
    n = len(rows)
    for col, (lo, hi) in policy.range_bounds.items():
        bad = sum(
            1
            for r in rows
            if r.get(col) is not None and not (lo <= float(r[col]) <= hi)
        )
        if bad:
            violations[col] = {"count": bad, "rate": bad / n, "bounds": [lo, hi]}
    if not violations:
        return GateResult("range_checks", GateStatus.PASS, "all values within bounds", {})
    worst = max(v["rate"] for v in violations.values())
    if worst > policy.max_range_violation_rate:
        return GateResult(
            "range_checks", GateStatus.FAIL,
            f"range violation rate {worst:.3%} exceeds {policy.max_range_violation_rate:.3%}",
            {"violations": violations},
        )
    return GateResult(
        "range_checks", GateStatus.WARN,
        f"{len(violations)} columns with out-of-range values (within tolerance)",
        {"violations": violations},
    )


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def null_rate_gate(rows: list[Row], policy: DataGatePolicy) -> GateResult:
    n = len(rows)
    rates = {
        col: sum(1 for r in rows if _is_null(r.get(col))) / n
        for col in policy.expected_schema
    }
    worst_col = max(rates, key=rates.get) if rates else None
    worst = rates[worst_col] if worst_col else 0.0
    evidence = {"null_rates": rates, "worst_column": worst_col, "worst_rate": worst}
    if worst > policy.max_null_rate:
        return GateResult(
            "null_rate", GateStatus.FAIL,
            f"null rate {worst:.3%} on '{worst_col}' exceeds {policy.max_null_rate:.3%}",
            evidence,
        )
    if worst > policy.max_null_rate * policy.null_warn_fraction:
        return GateResult(
            "null_rate", GateStatus.WARN,
            f"null rate {worst:.3%} on '{worst_col}' is elevated",
            evidence,
        )
    return GateResult("null_rate", GateStatus.PASS, "null rates within budget", evidence)


def class_balance_gate(rows: list[Row], policy: DataGatePolicy) -> GateResult:
    labels = [r[policy.label_column] for r in rows if not _is_null(r.get(policy.label_column))]
    if not labels:
        return GateResult(
            "class_balance", GateStatus.FAIL, "no labeled rows",
            {"minority_fraction": 0.0},
        )
    pos_rate = sum(1 for v in labels if int(v) == 1) / len(labels)
    minority = min(pos_rate, 1 - pos_rate)
    evidence = {
        "positive_rate": pos_rate,
        "minority_fraction": minority,
        "n_labeled": len(labels),
    }
    if minority < policy.min_minority_fraction:
        return GateResult(
            "class_balance", GateStatus.FAIL,
            f"minority fraction {minority:.3%} below floor {policy.min_minority_fraction:.3%}",
            evidence,
        )
    return GateResult(
        "class_balance", GateStatus.PASS,
        f"class balance ok (minority {minority:.1%})", evidence,
    )


def duplicate_rate_gate(rows: list[Row], policy: DataGatePolicy) -> GateResult:
    seen: set[str] = set()
    dups = 0
    for r in rows:
        key = json.dumps(r, sort_keys=True, default=str)
        if key in seen:
            dups += 1
        else:
            seen.add(key)
    rate = dups / len(rows) if rows else 0.0
    evidence = {"duplicate_rows": dups, "duplicate_rate": rate, "n_rows": len(rows)}
    if rate > policy.max_duplicate_rate:
        return GateResult(
            "duplicate_rate", GateStatus.FAIL,
            f"duplicate rate {rate:.3%} exceeds {policy.max_duplicate_rate:.3%}",
            evidence,
        )
    if rate > 0:
        return GateResult(
            "duplicate_rate", GateStatus.WARN,
            f"{dups} duplicate rows (within tolerance)", evidence,
        )
    return GateResult("duplicate_rate", GateStatus.PASS, "no duplicate rows", evidence)


def run_data_gates(rows: list[Row], policy: DataGatePolicy) -> GateReport:
    report = GateReport(stage="data_validation")
    report.add(schema_drift_gate(rows, policy))
    report.add(range_gate(rows, policy))
    report.add(null_rate_gate(rows, policy))
    report.add(class_balance_gate(rows, policy))
    report.add(duplicate_rate_gate(rows, policy))
    return report
