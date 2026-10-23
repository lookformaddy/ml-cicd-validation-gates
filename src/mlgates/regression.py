"""Regression detection: candidate vs current production on the holdout set.

Uses McNemar's paired test on per-row correctness (continuity-corrected
chi-square, df=1 — closed form via erf, pure stdlib), plus a raw accuracy-drop
floor. A candidate is blocked only when it is *statistically significantly*
worse than production — noise alone never blocks a deploy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import RegressionPolicy


def _chi2_pvalue_df1(chi2: float) -> float:
    """P(X > chi2) for chi-square with 1 degree of freedom."""
    if chi2 <= 0:
        return 1.0
    return 1.0 - math.erf(math.sqrt(chi2 / 2.0))


@dataclass
class RegressionVerdict:
    blocked: bool
    reason: str
    candidate_accuracy: float
    production_accuracy: float
    delta: float  # candidate - production
    b: int  # prod right, candidate wrong
    c: int  # candidate right, prod wrong
    chi2: float
    p_value: float

    def to_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "reason": self.reason,
            "candidate_accuracy": self.candidate_accuracy,
            "production_accuracy": self.production_accuracy,
            "delta": self.delta,
            "contingency": {"b_prod_right_cand_wrong": self.b, "c_cand_right_prod_wrong": self.c},
            "chi2": self.chi2,
            "p_value": self.p_value,
        }


def mcnemar_block(
    candidate_preds: list[int],
    production_preds: list[int],
    y_true: list[int],
    policy: RegressionPolicy,
) -> RegressionVerdict:
    n = len(y_true)
    acc_cand = sum(1 for p, t in zip(candidate_preds, y_true) if p == t) / n
    acc_prod = sum(1 for p, t in zip(production_preds, y_true) if p == t) / n
    delta = acc_cand - acc_prod
    b = sum(
        1 for c, p, t in zip(candidate_preds, production_preds, y_true)
        if p == t and c != t
    )
    c = sum(
        1 for c, p, t in zip(candidate_preds, production_preds, y_true)
        if c == t and p != t
    )
    pairs = b + c
    if pairs == 0:
        return RegressionVerdict(
            blocked=False, reason="models agree on every holdout row",
            candidate_accuracy=acc_cand, production_accuracy=acc_prod,
            delta=delta, b=b, c=c, chi2=0.0, p_value=1.0,
        )
    if pairs < policy.min_pairs:
        return RegressionVerdict(
            blocked=False,
            reason=f"only {pairs} disagreeing pairs (< {policy.min_pairs}); inconclusive",
            candidate_accuracy=acc_cand, production_accuracy=acc_prod,
            delta=delta, b=b, c=c, chi2=0.0, p_value=1.0,
        )
    chi2 = (abs(b - c) - 1.0) ** 2 / pairs  # continuity correction
    p_value = _chi2_pvalue_df1(chi2)
    significantly_worse = p_value < policy.alpha and delta < 0
    raw_drop = -delta >= policy.min_delta > 0
    blocked = significantly_worse or raw_drop
    if blocked:
        reason = (
            f"candidate regresses vs production: delta={delta:+.4f}, "
            f"McNemar chi2={chi2:.3f}, p={p_value:.4f} (alpha={policy.alpha})"
        )
    else:
        reason = (
            f"no significant regression: delta={delta:+.4f}, "
            f"McNemar p={p_value:.4f} (alpha={policy.alpha})"
        )
    return RegressionVerdict(
        blocked=blocked, reason=reason,
        candidate_accuracy=acc_cand, production_accuracy=acc_prod,
        delta=delta, b=b, c=c, chi2=chi2, p_value=p_value,
    )
