"""Regression detection: McNemar paired test blocks real regressions only."""
from __future__ import annotations

import pytest

from mlgates.config import RegressionPolicy
from mlgates.regression import mcnemar_block


@pytest.fixture()
def policy():
    return RegressionPolicy()


def test_identical_models_not_blocked(policy):
    y = [1, 0, 1, 0] * 25
    v = mcnemar_block(y, y, y, policy)
    assert not v.blocked
    assert v.b == 0 and v.c == 0
    assert "agree" in v.reason


def test_clear_regression_blocked(policy):
    # production perfect; candidate flips 25 positives to negative
    y = [1] * 50 + [0] * 50
    prod = list(y)
    cand = [0] * 25 + [1] * 25 + [0] * 50
    v = mcnemar_block(cand, prod, y, policy)
    assert v.blocked
    assert v.b == 25 and v.c == 0
    assert v.p_value < policy.alpha
    assert v.delta == pytest.approx(-0.25)


def test_improvement_not_blocked(policy):
    y = [1] * 50 + [0] * 50
    prod = [0] * 25 + [1] * 25 + [0] * 50  # production misses 25 positives
    cand = list(y)  # candidate perfect
    v = mcnemar_block(cand, prod, y, policy)
    assert not v.blocked
    assert v.delta == pytest.approx(0.25)


def test_noise_not_blocked(policy):
    # 3 vs 2 disagreements: real difference, but not statistically significant
    y = [1, 0, 1, 0, 1] + [0] * 45 + [1] * 50
    prod = [1, 0, 1, 1, 1] + [0] * 45 + [1] * 50  # prod wrong on idx 3
    cand = [1, 1, 1, 0, 0] + [0] * 45 + [1] * 50  # cand wrong on idx 1, 4
    v = mcnemar_block(cand, prod, y, policy)
    # b=1 (prod right cand wrong: idx1? prod[1]=0==y[1]=0 right, cand[1]=1 wrong -> b;
    #    idx3: prod wrong; idx4: prod[4]=1 right, cand[4]=0 wrong -> b=2),
    # c=1 (idx3: cand right, prod wrong)
    assert v.b + v.c == 3
    assert not v.blocked  # too few pairs to conclude anything


def test_too_few_pairs_inconclusive(policy):
    y = [1, 0] * 30
    prod = list(y)
    cand = [0, 0] + y[2:]  # 2 disagreements, both candidate-wrong
    v = mcnemar_block(cand, prod, y, policy)
    assert not v.blocked
    assert "inconclusive" in v.reason


def test_verdict_serializes():
    y = [1, 0] * 20
    v = mcnemar_block(y, y, y, RegressionPolicy())
    d = v.to_dict()
    assert d["blocked"] is False
    assert d["contingency"]["b_prod_right_cand_wrong"] == 0
