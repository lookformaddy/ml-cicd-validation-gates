"""Shared fixtures for the mlgates test suite."""
from __future__ import annotations

import pytest

from mlgates.config import (
    DataGatePolicy,
    ModelGatePolicy,
    PipelineConfig,
    PromotionPolicy,
    RegressionPolicy,
    default_config,
)
from mlgates.data_gen import generate_churn_dataset


@pytest.fixture()
def clean_rows():
    return generate_churn_dataset(500, seed=7)


@pytest.fixture()
def cfg() -> PipelineConfig:
    base = default_config()
    return PipelineConfig(
        seed=7,
        n_rows=500,
        holdout_fraction=0.25,
        trainer="logreg",
        trainer_epochs=40,
        trainer_lr=0.5,
        production_epochs=10,
        slice_columns=base.slice_columns,
        data=base.data,
        model=base.model,
        regression=base.regression,
        promotion=base.promotion,
    )
