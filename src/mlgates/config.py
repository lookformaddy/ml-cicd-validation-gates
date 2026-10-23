"""Pipeline configuration: typed policies for every gate family.

All knobs are dataclasses so they are explicit, reviewable in PRs, and
serializable into the run manifest (reproducibility). Environment overrides
exist for CI; code defaults are the documented policy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataGatePolicy:
    """Thresholds for the data-validation stage."""

    expected_schema: dict[str, str] = field(default_factory=dict)
    range_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    label_column: str = "is_churned"
    max_null_rate: float = 0.05
    null_warn_fraction: float = 0.5  # WARN when null rate exceeds this * max
    min_minority_fraction: float = 0.10
    max_duplicate_rate: float = 0.02
    max_range_violation_rate: float = 0.01
    missing_column_severity: str = "fail"
    new_column_severity: str = "warn"
    dtype_change_severity: str = "fail"


@dataclass(frozen=True)
class ModelGatePolicy:
    """Thresholds for the model-validation stage."""

    metric_floors: dict[str, float] = field(
        default_factory=lambda: {"accuracy": 0.65, "auc": 0.65}
    )
    min_delta_vs_baseline: float = 0.0  # candidate must beat baseline by this much
    max_slice_regression: float = 0.03  # worst tolerated slice accuracy drop vs baseline
    min_slice_samples: int = 30
    max_p99_latency_ms: float = 50.0
    max_model_bytes: int = 1_000_000


@dataclass(frozen=True)
class RegressionPolicy:
    """Statistical bar for blocking a candidate that regresses vs production."""

    alpha: float = 0.05  # significance level for the paired test
    min_delta: float = 0.0  # plus a raw accuracy-drop floor before blocking
    min_pairs: int = 10  # below this, paired disagreements are inconclusive


@dataclass(frozen=True)
class PromotionPolicy:
    require_human_approval: bool = True  # candidate -> production needs a signature
    allow_warn_through: bool = True  # WARNs never block; FAILs always do


@dataclass(frozen=True)
class PipelineConfig:
    seed: int = 42
    n_rows: int = 2000
    holdout_fraction: float = 0.25
    trainer: str = "logreg"  # "logreg" | "majority"
    trainer_epochs: int = 300
    trainer_lr: float = 0.5
    production_epochs: int = 50  # the "current production" model trains fewer epochs
    slice_columns: tuple[str, ...] = ("region", "age_group")
    data: DataGatePolicy = field(default_factory=DataGatePolicy)
    model: ModelGatePolicy = field(default_factory=ModelGatePolicy)
    regression: RegressionPolicy = field(default_factory=RegressionPolicy)
    promotion: PromotionPolicy = field(default_factory=PromotionPolicy)


def default_expected_schema() -> dict[str, str]:
    return {
        "tenure_months": "int",
        "monthly_spend": "float",
        "support_tickets": "int",
        "contract_type": "str",
        "region": "str",
        "age_group": "str",
        "is_churned": "int",
    }


def default_range_bounds() -> dict[str, tuple[float, float]]:
    return {
        "tenure_months": (1, 72),
        "monthly_spend": (0.0, 500.0),
        "support_tickets": (0, 30),
        "is_churned": (0, 1),
    }


def default_config() -> PipelineConfig:
    return PipelineConfig(
        data=DataGatePolicy(
            expected_schema=default_expected_schema(),
            range_bounds=default_range_bounds(),
        )
    )


def config_from_env(base: PipelineConfig | None = None) -> PipelineConfig:
    """Override the documented defaults from MLGATES_* environment variables."""
    cfg = base or default_config()
    get = os.environ.get
    data = cfg.data
    model = cfg.model
    if get("MLGATES_ACCURACY_FLOOR") is not None:
        floors = dict(model.metric_floors)
        floors["accuracy"] = float(get("MLGATES_ACCURACY_FLOOR"))
        model = ModelGatePolicy(
            metric_floors=floors,
            min_delta_vs_baseline=model.min_delta_vs_baseline,
            max_slice_regression=model.max_slice_regression,
            min_slice_samples=model.min_slice_samples,
            max_p99_latency_ms=float(get("MLGATES_LATENCY_BUDGET_MS", model.max_p99_latency_ms)),
            max_model_bytes=model.max_model_bytes,
        )
    if get("MLGATES_REQUIRE_APPROVAL") is not None:
        promo = PromotionPolicy(
            require_human_approval=get("MLGATES_REQUIRE_APPROVAL") != "0",
            allow_warn_through=cfg.promotion.allow_warn_through,
        )
    else:
        promo = cfg.promotion
    return PipelineConfig(
        seed=int(get("MLGATES_SEED", cfg.seed)),
        n_rows=int(get("MLGATES_N_ROWS", cfg.n_rows)),
        holdout_fraction=cfg.holdout_fraction,
        trainer=get("MLGATES_TRAINER", cfg.trainer),
        trainer_epochs=cfg.trainer_epochs,
        trainer_lr=cfg.trainer_lr,
        production_epochs=cfg.production_epochs,
        slice_columns=cfg.slice_columns,
        data=data,
        model=model,
        regression=cfg.regression,
        promotion=promo,
    )
