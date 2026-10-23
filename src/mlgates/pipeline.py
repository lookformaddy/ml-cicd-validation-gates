"""Pipeline orchestration: data validation -> training -> model validation ->
regression check -> reproducibility manifest -> promotion decisions.

One `run_pipeline(config)` call executes every stage, walks the promotion
stage machine as far as gates allow (stopping before production without a
signed human approval), and returns everything needed for the audit trail.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .config import PipelineConfig
from .data_gen import generate_churn_dataset, split_rows, to_xy
from .data_validation import run_data_gates
from .gates import GateReport
from .manifest import RunManifest, build_manifest, model_hash
from .model_validation import run_model_gates
from .promotion import Approval, PromotionDecider, PromotionDecision, Stage
from .regression import RegressionVerdict, mcnemar_block
from .trainer import Model, get_trainer

SRC_ROOT = Path(__file__).resolve().parent


@dataclass
class PipelineResult:
    run_id: str
    config: PipelineConfig
    data_report: GateReport
    model_report: GateReport
    candidate_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    regression: RegressionVerdict
    manifest: RunManifest
    progression: list[PromotionDecision] = field(default_factory=list)
    blocked: bool = False

    @property
    def furthest_stage(self) -> Stage:
        stage = Stage.EXPERIMENTAL
        for d in self.progression:
            if d.approved:
                stage = d.to_stage
            else:
                break
        return stage

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "blocked": self.blocked,
            "furthest_stage": self.furthest_stage.value,
            "data_gates": self.data_report.to_dict(),
            "model_gates": self.model_report.to_dict(),
            "candidate_metrics": self.candidate_metrics,
            "baseline_metrics": self.baseline_metrics,
            "regression": self.regression.to_dict(),
            "manifest": self.manifest.to_dict(),
            "progression": [d.to_dict() for d in self.progression],
        }


def _config_snapshot(cfg: PipelineConfig) -> dict:
    return {
        "seed": cfg.seed,
        "n_rows": cfg.n_rows,
        "holdout_fraction": cfg.holdout_fraction,
        "trainer": cfg.trainer,
        "trainer_epochs": cfg.trainer_epochs,
        "trainer_lr": cfg.trainer_lr,
        "production_epochs": cfg.production_epochs,
        "slice_columns": list(cfg.slice_columns),
        "data_policy": {
            "max_null_rate": cfg.data.max_null_rate,
            "min_minority_fraction": cfg.data.min_minority_fraction,
            "max_duplicate_rate": cfg.data.max_duplicate_rate,
            "max_range_violation_rate": cfg.data.max_range_violation_rate,
        },
        "model_policy": {
            "metric_floors": cfg.model.metric_floors,
            "min_delta_vs_baseline": cfg.model.min_delta_vs_baseline,
            "max_slice_regression": cfg.model.max_slice_regression,
            "max_p99_latency_ms": cfg.model.max_p99_latency_ms,
            "max_model_bytes": cfg.model.max_model_bytes,
        },
        "regression_policy": {
            "alpha": cfg.regression.alpha,
            "min_delta": cfg.regression.min_delta,
        },
        "promotion_policy": {
            "require_human_approval": cfg.promotion.require_human_approval,
        },
    }


def run_pipeline(
    cfg: PipelineConfig,
    approval: Approval | None = None,
    run_id: str | None = None,
) -> PipelineResult:
    run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"

    # 1. Data + split (deterministic in cfg.seed)
    rows = generate_churn_dataset(cfg.n_rows, cfg.seed)
    train_rows, holdout_rows = split_rows(rows, cfg.holdout_fraction, cfg.seed)

    # 2. Data validation gates (fail-closed: abort training on FAIL)
    data_report = run_data_gates(rows, cfg.data)

    candidate_model: Model | None = None
    candidate_metrics: dict[str, float] = {}
    baseline_metrics: dict[str, float] = {}
    model_report = GateReport(stage="model_validation")
    regression = RegressionVerdict(
        blocked=False, reason="skipped (data gates failed)",
        candidate_accuracy=0.0, production_accuracy=0.0, delta=0.0,
        b=0, c=0, chi2=0.0, p_value=1.0,
    )

    if not data_report.blocked:
        # 3. Train candidate + "current production" baseline model
        X_train, y_train = to_xy(train_rows)
        X_hold, y_hold = to_xy(holdout_rows)
        candidate_model = get_trainer(
            cfg.trainer, lr=cfg.trainer_lr, epochs=cfg.trainer_epochs
        ).fit(X_train, y_train)
        production_model = get_trainer(
            cfg.trainer, lr=cfg.trainer_lr, epochs=cfg.production_epochs
        ).fit(X_train, y_train)

        # 4. Model validation gates (candidate vs a naive baseline for floors)
        naive = get_trainer("majority").fit(X_train, y_train)
        from .model_validation import evaluate, slice_metrics

        baseline_metrics = evaluate(naive, X_hold, y_hold)
        baseline_slices = {
            col: slice_metrics(
                naive, holdout_rows, X_hold, y_hold, col, cfg.model.min_slice_samples
            )
            for col in cfg.slice_columns
        }
        model_report, candidate_metrics, _ = run_model_gates(
            candidate_model, holdout_rows, X_hold, y_hold,
            naive, baseline_metrics, baseline_slices,
            cfg.model, cfg.slice_columns,
        )

        # 5. Regression: candidate vs current production on the holdout set
        regression = mcnemar_block(
            candidate_model.predict(X_hold),
            production_model.predict(X_hold),
            y_hold,
            cfg.regression,
        )

    # 6. Reproducibility manifest (built even when gates fail — the failure
    #    itself must be reproducible for post-mortems)
    manifest_model = candidate_model or Model(
        weights=[], bias=0.0, feature_names=[], trainer_name="none"
    )
    manifest = build_manifest(
        run_id=run_id,
        seed=cfg.seed,
        n_rows=cfg.n_rows,
        holdout_fraction=cfg.holdout_fraction,
        trainer=cfg.trainer,
        trainer_epochs=cfg.trainer_epochs,
        config=_config_snapshot(cfg),
        rows=rows,
        model=manifest_model,
        src_root=SRC_ROOT,
    )
    # keep the manifest's model hash honest when training was skipped
    if candidate_model is None:
        manifest.model_hash = "no-model-trained"

    # 7. Walk the stage machine as far as gates allow
    decider = PromotionDecider(
        require_human_approval=cfg.promotion.require_human_approval
    )
    progression: list[PromotionDecision] = []
    stage = Stage.EXPERIMENTAL
    while True:
        decision = decider.decide(
            stage,
            data_blocked=data_report.blocked,
            model_blocked=model_report.blocked,
            regression_blocked=regression.blocked,
            regression_reason=regression.reason,
            approval=approval,
            run_id=run_id,
        )
        progression.append(decision)
        if not decision.approved:
            break
        stage = decision.to_stage
        if stage is Stage.PRODUCTION:
            break

    blocked = data_report.blocked or model_report.blocked
    return PipelineResult(
        run_id=run_id,
        config=cfg,
        data_report=data_report,
        model_report=model_report,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        regression=regression,
        manifest=manifest,
        progression=progression,
        blocked=blocked,
    )


def result_to_json(result: PipelineResult) -> str:
    return json.dumps(result.summary(), indent=2, default=str)
