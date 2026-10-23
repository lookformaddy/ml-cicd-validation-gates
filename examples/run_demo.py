"""Demo: run the full validation-gate pipeline on synthetic churn data.

Shows the fail-closed behavior end to end: clean run promotes to candidate
(staging needs no approval), then a second run with an impossible accuracy
floor demonstrates a blocked promotion with evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlgates.cli import _print_summary  # noqa: E402
from mlgates.config import ModelGatePolicy, PipelineConfig, default_config  # noqa: E402
from mlgates.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    base = default_config()

    print("=" * 72)
    print("RUN 1 — clean synthetic data, documented policy")
    print("=" * 72)
    clean = PipelineConfig(
        seed=42, n_rows=1500, trainer_epochs=120, data=base.data,
        model=base.model, regression=base.regression, promotion=base.promotion,
    )
    _print_summary(run_pipeline(clean))

    print()
    print("=" * 72)
    print("RUN 2 — same data, impossible accuracy floor (demonstrates fail-closed)")
    print("=" * 72)
    strict_model = ModelGatePolicy(
        metric_floors={"accuracy": 0.999, "auc": 0.999},
        min_delta_vs_baseline=base.model.min_delta_vs_baseline,
        max_slice_regression=base.model.max_slice_regression,
        min_slice_samples=base.model.min_slice_samples,
        max_p99_latency_ms=base.model.max_p99_latency_ms,
        max_model_bytes=base.model.max_model_bytes,
    )
    strict = PipelineConfig(
        seed=42, n_rows=1500, trainer_epochs=120, data=base.data,
        model=strict_model, regression=base.regression, promotion=base.promotion,
    )
    _print_summary(run_pipeline(strict))


if __name__ == "__main__":
    main()
