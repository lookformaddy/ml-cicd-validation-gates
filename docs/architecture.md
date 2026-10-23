# Architecture — ml-cicd-validation-gates

## Problem

In most teams the path from "notebook experiment" to "production model" is a
handful of Slack messages and a prayer. Models ship because training loss
looked good, while the failure modes that actually cause incidents — broken
input contracts, silent label skew, slice-level regressions on protected
segments, latency blowups, irreproducible runs — are never checked. This
project is the pipeline that stands in that gap: a CI/CD validation layer for
ML with explicit gates, evidence, and a fail-closed promotion policy.

## Pipeline stages

```
┌─────────────┐   ┌──────────┐   ┌──────────────────┐   ┌───────────────┐   ┌────────────────┐
│ data gen /  │──▶│  DATA    │──▶│     TRAINING     │──▶│     MODEL     │──▶│   REGRESSION   │
│ ingest      │   │  GATES   │   │ candidate + prod │   │     GATES     │   │  vs production │
│ (synthetic) │   │(fail-abort│   │ + naive baseline│   │               │   │  (McNemar)     │
└─────────────┘   └──────────┘   └──────────────────┘   └───────────────┘   └────────────────┘
        │                │                  │                    │                    │
        ▼                ▼                  ▼                    ▼                    ▼
  data_hash ─────▶ run_data_gates ──▶ trainers ──▶ run_model_gates ──▶ mcnemar_block ──▶ PromotionDecider
                                                                                              │
                                                                                              ▼
                                                                  ┌─────────────────────────────────────────┐
                                                                  │ experimental → staging → candidate ──✋──▶ production │
                                                                  │        (gates)      (no regression)  (signed approval)│
                                                                  └─────────────────────────────────────────┘
```

Every stage emits `GateResult`s (pass/warn/fail + machine-readable evidence)
aggregated into a `GateReport`. Aggregation is **fail-closed**: one FAIL
anywhere blocks promotion; WARNs pass but are recorded.

## Module map (`src/mlgates/`)

| Module | Responsibility |
|---|---|
| `gates.py` | `GateResult`/`GateReport` primitives and fail-closed aggregation |
| `config.py` | Typed, frozen policy dataclasses (thresholds are reviewable config, serialized into the manifest) |
| `data_gen.py` | Seeded synthetic churn data, featurization, deterministic train/holdout split |
| `trainer.py` | `Trainer` protocol (pluggable); deterministic stdlib logistic regression; majority baseline |
| `data_validation.py` | Schema drift, range checks, null-rate, class-balance, duplicate-rate gates |
| `model_validation.py` | Metric implementations (accuracy/AUC/precision/recall/F1), metric floors, baseline-delta, slice gates, empirical p99 latency budget, model-size budget |
| `regression.py` | McNemar paired test (continuity-corrected χ², df=1, closed form via erf) + raw-drop floor |
| `manifest.py` | Run manifests: code hash, data hash, config, seed, environment; `reproduce_command()`; `verify_manifest()` |
| `promotion.py` | Stage machine, signed human approvals, `PromotionDecider`, rollback records |
| `pipeline.py` | `run_pipeline()` orchestration returning a full audit-trail `PipelineResult` |
| `service.py` | Optional FastAPI API (run / approve / inspect) |
| `cli.py` | `mlgates demo \| run \| data-gates \| reproduce \| serve` |

## Key design decisions and trade-offs

1. **Pure stdlib core.** The library has zero runtime dependencies. This is a
   deliberate choice for a CI gate: it must install in seconds, run
   identically on any runner, and never break because a transitive dependency
   yanked a release. FastAPI is an optional extra for the service surface.

2. **Fail-closed aggregation.** A single FAIL blocks promotion; there is no
   "override" flag in code. Overrides happen through the human-approval record
   at the candidate→production step, which is signed and bound to the exact run
   id — so an override is always attributable, never silent.

3. **Warn vs fail severity.** Contract *breaks* (missing column, dtype change,
   threshold breaches) fail. Contract *drift* (a new column) warns. Rationale:
   new columns are the normal evolution of features; failing on them would
   train teams to bypass the gates.

4. **McNemar instead of raw accuracy delta.** Two models evaluated on the same
   holdout produce *paired* outcomes; a naive delta test ignores the pairing
   and either blocks on noise or misses real regressions. McNemar's test is the
   correct paired test for binary correctness, needs no resampling, and runs in
   O(n). A minimum-disagreement count keeps tiny holdouts from concluding
   anything (inconclusive ≠ pass-with-confidence, and the verdict says so).

5. **Three-model training per run.** Each run trains the candidate, a "current
   production" model (fewer epochs — a realistic stand-in for a weaker
   incumbent), and a naive majority baseline. The naive baseline anchors the
   metric floors ("beat trivial"); the production model anchors the regression
   gate ("don't regress the incumbent"). Cost: ~3× training, which is the right
   trade for a promotion decision.

6. **Reproducibility as a first-class artifact.** The manifest binds code hash
   (every `.py` under `src/`), data hash (order-sensitive), config snapshot,
   seed, and environment. `verify_manifest` recomputes hashes so anyone can
   confirm a historical run; the reproduce command is a one-liner. Manifests
   are built even when gates fail — a blocked run must be reproducible for the
   post-mortem.

7. **Empirical latency gate.** p99 is *measured* by timing single-row inference
   over the holdout sample, not estimated from FLOPs. It reflects the real
   serving path (including Python overhead), which is what the budget protects.

8. **Rollback is a first-class transition.** `production → staging` is the only
   legal move out of production and can only happen via `trigger_rollback`,
   producing a `RollbackRecord` with reason and triggerer — the wiring a
   post-deploy monitor calls when it detects regression.

## What it demonstrates

For a Director-level ML platform interview: designing promotion safety for ML
systems (fail-closed gate aggregation, paired statistical regression testing,
slice-level fairness-adjacent gates), reproducibility engineering (content
hashing, manifests, verify/reproduce contracts), and the organizational
control plane (stage machine, signed human approvals, rollback wiring) —
implemented as a tested, documented, deployable codebase rather than slides.
