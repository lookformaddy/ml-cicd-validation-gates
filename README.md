# ml-cicd-validation-gates

**The pipeline between "notebook experiment" and "production model."** Data
validation gates, model validation gates, statistical regression detection,
reproducibility manifests, and a fail-closed promotion stage machine — with
every decision carrying its evidence.

Topic: `ml-engineering`

## The enterprise problem

Most ML incidents don't come from bad modeling — they come from everything
*around* the model: a silently changed input column, label skew in the new
training data, a model that's better on average but worse for an important
customer segment, inference latency that blows the serving budget, or a run
nobody can reproduce during the post-mortem. This project is the CI/CD
validation layer that checks all of that *before* a model is allowed to
promote, with evidence attached to every gate.

## Architecture overview

```
data → DATA GATES → training → MODEL GATES → REGRESSION CHECK → PROMOTION
       (schema drift,        (metric floors,   (McNemar paired     (experimental →
        ranges, nulls,         baseline delta,   test vs current      staging →
        class balance,         slice gates,      production)          candidate ──✋──▶ production)
        duplicates)            latency/size)
                                    │                                     ▲
                                    ▼                                     │ signed human
                        REPRODUCIBILITY MANIFEST                           │ approval
              (code hash · data hash · config · seed · env)                │
```

- **Fail-closed aggregation:** any single FAIL blocks promotion. WARNs pass
  but are recorded. There is no silent override — overrides happen via a
  signed, run-bound human approval at the production step.
- **Stage machine:** `experimental → staging → candidate → production`, plus
  `production → staging` via rollback only.
- **Pure stdlib core** (zero runtime dependencies); FastAPI service is optional.

See [docs/architecture.md](docs/architecture.md) for the full design,
trade-offs, and module map.

## How to run it

```bash
pip install -e ".[dev]"        # dev deps (pytest); core needs nothing
make test                      # run the test suite
make demo                      # full pipeline: clean run + fail-closed run
mlgates run --seed 42 --rows 2000 --trainer logreg --epochs 300
mlgates data-gates --csv data/customers.csv   # data gates only, over your CSV
mlgates serve --port 8000      # FastAPI API (needs pip install ".[api]")
docker build -t ml-cicd-validation-gates .
```

Environment overrides: copy `.env.example` to `.env` and tune
`MLGATES_SEED`, `MLGATES_N_ROWS`, `MLGATES_ACCURACY_FLOOR`,
`MLGATES_LATENCY_BUDGET_MS`, `MLGATES_REQUIRE_APPROVAL`.

### Example output (clean run)

```
run_id:          run-9f3ac21be04d
data gates:      pass
  - schema_drift   pass  schema matches contract
  - range_checks   pass  all values within bounds
  - null_rate      pass  null rates within budget
  - class_balance  pass  class balance ok (minority 38.2%)
  - duplicate_rate pass  no duplicate rows
model gates:     pass
  - metric_floors  pass  all metrics above floors
  - baseline_delta pass  candidate beats baseline on all floored metrics
  - slice_gates    pass  no protected slice regresses beyond tolerance
  - latency_budget pass  p99 latency 0.012ms within budget
  - model_size     pass  model artifact 1843B within budget
candidate:       accuracy=0.8123 auc=0.8841 f1=0.7902
regression:      pass (no significant regression: delta=+0.0040, McNemar p=0.62)
promotion walk:
  experimental -> staging       APPROVED
  staging      -> candidate     APPROVED
  candidate    -> production    REJECTED
      - human approval required for production; none provided
furthest stage:  candidate
reproduce:       mlgates run --seed 42 --rows 2000 --trainer logreg --epochs 300 --expect-data-hash 4f2a…
```

## Key design decisions and trade-offs

- **McNemar's paired test over raw accuracy deltas** — candidate and
  production are scored on the *same* holdout rows, so the correct test is
  paired; noise alone never blocks a deploy.
- **Slice gates on protected segments** (region, age group) — a model that's
  better on average but worse for a segment is a regression; blocked beyond a
  configured tolerance.
- **Empirical p99 latency** — measured on the real inference path, not
  estimated from FLOPs.
- **Reproducibility as an artifact** — every run records code/data/config
  hashes; `verify_manifest` recomputes them; a one-line reproduce command is
  emitted. Manifests are built even for blocked runs (post-mortems need them).
- **Three models per run** — candidate, a weaker "current production"
  stand-in, and a naive majority baseline anchoring the metric floors.

## What it demonstrates

Designing promotion safety for ML systems: fail-closed gate aggregation,
paired statistical regression testing, slice-level quality gates,
reproducibility engineering via content hashing, and the organizational
control plane (stage machine, signed approvals, rollback wiring) — as a
tested, documented, deployable codebase.

## CI

`docs/github-actions-ci.yml` is the CI definition. It ships under `docs/`
(rather than `.github/workflows/`) because the GitHub App used to publish
this repo lacks the Workflows permission — move it to
`.github/workflows/ci.yml` to activate.

## Clean-room note

All code is original and all data is synthetic (seeded churn dataset). No
employer code, data, or proprietary designs are included.
