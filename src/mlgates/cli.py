"""CLI: mlgates demo | run | data-gates | reproduce | serve"""
from __future__ import annotations

import argparse
import json
import sys


def _demo(args) -> int:
    from .config import config_from_env, default_config
    from .pipeline import run_pipeline
    cfg = config_from_env(default_config())
    result = run_pipeline(cfg)
    _print_summary(result)
    return 0


def _print_summary(result) -> None:
    print(f"run_id:          {result.run_id}")
    print(f"data gates:      {result.data_report.overall.value}")
    for r in result.data_report.results:
        print(f"  - {r.gate:14s} {r.status.value:4s}  {r.message}")
    print(f"model gates:     {result.model_report.overall.value}")
    for r in result.model_report.results:
        print(f"  - {r.gate:14s} {r.status.value:4s}  {r.message}")
    if result.candidate_metrics:
        m = result.candidate_metrics
        print(
            f"candidate:       accuracy={m['accuracy']:.4f} auc={m['auc']:.4f} "
            f"f1={m['f1']:.4f}"
        )
    if result.baseline_metrics:
        b = result.baseline_metrics
        print(f"baseline:        accuracy={b['accuracy']:.4f} auc={b['auc']:.4f}")
    print(f"regression:      {'BLOCKED' if result.regression.blocked else 'pass'} "
          f"({result.regression.reason})")
    print("promotion walk:")
    for d in result.progression:
        mark = "APPROVED" if d.approved else "REJECTED"
        print(f"  {d.from_stage.value:12s} -> {d.to_stage.value:12s}  {mark}")
        for reason in d.reasons:
            print(f"      - {reason}")
    print(f"furthest stage:  {result.furthest_stage.value}")
    print(f"reproduce:       {result.manifest.reproduce_command()}")


def _run(args) -> int:
    from .config import PipelineConfig, default_config
    from .pipeline import result_to_json, run_pipeline
    base = default_config()
    cfg = PipelineConfig(
        seed=args.seed, n_rows=args.rows, trainer=args.trainer,
        trainer_epochs=args.epochs, data=base.data, model=base.model,
        regression=base.regression, promotion=base.promotion,
    )
    result = run_pipeline(cfg)
    if args.json:
        print(result_to_json(result))
    else:
        _print_summary(result)
    if args.expect_data_hash:
        actual = result.manifest.data_hash[: len(args.expect_data_hash)]
        if actual != args.expect_data_hash:
            print(
                f"DATA HASH MISMATCH: expected {args.expect_data_hash}, got {actual}",
                file=sys.stderr,
            )
            return 2
        print("data hash matches: run reproduced exactly")
    return 0 if not result.blocked else 1


def _data_gates(args) -> int:
    import csv
    from .config import default_config
    from .data_validation import run_data_gates
    with open(args.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    report = run_data_gates(rows, default_config().data)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if not report.blocked else 1


def _reproduce(args) -> int:
    print("Re-run this command to reproduce the run exactly:")
    print(f"  {args.command}")
    return 0


def _serve(args) -> int:
    import uvicorn
    from .service import create_app
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mlgates", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the full pipeline on synthetic data")
    d.set_defaults(func=_demo)

    r = sub.add_parser("run", help="run the pipeline with explicit parameters")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--rows", type=int, default=2000)
    r.add_argument("--trainer", default="logreg", choices=["logreg", "majority"])
    r.add_argument("--epochs", type=int, default=300)
    r.add_argument("--json", action="store_true")
    r.add_argument("--expect-data-hash", default="")
    r.set_defaults(func=_run)

    g = sub.add_parser("data-gates", help="run only the data gates over a CSV")
    g.add_argument("--csv", required=True)
    g.set_defaults(func=_data_gates)

    rp = sub.add_parser("reproduce", help="show the reproduce command for a run")
    rp.add_argument("--command", required=True)
    rp.set_defaults(func=_reproduce)

    s = sub.add_parser("serve", help="serve the FastAPI API (needs .[api])")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
