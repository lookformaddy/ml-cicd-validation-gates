"""Optional FastAPI service exposing the validation-gate pipeline.

Importing this module requires FastAPI. The core library works without it;
API tests auto-skip when FastAPI is not installed.
"""
from __future__ import annotations

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised by skip tests
    raise ImportError(
        "FastAPI is required for the service layer: pip install '.[api]'"
    ) from exc

from .config import PipelineConfig, default_config
from .pipeline import PipelineResult, run_pipeline
from .promotion import sign_approval

_RUNS: dict[str, PipelineResult] = {}


class RunRequest(BaseModel):
    seed: int = Field(default=42)
    n_rows: int = Field(default=500, le=20000)
    trainer: str = Field(default="logreg")
    epochs: int = Field(default=100, le=1000)


class ApproveRequest(BaseModel):
    approver: str
    comment: str = ""


class RollbackRequest(BaseModel):
    reason: str
    triggered_by: str


def create_app() -> FastAPI:
    app = FastAPI(title="ml-cicd-validation-gates", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "runs": len(_RUNS)}

    @app.post("/runs")
    def start_run(req: RunRequest) -> dict:
        base = default_config()
        cfg = PipelineConfig(
            seed=req.seed, n_rows=req.n_rows, trainer=req.trainer,
            trainer_epochs=req.epochs, data=base.data, model=base.model,
            regression=base.regression, promotion=base.promotion,
        )
        result = run_pipeline(cfg)
        _RUNS[result.run_id] = result
        return result.summary()

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        result = _RUNS.get(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="unknown run")
        return result.summary()

    @app.post("/runs/{run_id}/approve")
    def approve(run_id: str, req: ApproveRequest) -> dict:
        result = _RUNS.get(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="unknown run")
        approval = sign_approval(run_id, req.approver, req.comment)
        # Re-run the promotion walk on the same run id with the signed
        # approval attached (approval is bound to this exact run id).
        progressed = run_pipeline(result.config, approval=approval, run_id=run_id)
        _RUNS[run_id] = progressed
        return {"approval": approval.to_dict(), "summary": progressed.summary()}

    @app.get("/runs/{run_id}/manifest/verify")
    def verify(run_id: str) -> dict:
        # Verification needs the original inputs; the demo store keeps them
        # implicit via the seed, so this endpoint documents the contract and
        # reports manifest presence.
        result = _RUNS.get(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="unknown run")
        return {
            "run_id": run_id,
            "manifest": result.manifest.to_dict(),
            "note": "recompute locally via mlgates reproduce --manifest <json>",
        }

    return app


app = create_app()
