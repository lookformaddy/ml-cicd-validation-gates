"""Reproducibility: run manifests, artifact hashing, and reproduce commands.

Every pipeline run records a manifest binding together the code hash, data
hash, config, seed, and environment. `verify_manifest` recomputes the hashes
so anyone can confirm a run is exactly reproducible; `reproduce_command`
prints the one-liner that rebuilds it.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_gen import Row
from .trainer import Model


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_mapping(mapping: dict[str, Any]) -> str:
    canonical = json.dumps(mapping, sort_keys=True, default=str)
    return sha256_text(canonical)


def data_hash(rows: list[Row]) -> str:
    """Order-sensitive hash of the dataset (row order is part of the run)."""
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(row, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def code_hash(src_root: str | Path) -> str:
    """Hash of every .py file under src_root, sorted by relative path."""
    root = Path(src_root)
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(path.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()


def model_hash(model: Model) -> str:
    return hash_mapping(model.to_dict())


@dataclass
class RunManifest:
    run_id: str
    created_at: str
    seed: int
    n_rows: int
    holdout_fraction: float
    trainer: str
    trainer_epochs: int
    config: dict[str, Any] = field(default_factory=dict)
    data_hash: str = ""
    code_hash: str = ""
    model_hash: str = ""
    python_version: str = ""
    platform: str = ""

    def reproduce_command(self) -> str:
        return (
            f"mlgates run --seed {self.seed} --rows {self.n_rows} "
            f"--trainer {self.trainer} --epochs {self.trainer_epochs} "
            f"--expect-data-hash {self.data_hash[:12]}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "seed": self.seed,
            "n_rows": self.n_rows,
            "holdout_fraction": self.holdout_fraction,
            "trainer": self.trainer,
            "trainer_epochs": self.trainer_epochs,
            "config": self.config,
            "data_hash": self.data_hash,
            "code_hash": self.code_hash,
            "model_hash": self.model_hash,
            "python_version": self.python_version,
            "platform": self.platform,
            "reproduce_command": self.reproduce_command(),
        }


def build_manifest(
    run_id: str,
    seed: int,
    n_rows: int,
    holdout_fraction: float,
    trainer: str,
    trainer_epochs: int,
    config: dict[str, Any],
    rows: list[Row],
    model: Model,
    src_root: str | Path,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        n_rows=n_rows,
        holdout_fraction=holdout_fraction,
        trainer=trainer,
        trainer_epochs=trainer_epochs,
        config=config,
        data_hash=data_hash(rows),
        code_hash=code_hash(src_root),
        model_hash=model_hash(model),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )


def verify_manifest(
    manifest: RunManifest, rows: list[Row], src_root: str | Path
) -> dict[str, bool]:
    """Recompute hashes and compare with the recorded manifest."""
    return {
        "data_hash_match": data_hash(rows) == manifest.data_hash,
        "code_hash_match": code_hash(src_root) == manifest.code_hash,
    }
