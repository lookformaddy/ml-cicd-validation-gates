"""Gate primitives: pass/warn/fail results with evidence, and fail-closed aggregation.

Design: every check in the pipeline emits a GateResult with machine-readable
evidence. GateReport aggregates them with a single invariant — FAIL-CLOSED:
any FAIL anywhere blocks promotion. WARNs are allowed through but recorded,
so humans see exactly what was accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GateStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class GateResult:
    """Outcome of a single validation gate."""

    gate: str
    status: GateStatus
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def is_fail(self) -> bool:
        return self.status is GateStatus.FAIL

    @property
    def is_warn(self) -> bool:
        return self.status is GateStatus.WARN

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status.value,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class GateReport:
    """Aggregated results for one pipeline stage (e.g. data validation)."""

    stage: str
    results: list[GateResult] = field(default_factory=list)

    def add(self, result: GateResult) -> None:
        self.results.append(result)

    def failures(self) -> list[GateResult]:
        return [r for r in self.results if r.is_fail]

    def warnings(self) -> list[GateResult]:
        return [r for r in self.results if r.is_warn]

    @property
    def blocked(self) -> bool:
        """Fail-closed: any single FAIL blocks the whole stage."""
        return any(r.is_fail for r in self.results)

    @property
    def overall(self) -> GateStatus:
        if self.blocked:
            return GateStatus.FAIL
        if any(r.is_warn for r in self.results):
            return GateStatus.WARN
        return GateStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "overall": self.overall.value,
            "blocked": self.blocked,
            "results": [r.to_dict() for r in self.results],
        }
