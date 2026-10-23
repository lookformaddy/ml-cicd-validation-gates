"""mlgates — ML CI/CD validation gates.

The pipeline that stands between "notebook experiment" and "production model":
data validation gates, model validation gates, regression detection, run
reproducibility manifests, and fail-closed promotion decisions.

Pure standard library (the FastAPI service is an optional extra).
"""

from .gates import GateReport, GateResult, GateStatus

__all__ = ["GateReport", "GateResult", "GateStatus"]
__version__ = "0.1.0"
