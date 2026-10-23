"""API surface tests. Auto-skip when FastAPI/httpx are not installed."""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")  # noqa: F841  (needed by TestClient)

from fastapi.testclient import TestClient  # noqa: E402

from mlgates.service import create_app  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(create_app())


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_run_lifecycle_with_approval(client):
    resp = client.post(
        "/runs", json={"seed": 21, "n_rows": 300, "trainer": "logreg", "epochs": 20}
    )
    assert resp.status_code == 200
    body = resp.json()
    run_id = body["run_id"]
    assert body["furthest_stage"] == "candidate"

    got = client.get(f"/runs/{run_id}")
    assert got.status_code == 200
    assert got.json()["run_id"] == run_id

    approved = client.post(
        f"/runs/{run_id}/approve",
        json={"approver": "vp-eng@example.com", "comment": "ship it"},
    )
    assert approved.status_code == 200
    assert approved.json()["summary"]["furthest_stage"] == "production"


def test_unknown_run_404(client):
    assert client.get("/runs/nope").status_code == 404
