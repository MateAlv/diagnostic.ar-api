"""The in-process GenPhenia model: the pinned v4 network, and the request's edges.

Rankings are checked against model-v4.0.0's staging fixture in test_v4.py;
here, that the bundle is the released one and that malformed or unknown terms
are handled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICE_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402

FIXTURE = json.loads((Path(__file__).parent / "data" / "v4_staging_fixture.json").read_text())["cases"]
MODEL_SHA256 = "62b2c5e8eb4609d1003a81c3632ea82cf2b1173f4ffc646a02ecb66d38d95cc3"
CHECKPOINT_SHA256 = "bff1a05b982bf2be5711c76ec3850d66d3c629921b25aa43c956d45146d621e2"


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def test_the_bundle_is_the_released_v4_network(client):
    health = client.get("/healthz").json()
    assert health["mode"] == "in-process"
    # The engine reports the architecture, which v4 shares with v3.0.
    assert health["model_version"] == "3.0"
    assert health["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert health["model"] == {"release": "model-v4.0.0", "sha256": MODEL_SHA256}
    assert health["specialties_available"] is True


def test_an_id_the_model_does_not_know_is_left_out_and_reported(client):
    case = next(c for c in FIXTURE if len(c["hpo_ids"]) > 2)
    response = client.post("/rank-genes", json={"hpo_ids": case["hpo_ids"] + ["HP:9999999"], "top_k": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["unknown_hpo_ids"] == ["HP:9999999"]
    assert [r["gene_symbol"] for r in body["results"]] == [g.upper() for g, _ in case["network_top"]]


def test_nothing_the_model_knows_is_a_422(client):
    response = client.post("/rank-genes", json={"hpo_ids": ["HP:9999999"], "top_k": 5})
    assert response.status_code == 422
    assert response.json()["detail"]["unknown_hpo_ids"] == ["HP:9999999"]


def test_a_malformed_id_is_a_422(client):
    response = client.post("/rank-genes", json={"hpo_ids": ["banana"], "top_k": 5})
    assert response.status_code == 422


def test_a_malformed_excluded_id_is_a_422(client):
    response = client.post(
        "/rank-genes", json={"hpo_ids": FIXTURE[0]["hpo_ids"], "excluded_hpo_ids": ["banana"], "top_k": 5}
    )
    assert response.status_code == 422
