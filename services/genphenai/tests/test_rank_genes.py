"""The in-process GenPhenia model against what production served.

``data/reference_rankings.json`` was captured on 2026-09-25 from the brainy
deployment (a separate inference server fed by /lucai and /home/gcolangelo
paths), through staging's /rank-genes. Matching it to rounding shows the
bundle in this repo is the same model, with the same post-processing.
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

REFERENCE = json.loads((Path(__file__).parent / "data" / "reference_rankings.json").read_text())
CHECKPOINT_SHA256 = "ee10aa429ba1a134ceec0e38002f3d9d0c17babe87fb755e5329778e28973955"


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def test_the_bundle_is_the_production_checkpoint(client):
    health = client.get("/healthz").json()
    assert health["mode"] == "in-process"
    assert health["model_version"] == "3.0"
    assert health["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert health["specialties_available"] is True


@pytest.mark.parametrize("case", sorted(REFERENCE))
def test_rankings_match_production(client, case):
    expected = REFERENCE[case]
    response = client.post("/rank-genes", json={"hpo_ids": expected["hpo_ids"], "top_k": 10})
    assert response.status_code == 200
    body = response.json()

    got = [(r["gene_symbol"], r["score"]) for r in body["results"]]
    assert [gene for gene, _ in got] == [gene for gene, _ in expected["top_genes"]]
    for (_, score), (_, expected_score) in zip(got, expected["top_genes"]):
        assert score == pytest.approx(expected_score, abs=2e-6)

    assert body["confidence"] == pytest.approx(expected["confidence"], abs=1e-9)
    assert body["specialties"]["specialty_histogram"] == expected["specialty_histogram"]
    assert body["unknown_hpo_ids"] == []


def test_an_id_the_model_does_not_know_is_left_out_and_reported(client):
    known = REFERENCE["neurofibromatosis_like"]["hpo_ids"]
    response = client.post("/rank-genes", json={"hpo_ids": known + ["HP:9999999"], "top_k": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["unknown_hpo_ids"] == ["HP:9999999"]
    assert [r["gene_symbol"] for r in body["results"]][:2] == ["NF1", "SPRED1"]


def test_nothing_the_model_knows_is_a_422(client):
    response = client.post("/rank-genes", json={"hpo_ids": ["HP:9999999"], "top_k": 5})
    assert response.status_code == 422
    assert response.json()["detail"]["unknown_hpo_ids"] == ["HP:9999999"]


def test_a_malformed_id_is_a_422(client):
    response = client.post("/rank-genes", json={"hpo_ids": ["banana"], "top_k": 5})
    assert response.status_code == 422
