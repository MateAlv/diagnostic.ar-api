"""rr-v1[neg=0.5] in the service: the question agent's "no" answers re-rank the model's genes.

What the release promises, checked here against the bundle model.lock pins:
- without excluded terms the response is the model's own, bit for bit;
- with them, the ranks are the ones diagnostic.ar-training measured
  (EXP-B-003, R1-dev; data/rerank_parity_r1_dev.json, copied from
  diagnostic.ar-model tests/fixtures/r1-dev-parity-sample.json at rerank-v1.0.0);
- results, specialties and confidence come from the one re-ranked distribution;
- a missing or altered bundle stops the service.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SERVICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICE_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402
from rerank.knowledge import GeneKnowledge, Ontology  # noqa: E402
from rerank.model import ReRanker  # noqa: E402

PARITY = json.loads((Path(__file__).parent / "data" / "rerank_parity_r1_dev.json").read_text())["cases"]
NAME = "rr-v1[neg=0.5]"


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def _case(group: str) -> dict:
    """The first parity case whose excluded terms moved its gene, or did not."""
    for case in PARITY:
        if (case["rank_rr"] != case["rank_e0"]) == (group == "moved") and case["excluded"]:
            return case
    raise LookupError(group)


def rank_of(probs: np.ndarray, genes, gene: str) -> float:
    """diagnostic.ar-training's rank: ties count against the true gene."""
    idx = {g.upper(): i for i, g in enumerate(genes)}[gene.upper()]
    target = probs[idx]
    return float(1 + (probs > target).sum() + ((probs == target).sum() - 1))


def test_health_names_the_pinned_reranker(client):
    rerank = client.get("/healthz").json()["rerank"]
    assert rerank == {
        "name": NAME,
        "release": "rerank-v1.0.0",
        "sha256": "4b83f5e39e003aca2dde834948392800655102199f8af099985191b69f8e79e2",
    }


@pytest.mark.parametrize("excluded", [None, [], ["HP:9999999"]])
def test_without_usable_exclusions_the_response_is_the_models_own(client, excluded):
    case = _case("moved")
    plain = client.post("/rank-genes", json={"hpo_ids": case["present"], "top_k": 50}).json()
    body = {"hpo_ids": case["present"], "top_k": 50}
    if excluded is not None:
        body["excluded_hpo_ids"] = excluded
    got = client.post("/rank-genes", json=body).json()
    for key in ("results", "specialties", "confidence", "unknown_hpo_ids"):
        assert got[key] == plain[key], key
    assert got["rerank"] is None and got["excluded_hpo_count"] == 0
    assert got["unknown_excluded_hpo_ids"] == (excluded or [])


def test_the_ranks_match_the_training_repo(client):
    engine, reranker = main._engine, main._reranker
    mismatches, moved = [], 0
    for case in PARITY:
        known, _ = main.split_known_ids(engine, case["present"])
        result = engine.predict(hpo_ids=known, top_k=1, return_distribution=True)
        e0 = np.fromiter(result["distribution"].values(), dtype=float)
        rr = np.fromiter(reranker.rerank_distribution(result["distribution"], case["excluded"]).values(), dtype=float)
        got = (rank_of(e0, engine.genes, case["gene"]), rank_of(rr, engine.genes, case["gene"]))
        moved += got[0] != got[1]
        if got != (case["rank_e0"], case["rank_rr"]):
            mismatches.append((case["case_id"], got, (case["rank_e0"], case["rank_rr"])))
    assert not mismatches, mismatches[:5]
    assert moved >= 25


def test_a_no_moves_the_genes_that_annotate_it(client):
    case = _case("moved")
    body = {"hpo_ids": case["present"], "top_k": 5229}
    plain = client.post("/rank-genes", json=body).json()
    got = client.post("/rank-genes", json={**body, "excluded_hpo_ids": case["excluded"]}).json()
    assert got["rerank"] == NAME
    sent = set(case["excluded"]) - set(case["present"])
    assert got["excluded_hpo_count"] + len(got["unknown_excluded_hpo_ids"]) == len(sent)
    assert got["excluded_hpo_count"] > 0
    order = lambda b: [r["gene_symbol"] for r in b["results"]]  # noqa: E731
    assert order(got) != order(plain)


def test_results_specialties_and_confidence_share_the_reranked_distribution(client):
    case = _case("moved")
    known, _ = main.split_known_ids(main._engine, case["present"])
    excluded, _ = main.split_excluded_ids(main._reranker, case["excluded"], known)
    score_map = main.predict_reranked(main._engine, main._reranker, known, excluded)
    got = client.post(
        "/rank-genes", json={"hpo_ids": case["present"], "top_k": 10, "excluded_hpo_ids": case["excluded"]}
    ).json()
    top = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:10]
    assert [r["gene_symbol"] for r in got["results"]] == [g for g, _ in top]
    assert got["confidence"] == main._confidence_from_distribution(score_map)
    assert got["specialties"] == main._recommend_specialties(score_map, main._gene_to_specialties)


def test_a_term_both_present_and_excluded_counts_as_present(client):
    case = _case("moved")
    present = case["present"][0]
    got = client.post(
        "/rank-genes",
        json={"hpo_ids": case["present"], "top_k": 5, "excluded_hpo_ids": [present, present]},
    ).json()
    assert got["excluded_hpo_count"] == 0 and got["rerank"] is None


def test_a_malformed_excluded_id_is_a_422(client):
    case = _case("moved")
    response = client.post(
        "/rank-genes", json={"hpo_ids": case["present"], "top_k": 5, "excluded_hpo_ids": ["banana"]}
    )
    assert response.status_code == 422


def test_an_altered_bundle_is_refused(tmp_path):
    bundle = tmp_path / "rerank-v1.0"
    shutil.copytree(main._RERANK_DIR, bundle)
    with gzip.open(bundle / "kb.json.gz", "wt") as handle:
        handle.write("{}")
    with pytest.raises(ValueError, match="sha256"):
        main.load_reranker(bundle, main._engine.genes)


def test_a_pinned_reranker_that_is_missing_stops_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_RERANK_DIR", tmp_path / "nothing-here")
    with pytest.raises(FileNotFoundError):
        main.startup()


# The module itself, on a toy knowledge base (from diagnostic.ar-model tests/test_rerank_identity.py).
GENES = ("A", "B", "C", "D")
KB = {
    "schema": 2,
    "diseases": {
        "OMIM:1": {"phenotypes": {"HP:0000003": {"freq": 0.8, "onset": [], "sex": None}}},
        "OMIM:2": {"phenotypes": {"HP:0000004": {"freq": None, "onset": [], "sex": None}}},
    },
    "genes": {"A": {"OMIM:1": 1.0}, "B": {"OMIM:2": 1.0}},
    "vocabulary": {},
}
PARENTS = {"HP:0000001": (), "HP:0000002": ("HP:0000001",), "HP:0000003": ("HP:0000002",), "HP:0000004": ()}


def toy() -> ReRanker:
    return ReRanker(GeneKnowledge(KB, Ontology(PARENTS)), GENES, neg_weight=0.5, top_k=4)


def test_toy_no_negations_returns_the_input_unchanged():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    assert toy().rerank(p, None) is p and toy().rerank(p, []) is p


def test_toy_a_negated_annotated_term_penalises_its_gene():
    out = toy().rerank(np.array([0.4, 0.3, 0.2, 0.1]), ["HP:0000002"])
    assert out[0] < 0.4 and out.sum() == pytest.approx(1.0)
