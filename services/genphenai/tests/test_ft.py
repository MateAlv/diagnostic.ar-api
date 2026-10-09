"""ft-v1.1-confirmed in the service, against the release model.lock pins.

What HANDOFF-ft-v1.1-prod.md (diagnostic.ar-training) asks of the integration:
- /healthz names ft-v1.1-confirmed and its sha256s, and a modified file stops startup;
- with FT_V11_ENABLED off, or with no sex, age, onset or excluded term, the
  response is v3.0's own, bit for bit;
- with it on, results, specialties and confidence come from ft-v1.1's
  distribution, and rr-v1 is not applied on top;
- a female patient with an X-linked-recessive-only gene, a patient with a gene
  limited to the other sex, and a patient whose onset is earlier than a gene's
  earliest onset each see that gene go down when it is in v3.0's top 10.
The cases come from data/rerank_parity_r1_dev.json, real r1-v1 dev patients.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SERVICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICE_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import ft  # noqa: E402
from app import main  # noqa: E402

PARITY = json.loads((Path(__file__).parent / "data" / "rerank_parity_r1_dev.json").read_text())["cases"]
NAME = "ft-v1.1-confirmed"
BUNDLE_SHA256 = "6bb6bb40d85a815c3bf8fb467cbfa9c32d34164ced280eb96d4f4228ddeee6c4"
TARBALL_SHA256 = "1ff0130b793847dfb2a188c533b94c7047e3fc5875d48b0eed21c96319f27bc6"


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def ft_on(client, monkeypatch):
    monkeypatch.setattr(main, "_ft_enabled", True)


@pytest.fixture
def ft_off(client, monkeypatch):
    monkeypatch.setattr(main, "_ft_enabled", False)


def rank(client, body: dict) -> dict:
    response = client.post("/rank-genes", json={"top_k": 50, **body})
    assert response.status_code == 200, response.text
    return response.json()


def e0_top(client, present: list[str], k: int = 10) -> list[str]:
    return [r["gene_symbol"] for r in rank(client, {"hpo_ids": present})["results"][:k]]


def place(body: dict, gene: str) -> int:
    genes = [r["gene_symbol"] for r in body["results"]]
    return genes.index(gene) + 1 if gene in genes else len(genes) + 1


def test_health_names_the_pinned_package(client, ft_off):
    health = client.get("/healthz").json()["ft"]
    assert health == {
        "name": NAME,
        "release": "ft-v1.1.0",
        "sha256": TARBALL_SHA256,
        "bundle_sha256": BUNDLE_SHA256,
        "enabled": False,
    }


def test_the_flag_is_read_at_startup(monkeypatch):
    monkeypatch.setenv("FT_V11_ENABLED", "true")
    assert ft.enabled()
    monkeypatch.setenv("FT_V11_ENABLED", "false")
    assert not ft.enabled()
    monkeypatch.delenv("FT_V11_ENABLED")
    assert not ft.enabled()


@pytest.mark.parametrize("name", ["ft_prod.py", "ft-v1.1-confirmed-bundle.json.gz", "ft.json"])
def test_a_modified_package_stops_the_service(tmp_path, client, name):
    package = tmp_path / NAME
    shutil.copytree(main._FT_DIR, package)
    target = package / name
    if name == "ft.json":
        config = json.loads(target.read_text())
        config["files"]["ft-v1.1-confirmed-bundle.json.gz"]["sha256"] = "0" * 64
        target.write_text(json.dumps(config))
    else:
        with target.open("ab") as handle:
            handle.write(b"\n")
    with pytest.raises(ValueError, match="sha256"):
        main.load_ft(package, main._engine.genes)


INPUTS = {"sex": "female", "age_years": 4.0, "onset_years": 1.0}


@pytest.mark.parametrize("case", PARITY[:10], ids=lambda c: c["case_id"])
def test_flag_off_new_fields_change_nothing(client, ft_off, case):
    plain = rank(client, {"hpo_ids": case["present"]})
    with_fields = rank(client, {"hpo_ids": case["present"], **INPUTS})
    assert with_fields == plain
    assert plain["ft"] is None


@pytest.mark.parametrize("case", PARITY[:10], ids=lambda c: c["case_id"])
def test_flag_on_without_new_inputs_is_v3_bit_for_bit(client, monkeypatch, case):
    monkeypatch.setattr(main, "_ft_enabled", False)
    off = rank(client, {"hpo_ids": case["present"]})
    monkeypatch.setattr(main, "_ft_enabled", True)
    for unknown in ({}, {"sex": "no_binario"}, {"sex": None, "age_years": None}, {"excluded_hpo_ids": []}):
        on = rank(client, {"hpo_ids": case["present"], **unknown})
        assert on == off
        assert on["ft"] is None and on["rerank"] is None


def test_flag_on_ranks_from_ft_distribution(client, ft_on):
    case = PARITY[0]
    body = rank(client, {"hpo_ids": case["present"], "top_k": 5229, **INPUTS})
    assert body["ft"] == NAME and body["rerank"] is None

    result = main._engine.predict(hpo_ids=case["present"], top_k=1, return_distribution=True)
    genes = list(result["distribution"])
    e0 = np.fromiter(result["distribution"].values(), dtype=np.float64, count=len(genes))
    expected = main._ft_model.apply_probs(e0, ft.case_inputs(**INPUTS))
    order = np.argsort(-expected, kind="stable")
    got = {r["gene_symbol"]: r["score"] for r in body["results"]}
    assert [r["gene_symbol"] for r in body["results"][:20]] == [genes[i].upper() for i in order[:20]]
    for i in order[:50]:
        assert got[genes[i].upper()] == round(float(expected[i]), 6)
    score_map = {g.upper(): float(p) for g, p in zip(genes, expected)}
    assert body["confidence"] == main._confidence_from_distribution(score_map)
    assert body["specialties"] == main._recommend_specialties(score_map, main._gene_to_specialties)


def test_excluded_terms_go_to_ft_and_not_to_rr_v1(client, ft_on):
    case = next(c for c in PARITY if c["excluded"] and c["rank_rr"] != c["rank_e0"])
    body = rank(client, {"hpo_ids": case["present"], "excluded_hpo_ids": case["excluded"]})
    assert body["ft"] == NAME and body["rerank"] == NAME
    known = main.split_known_ids(main._engine, case["present"])[0]
    excluded = main.normalize_excluded_ids(case["excluded"], known)
    expected = main.predict_ft(main._engine, main._ft_model, known, ft.case_inputs(excluded=excluded))
    top = sorted(expected.items(), key=lambda item: item[1], reverse=True)[:50]
    assert [(r["gene_symbol"], r["score"]) for r in body["results"]] == [(g, round(p, 6)) for g, p in top]


def test_a_term_given_as_present_is_not_excluded(client, ft_on):
    case = PARITY[0]
    body = rank(client, {"hpo_ids": case["present"], "excluded_hpo_ids": case["present"][:2]})
    assert body["ft"] is None and body["rerank"] is None
    assert body == rank(client, {"hpo_ids": case["present"]})


def _first_case(client, condition):
    """The first parity case whose v3.0 top 10 has a gene meeting `condition`, and that gene."""
    model = main._ft_model
    index = {g.upper(): i for i, g in enumerate(model.genes)}
    for case in PARITY:
        for gene in e0_top(client, case["present"]):
            found = condition(index[gene])
            if found is not None:
                return case, gene, found
    pytest.fail("no parity case has such a gene in v3.0's top 10")


def test_checklist_female_with_an_x_linked_recessive_only_gene(client, ft_on):
    case, gene, _ = _first_case(client, lambda i: True if main._ft_model.xlr[i] else None)
    before = rank(client, {"hpo_ids": case["present"]})
    after = rank(client, {"hpo_ids": case["present"], "sex": "female"})
    assert place(after, gene) > place(before, gene)
    assert rank(client, {"hpo_ids": case["present"], "sex": "male"})["results"][0]["gene_symbol"] is not None


def test_checklist_a_gene_limited_to_the_other_sex(client, ft_on):
    # No parity case has one in its top 10, so the case is made: primary
    # amenorrhea and premature ovarian insufficiency put NOBOX, limited to
    # women, first.
    present = ["HP:0000137", "HP:0008209", "HP:0000786"]
    assert main._ft_model.female[main._ft_model.genes.index("NOBOX")]
    before = rank(client, {"hpo_ids": present})
    assert place(before, "NOBOX") <= 10
    assert place(rank(client, {"hpo_ids": present, "sex": "male"}), "NOBOX") > place(before, "NOBOX")
    assert place(rank(client, {"hpo_ids": present, "sex": "female"}), "NOBOX") <= place(before, "NOBOX")


def test_checklist_onset_earlier_than_the_gene_earliest_onset(client, ft_on):
    model = main._ft_model
    case, gene, onset = _first_case(
        client, lambda i: float(model.strict[i]) if model.strict[i] > 1.0 else None
    )
    before = rank(client, {"hpo_ids": case["present"]})
    after = rank(client, {"hpo_ids": case["present"], "onset_years": onset / 2})
    assert place(after, gene) > place(before, gene)
    # Onset wins over the age: an old patient with an early onset still lowers it.
    with_age = rank(client, {"hpo_ids": case["present"], "age_years": 80.0, "onset_years": onset / 2})
    assert with_age == after


def test_no_gene_rises_above_renormalisation(client, ft_on):
    model = main._ft_model
    for case in PARITY[:10]:
        known = main.split_known_ids(main._engine, case["present"])[0]
        inputs = ft.case_inputs(sex="male", age_years=0.5, excluded=case["excluded"])
        distribution = main._engine.predict(hpo_ids=known, top_k=1, return_distribution=True)["distribution"]
        e0 = np.fromiter(distribution.values(), dtype=np.float64, count=len(distribution))
        out = main.predict_ft(main._engine, model, known, inputs)
        out = np.array([out[g.upper()] for g in model.genes])
        r = model.residual(inputs)
        ceiling = e0 / float(np.sum(e0 * np.exp(r)))
        assert np.all(r <= 0)
        assert np.all(out <= ceiling * (1 + 1e-12))
        assert np.all(out[r < 0] < ceiling[r < 0])
