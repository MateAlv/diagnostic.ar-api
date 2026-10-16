"""v4-layer in the service, against model-v4.0.0's staging fixture.

What HANDOFF-v4-prod.md (diagnostic.ar-training) asks of the integration:
- /healthz names the layer and its sha256s; a modified file, another pinned
  model or another model.onnx stops startup;
- the fixture's 32 cases through /rank-genes: the network's top 10 without
  inputs, and v4-layer's with the case's sex, age and onset, the same genes in
  the same order;
- with no sex, age or onset, or with V4_LAYER_ENABLED off, the response is the
  network's own, bit for bit, whatever the excluded terms say;
- results, specialties and confidence all come from the layer's distribution;
- the sex rule: no gene impossible for the patient's sex in the top 10.
The fixture's patients are ones the network was fine-tuned on: it checks
parity with the release, not the model's quality.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICE_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import v4layer  # noqa: E402
from app import main  # noqa: E402

FIXTURE = json.loads((Path(__file__).parent / "data" / "v4_staging_fixture.json").read_text())["cases"]
NAME = "v4-layer"
MODEL_SHA256 = "62b2c5e8eb4609d1003a81c3632ea82cf2b1173f4ffc646a02ecb66d38d95cc3"
TARBALL_SHA256 = "80d1b159472b5730cfa57f3c636c864c854c820be0c2ee740f67e68c2f399df9"
BUNDLE_SHA256 = "3bdeeaedff851dc81983425f489d66bcff2b124a20df2283332d6c16bd69e21e"
# The fixture allows 2e-6, but float32 ONNX kernels differ across CPUs: GitHub's
# runners drift up to 8e-6 (diagnostic.ar-model #13). The order stays exact.
TOLERANCE = 2e-5


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def layer_on(client, monkeypatch):
    monkeypatch.setattr(main, "_layer_enabled", True)


@pytest.fixture
def layer_off(client, monkeypatch):
    monkeypatch.setattr(main, "_layer_enabled", False)


def rank(client, case: dict, inputs: dict | None = None, top_k: int = 10) -> dict:
    inputs = inputs or {}
    body = {
        "hpo_ids": case["hpo_ids"],
        "top_k": top_k,
        "sex": inputs.get("sex"),
        "age_years": inputs.get("age_years"),
        "onset_years": inputs.get("onset_years"),
        "excluded_hpo_ids": inputs.get("excluded") or [],
    }
    response = client.post("/rank-genes", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def assert_top(body: dict, expected: list) -> None:
    got = [(r["gene_symbol"], r["score"]) for r in body["results"]]
    assert [gene for gene, _ in got] == [gene.upper() for gene, _ in expected]
    for (_, score), (_, probability) in zip(got, expected):
        assert score == pytest.approx(probability, abs=TOLERANCE)


def has_inputs(case: dict) -> bool:
    inputs = case["inputs"]
    return v4layer.has_inputs(v4layer.case_inputs(inputs["sex"], inputs["age_years"], inputs["onset_years"]))


def test_health_names_the_pinned_layer(client, layer_on):
    assert client.get("/healthz").json()["layer"] == {
        "name": NAME,
        "release": "model-v4.0.0",
        "sha256": TARBALL_SHA256,
        "bundle_sha256": BUNDLE_SHA256,
        "enabled": True,
    }


def test_the_layer_is_on_unless_the_flag_says_off():
    assert v4layer.enabled({})
    assert v4layer.enabled({"V4_LAYER_ENABLED": "true"})
    for off in ("false", "0", "no", "OFF"):
        assert not v4layer.enabled({"V4_LAYER_ENABLED": off})


@pytest.mark.parametrize("case", FIXTURE, ids=lambda c: f"{c['kind']}:{c['case_id']}")
def test_the_network_matches_the_fixture(client, layer_on, case):
    body = rank(client, case)
    assert_top(body, case["network_top"])
    assert body["layer"] is None


@pytest.mark.parametrize("case", FIXTURE, ids=lambda c: f"{c['kind']}:{c['case_id']}")
def test_the_layer_matches_the_fixture(client, layer_on, case):
    body = rank(client, case, case["inputs"])
    assert_top(body, case["f3_top"])
    assert body["layer"] == (NAME if has_inputs(case) else None)
    assert body["rerank"] is None


def test_the_fixture_has_every_kind_of_case():
    kinds = [c["kind"] for c in FIXTURE]
    assert (kinds.count("sex-rule"), kinds.count("addon"), kinds.count("other"), kinds.count("no-inputs")) == (9, 10, 10, 3)
    assert all(c["f3_top"] == c["network_top"] for c in FIXTURE if c["kind"] == "no-inputs")


@pytest.mark.parametrize("case", FIXTURE[:5], ids=lambda c: c["case_id"])
def test_without_inputs_the_layer_is_the_network_bit_for_bit(client, case):
    known, _ = main.split_known_ids(main._engine, case["hpo_ids"])
    network = main.predict_scores(main._engine, known)
    layered = main.predict_layer(main._engine, main._layer, known, v4layer.case_inputs())
    assert layered == network


def test_excluded_terms_change_nothing(client, layer_on):
    case = next(c for c in FIXTURE if c["kind"] == "addon")
    excluded = ["HP:0001250", "HP:0000252"]
    plain = rank(client, case, case["inputs"])
    with_excluded = rank(client, case, {**case["inputs"], "excluded": excluded})
    assert with_excluded["results"] == plain["results"]
    assert with_excluded["confidence"] == plain["confidence"]
    assert with_excluded["specialties"] == plain["specialties"]
    assert with_excluded["excluded_hpo_count"] == len(excluded)
    assert with_excluded["rerank"] is None

    only_excluded = rank(client, case, {"excluded": excluded})
    assert only_excluded["results"] == rank(client, case)["results"]
    assert only_excluded["layer"] is None


def test_with_the_flag_off_the_inputs_change_nothing(client, layer_off):
    case = next(c for c in FIXTURE if c["kind"] == "sex-rule")
    body = rank(client, case, case["inputs"])
    assert_top(body, case["network_top"])
    assert body["layer"] is None
    assert body == rank(client, case)


def test_results_specialties_and_confidence_come_from_the_layer(client, layer_on):
    case = next(c for c in FIXTURE if c["kind"] == "addon")
    inputs = v4layer.case_inputs(case["inputs"]["sex"], case["inputs"]["age_years"], case["inputs"]["onset_years"])
    known, _ = main.split_known_ids(main._engine, case["hpo_ids"])
    distribution = main.predict_layer(main._engine, main._layer, known, inputs)
    assert distribution != main.predict_scores(main._engine, known)

    body = rank(client, case, case["inputs"])
    assert body["confidence"] == main._confidence_from_distribution(distribution)
    assert body["specialties"] == main._recommend_specialties(distribution, main._gene_to_specialties)


@pytest.mark.parametrize("case", [c for c in FIXTURE if c["kind"] == "sex-rule"], ids=lambda c: c["case_id"])
def test_the_sex_rule_keeps_impossible_genes_out_of_the_top(client, layer_on, case):
    sex = case["inputs"]["sex"]
    impossible = {g.upper() for g, blocked in zip(main._layer.genes, main._layer.blocked(sex)) if blocked}
    network = {r["gene_symbol"] for r in rank(client, case)["results"]}
    layered = {r["gene_symbol"] for r in rank(client, case, case["inputs"])["results"]}
    assert network & impossible
    assert not layered & impossible


@pytest.fixture
def package(client, tmp_path) -> Path:
    copy = tmp_path / "v4-layer"
    shutil.copytree(main._V4_LAYER_DIR, copy)
    return copy


def load(package: Path, model_sha256: str = MODEL_SHA256, onnx_path: Path | None = None):
    return v4layer.load(package, main._engine.genes, model_sha256, onnx_path or Path(main._engine.runtime.onnx_path))


def test_the_released_package_loads(package):
    assert load(package).genes == tuple(main._engine.genes)


@pytest.mark.parametrize("name", ["v4-layer-bundle.json.gz", "v4_layer.py"])
def test_a_modified_file_is_refused(package, name):
    with (package / name).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="sha256"):
        load(package)


def test_another_pinned_model_is_refused(package):
    with pytest.raises(ValueError, match="fitted on"):
        load(package, model_sha256="2201b7f2868afaf00439f76a6197bcf1d94eccc5ea171173719ca48c517a61c8")


def test_another_model_onnx_is_refused(package, tmp_path):
    other = tmp_path / "model.onnx"
    other.write_bytes(Path(main._engine.runtime.onnx_path).read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        load(package, onnx_path=other)


def test_a_modified_package_stops_startup(client, package, monkeypatch):
    with (package / "v4-layer-bundle.json.gz").open("ab") as handle:
        handle.write(b"\n")
    with monkeypatch.context() as patched:
        patched.setattr(main, "_V4_LAYER_DIR", package)
        with pytest.raises(ValueError, match="sha256"):
            main.startup()
    main.startup()
    assert main._layer is not None


def test_without_the_pin_the_network_runs_alone(client, tmp_path, monkeypatch):
    lock = tmp_path / "model.lock"
    lock.write_text("\n".join(
        line for line in main._MODEL_LOCK.read_text().splitlines() if not line.startswith("V4_LAYER_")
    ))
    case = next(c for c in FIXTURE if c["kind"] == "sex-rule")
    with monkeypatch.context() as patched:
        patched.setattr(main, "_MODEL_LOCK", lock)
        main.startup()
        assert client.get("/healthz").json()["layer"] is None
        assert_top(rank(client, case, case["inputs"]), case["network_top"])
    main.startup()
    assert main._layer is not None
