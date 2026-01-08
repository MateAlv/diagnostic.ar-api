from pathlib import Path

from services.phenotyper.index import HpoIndex, parse_obo


def test_hpo_index_parsing(tmp_path: Path):
    obo_path = Path("tests/data/hp-small.obo")
    terms = parse_obo(obo_path)
    assert any(term.hpo_id == "HP:0001250" for term in terms)

    index_path = tmp_path / "hpo_index.json"
    obo_copy = tmp_path / "hp.obo"
    obo_copy.write_text(obo_path.read_text(encoding="utf-8"), encoding="utf-8")

    index = HpoIndex.load_or_build(
        obo_path=str(obo_copy),
        index_path=str(index_path),
        download_url="http://example.invalid/hp.obo",
    )
    match = index.match("convulsion")
    assert match is not None
    hpo_id, label, matched_by, confidence = match
    assert hpo_id == "HP:0001250"
    assert label == "Seizure"
    assert matched_by in {"exact", "synonym", "fuzzy"}
    assert confidence > 0.5
