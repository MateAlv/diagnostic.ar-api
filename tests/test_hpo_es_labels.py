"""The Spanish HPO labels shipped with the image.

These are what `/extract-hpo` returns as `label_es` — the symptom names a
Spanish-speaking family reads in the wizard. The committed copy drifted months
behind upstream once already, because `download_hpo.py` fetched `hp.obo` and
nothing fetched this file. These tests make that drift visible instead of
silent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_hpo_es import build_map

DATA = Path("data/hpo")
HP_ES_JSON = DATA / "hp-es.json"
HP_ES_TSV = DATA / "hp-es.tsv"

# The published file carries ~20k labels. A copy far below that is truncated
# or stale, either of which costs users symptom names.
MIN_LABELS = 19_000


@pytest.fixture(scope="module")
def labels() -> dict[str, str]:
    return json.loads(HP_ES_JSON.read_text(encoding="utf-8"))


def test_the_shipped_labels_are_not_stale(labels: dict[str, str]) -> None:
    assert len(labels) >= MIN_LABELS, (
        f"only {len(labels)} Spanish labels; refresh with "
        "scripts/download_hpo_es.py then scripts/build_hpo_es.py"
    )


def test_the_json_matches_the_tsv_it_was_built_from(labels: dict[str, str]) -> None:
    """A refreshed .tsv with a stale .json would ship the old labels silently."""
    assert labels == build_map(HP_ES_TSV)


def test_every_key_is_an_hpo_id(labels: dict[str, str]) -> None:
    offenders = [key for key in labels if not key.startswith("HP:")]
    assert offenders == []


def test_no_label_is_blank(labels: dict[str, str]) -> None:
    offenders = [key for key, value in labels.items() if not value.strip()]
    assert offenders == []


def test_only_term_names_are_kept() -> None:
    """The builder filters on rdfs:label.

    The current file happens to carry only term names, but babelon also defines
    predicates for definitions and synonyms. If upstream starts publishing them
    here, this fails rather than letting a definition be shown as a symptom name.
    """
    with HP_ES_TSV.open(encoding="utf-8") as handle:
        predicates = {row.get("predicate_id") for row in csv.DictReader(handle, delimiter="\t")}
    assert predicates == {"rdfs:label"}, (
        f"unexpected predicates in the source: {predicates - {'rdfs:label'}}. "
        "Confirm build_hpo_es.py still selects the right rows."
    )


@pytest.mark.parametrize(
    "hpo_id,expected",
    [
        ("HP:0001250", "Convulsiones"),
        ("HP:0002110", "Bronquiectasias"),
        ("HP:0012236", "Cloruro en sudor elevado"),
        # Added upstream after the previously committed copy was cut.
        ("HP:0020001", "Anoftalmos clínico"),
    ],
)
def test_labels_seen_in_the_wizard(labels: dict[str, str], hpo_id: str, expected: str) -> None:
    assert labels[hpo_id] == expected
