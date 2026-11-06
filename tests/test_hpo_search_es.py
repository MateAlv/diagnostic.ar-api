"""The symptom search box (GET /hpo/es/search), on the real HPO data.

The extractor's matchers share this index, so the last tests pin what they
return: changing the search must not move extraction.
"""

import unicodedata
from pathlib import Path

import pytest

from services.phenotyper.index import HpoIndex, _hpo_numeric_id

DATA = Path(__file__).resolve().parents[1] / "data" / "hpo"


@pytest.fixture(scope="module")
def index() -> HpoIndex:
    return HpoIndex.load_or_build(
        obo_path=str(DATA / "hp.obo"),
        index_path=str(DATA / "hpo_index.json"),
        download_url="http://example.invalid/hp.obo",
        hpo_es_path=str(DATA / "hp-es.json"),
    )


def ids(results):
    return [r["hpo_id"] for r in results]


def plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def test_every_result_shares_the_query_words(index):
    # The old fuzzy fill matched "Infecciones del tracto urinario" on "del".
    for query, stem in [
        ("retraso del desarrollo", "desarroll"),
        ("dilatación de la raíz aórtica", "aort"),
        ("luxación del cristalino", "cristalin"),
        ("pecas en axila", "axila"),
    ]:
        results = index.search_es(query, 20)
        assert results, query
        for r in results:
            assert stem in plain(r["label_es"]), (query, r)


def test_the_right_term_comes_first(index):
    assert ids(index.search_es("dilatación de la raíz aórtica"))[0] == "HP:0002616"
    assert ids(index.search_es("pecas en axila"))[0] == "HP:0000997"
    assert ids(index.search_es("talla alta"))[0] == "HP:0000098"
    assert ids(index.search_es("aracnodactilia"))[0] == "HP:0001166"
    assert ids(index.search_es("retraso global del desarrollo"))[0] == "HP:0001263"
    assert "HP:0001252" in ids(index.search_es("hipotonía"))[:3]


def test_everyday_words_reach_their_term(index):
    assert ids(index.search_es("sordera"))[0] == "HP:0000365"
    assert ids(index.search_es("epilepsia"))[0] == "HP:0001250"
    assert ids(index.search_es("hipoacusia"))[0] == "HP:0000365"


def test_a_typo_is_forgiven_but_not_the_opposite_word(index):
    assert ids(index.search_es("convulsines"))[0] == "HP:0001250"
    assert not any("macrocefal" in plain(r["label_es"]) for r in index.search_es("microcefalia", 20))
    assert not any("hiperton" in plain(r["label_es"]) for r in index.search_es("hipotonia", 20))
    assert not any("talla baja" in plain(r["label_es"]) for r in index.search_es("talla alta", 20))


def test_a_typo_gives_the_closest_spelling_before_the_opposite_term(index):
    first = lambda query: plain(index.search_es(query)[0]["label_es"])
    assert first("hiperrflexia") == "hiperreflexia"
    assert first("macrocfalia") == "macrocefalia"
    assert first("microcfalia congenita") == "microcefalia congenita"
    assert first("hiperpigentacion macular") == "hiperpigmentacion macular"


def test_a_word_cut_short_gets_a_second_chance_but_a_whole_word_does_not(index):
    assert plain(index.search_es("juicio anomal")[0]["label_es"]) == "juicio anormal"
    for query in ["microcefalia severa", "microcefalia leve", "hipotonia severa"]:
        for r in index.search_es(query, 20):
            label = plain(r["label_es"])
            assert "macrocefal" not in label and "hiperton" not in label, (query, r)


def test_con_and_sin_are_not_swapped(index):
    query = "convusion focal sensorial con perdida de conciencia"
    assert plain(index.search_es(query)[0]["label_es"]) == "convulsion focal sensorial con perdida de conciencia"
    query = "convusion focal sensorial sin perdida de conciencia"
    assert plain(index.search_es(query)[0]["label_es"]) == "convulsion focal sensorial sin perdida de conciencia"


def test_a_half_typed_word_still_finds_the_term(index):
    assert "HP:0001250" in ids(index.search_es("convul"))[:3]
    assert "HP:0012758" in ids(index.search_es("retraso del des"))[:3]


def test_obsolete_terms_are_never_offered(index):
    for query in ["obsoleto", "anomalía de la ubicación", "globo ocular", "convulsiones"]:
        for r in index.search_es(query, 20):
            assert r["hpo_id"] in index.terms_by_id, r
            assert not plain(r["label_es"]).startswith("obsolet"), r


def test_the_limit_is_respected(index):
    assert len(index.search_es("anomalía", 5)) == 5
    assert ids(index.search_es("sordera", 1)) == ["HP:0000365"]


def test_a_query_of_only_small_words_offers_nothing_unrelated(index):
    for r in index.search_es("de la", 20):
        assert plain(r["label_es"]).startswith("de la"), r


def test_ids_tie_break_numerically():
    assert _hpo_numeric_id("HP:0000010") == 10
    assert _hpo_numeric_id("not an id") == 10_000_000


# What the extractor's matchers returned before the search changed. They are
# not part of the search box and must stay exactly as they were.
EXTRACTOR_ES = {
    "convulsiones": ("HP:0001250", "Seizure", "Convulsiones", "exact_es", 0.95),
    "hipotonia muscular": ("HP:0001252", "Hypotonia", "Hipotonía muscular", "exact_es", 0.95),
    "retraso del desarrollo": (
        "HP:0012758", "Neurodevelopmental delay", "Retraso del desarrollo neurológico", "fuzzy_es", 0.86,
    ),
    "talla alta": ("HP:0000098", "Tall stature", "Talla alta", "exact_es", 0.95),
    "sordera": ("HP:0009900", "Unilateral deafness", "Sordera unilateral", "fuzzy_es", 0.86),
    "pecas en axila": ("HP:0001480", "Freckling", "Pecas", "fuzzy_es", 0.86),
}
EXTRACTOR_EN = {
    "seizure": ("HP:0001250", "Seizure", "exact", 0.95),
    "global developmental delay": ("HP:0001263", "Global developmental delay", "exact", 0.95),
    "deafness": ("HP:0000365", "Hearing impairment", "synonym", 0.85),
}


def test_the_extractor_matchers_are_unchanged_by_searching(index):
    for query in ["sordera", "retraso del desarrollo", "convulsines", "pecas en axila"]:
        index.search_es(query, 20)
    for text, expected in EXTRACTOR_ES.items():
        got = index.match_es(text)
        assert got[:4] == expected[:4], text
        assert got[4] == pytest.approx(expected[4]), text
    for text, expected in EXTRACTOR_EN.items():
        got = index.match(text)
        assert got[:3] == expected[:3], text
        assert got[3] == pytest.approx(expected[3]), text
