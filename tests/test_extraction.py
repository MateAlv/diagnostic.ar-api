from pathlib import Path

from app.config import Settings
from app.pipeline import Pipeline
from services.phenotyper.phenotyper import Phenotyper


class FakeTranslator:
    model_name = "fake-translator"

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        return "The patient has seizure and fever."


def test_sample_span_extraction(tmp_path: Path):
    obo_content = (
        "format-version: 1.2\n\n"
        "[Term]\n"
        "id: HP:0001250\n"
        "name: Seizure\n"
        "synonym: \"Convulsion\" EXACT []\n\n"
        "[Term]\n"
        "id: HP:0001945\n"
        "name: Fever\n"
        "synonym: \"Pyrexia\" EXACT []\n"
    )
    obo_path = tmp_path / "hp.obo"
    obo_path.write_text(obo_content, encoding="utf-8")
    index_path = tmp_path / "hpo_index.json"

    phenotyper = Phenotyper(
        hpo_obo_path=str(obo_path),
        hpo_index_path=str(index_path),
        hpo_download_url="http://example.invalid/hp.obo",
        spacy_model="en_core_web_sm",
        min_confidence=0.6,
        enable_span_backtranslation=False,
    )

    settings = Settings(
        cache_enabled=False,
        normalization_rules_path="data/normalization/es_ar.yml",
        hpo_obo_path=str(obo_path),
        hpo_index_path=str(index_path),
        hpo_download_url="http://example.invalid/hp.obo",
    )

    pipeline = Pipeline(settings, translator=FakeTranslator(), phenotyper=phenotyper)

    text_es = "El paciente tiene convulsiones y fiebre."
    response = pipeline.extract(text_es, "es-AR")
    found = {item["hpo_id"] for item in response["phenotypes"]}
    assert "HP:0001250" in found
    assert "HP:0001945" in found
