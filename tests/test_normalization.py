from pathlib import Path

from app.normalization import Normalizer


def test_normalization_rules():
    rules_path = Path("data/normalization/es_ar.yml")
    normalizer = Normalizer(str(rules_path))
    text = "Me duele mal la cabeza y se me hinchó el pie."
    normalized = normalizer.apply(text)
    assert "me duele mucho" in normalized.lower()
    assert "hinchazon" in normalized.lower()
