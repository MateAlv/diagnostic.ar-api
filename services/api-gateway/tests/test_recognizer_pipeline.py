"""Tests for the recognizer-backed /extract-hpo pipeline.

The recognizer itself is stubbed: these cover the seam -- caching, the shape
handed back to the route, and which text offsets are resolved against.
"""
from __future__ import annotations

import pytest

from app.recognizer_pipeline import PIPELINE_NAME, RecognizerPipeline


class StubExtractor:
    def __init__(self, response=None):
        self.response = response or {
            "symptoms_extracted": ["fiebre"],
            "model": {"symptom_extractor": "x", "translator": "none", "phenotyper": "y"},
            "phenotypes": [{"hpo_id": "HP:0001945", "span_es": "fiebre"}],
        }
        self.calls = []

    def extract(self, text, offset_text=None):
        self.calls.append((text, offset_text))
        return self.response


class StubSettings:
    normalization_rules_path = None
    redis_url = "redis://127.0.0.1:6379/0"
    cache_ttl_seconds = 60
    cache_enabled = False
    recognizer_provider = "claude-cli"
    recognizer_model = "claude-opus-5"
    recognizer_text_parser = "full-text"


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(
        "app.recognizer_pipeline.Normalizer", lambda path: type("N", (), {"apply": staticmethod(lambda t: t)})()
    )
    return RecognizerPipeline(StubSettings(), extractor=StubExtractor())


class TestResponseShape:
    def test_returns_the_extractor_response(self, pipeline):
        response, meta = pipeline.extract_with_meta("Presenta fiebre")
        assert response["phenotypes"][0]["hpo_id"] == "HP:0001945"
        assert meta["cache_hit"] is False

    def test_extract_drops_the_metadata(self, pipeline):
        assert "phenotypes" in pipeline.extract("Presenta fiebre")

    def test_duration_is_reported(self, pipeline):
        _, meta = pipeline.extract_with_meta("Presenta fiebre")
        assert isinstance(meta["duration_ms"], int)


class TestOffsetText:
    def test_offsets_resolve_against_the_original_text(self, pipeline):
        # Normalisation rewrites by regex and can change length, so an offset
        # taken from the normalised copy would highlight the wrong characters.
        pipeline.extract_with_meta("Presenta  fiebre", normalized_text="Presenta fiebre")
        extracted_from, offset_against = pipeline._extractor.calls[0]
        assert extracted_from == "Presenta fiebre"
        assert offset_against == "Presenta  fiebre"

    def test_extraction_uses_the_normalized_copy(self, pipeline):
        pipeline.extract_with_meta("RAW", normalized_text="normalised")
        assert pipeline._extractor.calls[0][0] == "normalised"


class TestCacheKey:
    def test_key_is_namespaced_to_this_pipeline(self, pipeline):
        # A shared Redis holds both pipelines' entries; an unnamespaced key
        # would serve the incumbent's answer for a recognizer request.
        assert pipeline._cache_key("texto", "es-AR").startswith(PIPELINE_NAME)

    def test_model_is_part_of_the_key(self, pipeline):
        before = pipeline._cache_key("texto", "es-AR")
        pipeline.settings.recognizer_model = "claude-sonnet-5"
        assert pipeline._cache_key("texto", "es-AR") != before

    def test_parser_is_part_of_the_key(self, pipeline):
        before = pipeline._cache_key("texto", "es-AR")
        pipeline.settings.recognizer_text_parser = "sentences"
        assert pipeline._cache_key("texto", "es-AR") != before

    def test_text_and_locale_are_part_of_the_key(self, pipeline):
        base = pipeline._cache_key("texto", "es-AR")
        assert pipeline._cache_key("otro", "es-AR") != base
        assert pipeline._cache_key("texto", "es-ES") != base


class TestCaching:
    def test_a_cache_hit_does_not_reach_the_model(self, pipeline):
        stored = {}
        pipeline.cache.get = lambda key: stored.get(key)
        pipeline.cache.set = lambda key, value: stored.__setitem__(key, value)

        pipeline.extract_with_meta("Presenta fiebre")
        _, meta = pipeline.extract_with_meta("Presenta fiebre")

        assert meta["cache_hit"] is True
        assert len(pipeline._extractor.calls) == 1

    def test_model_failure_propagates(self, pipeline):
        class Failing:
            def extract(self, text, offset_text=None):
                raise RuntimeError("provider unreachable")

        pipeline._extractor = Failing()
        with pytest.raises(RuntimeError):
            pipeline.extract_with_meta("Presenta fiebre")
