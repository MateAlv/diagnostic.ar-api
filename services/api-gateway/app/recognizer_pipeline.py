"""The Spanish symptom recognizer behind the /extract-hpo contract.

Exposes the same three methods as `Pipeline`, so `main.py` chooses one at
startup and the route does not change. The two differ in how they reach an
HPO code, not in what they return:

    Pipeline            symptoms via Ollama, then fuzzy HPO matching
    RecognizerPipeline  mentions via an LLM, then vector retrieval + re-rank

Caching is deliberately shared with the incumbent's Redis cache rather than
reimplemented. Every call here costs money, so a repeated history must not
reach the model twice.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

from .cache import RedisCache
from .config import Settings
from .normalization import Normalizer

logger = logging.getLogger(__name__)

PIPELINE_NAME = "spanish-symptoms-recognizer"


class RecognizerPipeline:
    """`/extract-hpo` backed by the Spanish symptom recognizer.

    The recognizer is imported lazily: it loads a 485MB encoder and the HPO
    embedding batches, which must not happen when the feature flag is off.
    """

    def __init__(self, settings: Settings, extractor=None) -> None:
        self.settings = settings
        self.normalizer = Normalizer(settings.normalization_rules_path)
        self.cache = RedisCache(
            settings.redis_url, settings.cache_ttl_seconds, settings.cache_enabled
        )
        self._extractor = extractor

    @property
    def extractor(self):
        """Built on first use and reused, so model loading happens once."""
        if self._extractor is None:
            from symptoms_recognizer.adapters.extract_hpo import RecognizerExtractor

            logger.info(
                "loading recognizer provider=%s model=%s",
                self.settings.recognizer_provider,
                self.settings.recognizer_model,
            )
            self._extractor = RecognizerExtractor(
                provider=self.settings.recognizer_provider,
                model=self.settings.recognizer_model,
                text_parser=self.settings.recognizer_text_parser,
            )
        return self._extractor

    def normalize_text(self, text_es: str, locale: str = "es-AR") -> str:
        normalized = text_es
        if locale.lower().startswith("es"):
            normalized = self.normalizer.apply(text_es)
        return normalized

    def _cache_key(self, normalized_text: str, locale: str) -> str:
        # The model and parser are part of the key: changing either changes
        # the answer, and a stale hit would silently serve the old config.
        return (
            f"{PIPELINE_NAME}::{normalized_text}::{locale}"
            f"::{self.settings.recognizer_model}"
            f"::{self.settings.recognizer_text_parser}"
        )

    def extract_with_meta(
        self, text_es: str, locale: str = "es-AR", normalized_text: str | None = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        normalized = normalized_text or self.normalize_text(text_es, locale)
        cache_key = self._cache_key(normalized, locale)

        cached = self.cache.get(cache_key)
        if cached:
            return cached, {"normalized_text": normalized, "cache_hit": True, "duration_ms": 0}

        started = time.time()

        # Extract from the normalised copy, but resolve offsets against what
        # the caller actually sent: normalisation rewrites by regex and can
        # change length, so an offset taken from the copy would highlight the
        # wrong characters in the user's own text.
        response = self.extractor.extract(normalized, offset_text=text_es)

        duration_ms = int((time.time() - started) * 1000)
        self.cache.set(cache_key, response)

        logger.info(
            "recognizer extract complete phenotypes=%d in %sms",
            len(response.get("phenotypes", [])),
            duration_ms,
        )

        return response, {
            "normalized_text": normalized,
            "cache_hit": False,
            "duration_ms": duration_ms,
        }

    def extract(self, text_es: str, locale: str = "es-AR") -> Dict[str, Any]:
        response, _ = self.extract_with_meta(text_es, locale)
        return response
