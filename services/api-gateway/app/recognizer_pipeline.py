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
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .cache import RedisCache
from .config import Settings
from .normalization import Normalizer

logger = logging.getLogger(__name__)

PIPELINE_NAME = "spanish-symptoms-recognizer"


def credentials_problem(provider: str, environ=os.environ, home: Optional[Path] = None) -> Optional[str]:
    """Why the provider cannot authenticate, or None when it has something to try.

    Only checks that a credential is present, not that it is valid — proving
    that costs a model call. It catches the failure that actually happened:
    a container with no credential at all, which only surfaced on the first
    clinical history.
    """
    if provider == "claude-cli":
        if environ.get("CLAUDE_CODE_OAUTH_TOKEN") or environ.get("ANTHROPIC_API_KEY"):
            return None
        credentials = (home or Path.home()) / ".claude" / ".credentials.json"
        if credentials.is_file():
            return None
        return (
            "claude-cli has no credential: set CLAUDE_CODE_OAUTH_TOKEN "
            "(from `claude setup-token`) or mount ~/.claude/.credentials.json"
        )
    if provider == "anthropic" and not environ.get("ANTHROPIC_API_KEY"):
        return "RECOGNIZER_PROVIDER=anthropic needs ANTHROPIC_API_KEY"
    return None


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
        self._warm_up_error: Optional[str] = None

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

    def warm_up(self) -> None:
        """Load the encoder and check credentials at startup.

        Never raises: a gateway that fails to start also takes search and
        specialties down with it. The problem is logged and reported by
        /healthz instead.
        """
        problem = credentials_problem(self.settings.recognizer_provider)
        if problem:
            self._warm_up_error = problem
            logger.error("recognizer cannot authenticate: %s", problem)
            return
        try:
            self.extractor
            logger.info("recognizer ready")
        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            self._warm_up_error = f"{type(exc).__name__}: {exc}"
            logger.exception("recognizer failed to load")

    def status(self) -> Dict[str, Any]:
        return {
            "pipeline": PIPELINE_NAME,
            "provider": self.settings.recognizer_provider,
            "model": self.settings.recognizer_model,
            "ready": self._extractor is not None and self._warm_up_error is None,
            "error": self._warm_up_error,
        }

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
