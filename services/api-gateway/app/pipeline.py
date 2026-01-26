from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

from services.translator.nllb import NllbTranslator
from services.phenotyper.phenotyper import Phenotyper

from .cache import RedisCache
from .config import Settings
from .normalization import Normalizer
from .utils import hash_text

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        translator: NllbTranslator | None = None,
        phenotyper: Phenotyper | None = None,
    ) -> None:
        self.settings = settings
        self.normalizer = Normalizer(settings.normalization_rules_path)
        self.cache = RedisCache(
            settings.redis_url, settings.cache_ttl_seconds, settings.cache_enabled
        )
        self.translator = translator or NllbTranslator(
            model_name=settings.nllb_model_name,
            device=settings.nllb_device,
            max_length=settings.nllb_max_length,
        )
        self.phenotyper = phenotyper or Phenotyper(
            hpo_obo_path=settings.hpo_obo_path,
            hpo_index_path=settings.hpo_index_path,
            hpo_download_url=settings.hpo_download_url,
            spacy_model=settings.spacy_model,
            min_confidence=settings.min_confidence,
            enable_span_backtranslation=settings.enable_span_backtranslation,
        )

    def _cache_key(self, normalized_text: str, locale: str) -> str:
        key_payload = f"{normalized_text}::{locale}::{self.translator.model_name}::{self.phenotyper.version}"
        return hash_text(key_payload)

    def normalize_text(self, text_es: str, locale: str = "es-AR") -> str:
        normalized = text_es
        if locale.lower().startswith("es"):
            normalized = self.normalizer.apply(text_es)
        return normalized

    def extract_with_meta(
        self, text_es: str, locale: str = "es-AR", normalized_text: str | None = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        normalized = normalized_text or self.normalize_text(text_es, locale)
        cache_key = self._cache_key(normalized, locale)
        cached = self.cache.get(cache_key)
        if cached:
            return cached, {"normalized_text": normalized, "cache_hit": True, "duration_ms": 0}

        started = time.time()
        text_en = self.translator.translate(
            normalized, src_lang="spa_Latn", tgt_lang="eng_Latn"
        )
        phenotypes = self.phenotyper.extract(
            text_en=text_en,
            text_es=text_es,
            translator=self.translator,
        )
        response = {
            "text_en": text_en,
            "model": {
                "translation": self.translator.model_name,
                "phenotyper": self.phenotyper.version,
            },
            "phenotypes": phenotypes,
        }

        self.cache.set(cache_key, response)
        duration_ms = int((time.time() - started) * 1000)
        logger.info("extract complete in %sms", duration_ms)
        return response, {
            "normalized_text": normalized,
            "cache_hit": False,
            "duration_ms": duration_ms,
        }

    def extract(self, text_es: str, locale: str = "es-AR") -> Dict[str, Any]:
        response, _ = self.extract_with_meta(text_es, locale)
        return response
