from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

from rapidfuzz import fuzz as _fuzz
from services.symptom_extractor import SymptomExtractor, create_symptom_extractor
from services.translator.ollama import OllamaTranslator
from services.phenotyper.phenotyper import Phenotyper

from .cache import RedisCache
from .config import Settings
from .normalization import Normalizer
from .utils import hash_text

logger = logging.getLogger(__name__)


class Pipeline:
    """
    HPO extraction pipeline with symptom-first extraction.

    New architecture:
    1. Extract symptoms from Spanish text using LLM (Qwen2.5-7B via Ollama)
    2. For each symptom:
       a. Try direct Spanish HPO matching (fast, no translation needed)
       b. If no match, translate to English and match HPO
    3. Return combined results with provenance info
    """

    def __init__(
        self,
        settings: Settings,
        symptom_extractor: SymptomExtractor | None = None,
        translator: OllamaTranslator | None = None,
        phenotyper: Phenotyper | None = None,
    ) -> None:
        self.settings = settings
        self.normalizer = Normalizer(settings.normalization_rules_path)
        self.cache = RedisCache(
            settings.redis_url, settings.cache_ttl_seconds, settings.cache_enabled
        )

        # Symptom extractor (Ollama LLM)
        self.symptom_extractor = symptom_extractor
        if settings.enable_symptom_extraction and self.symptom_extractor is None:
            self.symptom_extractor = create_symptom_extractor(
                ollama_url=settings.ollama_url,
                model=settings.symptom_extractor_model,
            )

        # Translator (Ollama for symptoms that don't match Spanish HPO)
        self.translator = translator or OllamaTranslator(
            ollama_url=settings.ollama_url,
            model=settings.translator_model,
        )

        # Phenotyper (HPO matching)
        self.phenotyper = phenotyper or Phenotyper(
            hpo_obo_path=settings.hpo_obo_path,
            hpo_index_path=settings.hpo_index_path,
            hpo_download_url=settings.hpo_download_url,
            spacy_model=settings.spacy_model,
            min_confidence=settings.min_confidence,
            enable_span_backtranslation=False,  # Not needed with symptom extraction
            hpo_es_path=settings.hpo_es_path,
        )

    def _cache_key(self, normalized_text: str, locale: str) -> str:
        model_id = self.symptom_extractor.model if self.symptom_extractor else "none"
        key_payload = f"{normalized_text}::{locale}::{model_id}::{self.phenotyper.version}"
        return hash_text(key_payload)

    def normalize_text(self, text_es: str, locale: str = "es-AR") -> str:
        normalized = text_es
        if locale.lower().startswith("es"):
            normalized = self.normalizer.apply(text_es)
        return normalized

    def _match_symptom_spanish(self, symptom_es: str) -> Dict[str, Any] | None:
        """Try to match a Spanish symptom directly to HPO."""
        if not self.settings.enable_spanish_hpo_matching:
            return None

        match = self.phenotyper.hpo_index.match_es(symptom_es)
        if not match:
            return None

        hpo_id, label_en, label_es, match_type, confidence = match

        if confidence < self.settings.min_confidence:
            return None

        return {
            "hpo_id": hpo_id,
            "label": label_en,
            "label_es": label_es,
            "span_es": symptom_es,
            "span_en": "",  # No English translation needed
            "start_es": -1,
            "end_es": -1,
            "start_en": -1,
            "end_en": -1,
            "negated": False,
            "confidence": round(confidence, 3),
            "matched_by": match_type,
        }

    def _match_symptom_translated(self, symptom_es: str) -> Dict[str, Any] | None:
        """Translate symptom to English and match to HPO."""
        # Translate short phrase (much faster than full text)
        symptom_en = self.translator.translate(
            symptom_es, src_lang="spa_Latn", tgt_lang="eng_Latn"
        )

        if not symptom_en or not symptom_en.strip():
            logger.warning("  [TR] '%s' → empty translation", symptom_es)
            return None

        symptom_en = symptom_en.strip()
        logger.info("  [TR] '%s' → '%s'", symptom_es, symptom_en)

        # Skip if the translation looks like a failure (output too similar to Spanish input)
        es_norm = symptom_es.lower().strip()
        en_norm = symptom_en.lower().strip()
        for prefix in ("the ", "a ", "an "):
            if en_norm.startswith(prefix):
                en_norm = en_norm[len(prefix):]
        for suffix in (" is", " are"):
            if en_norm.endswith(suffix):
                en_norm = en_norm[: -len(suffix)]
        en_norm = en_norm.strip()
        translation_similarity = _fuzz.ratio(es_norm, en_norm)
        if translation_similarity > 70:
            logger.warning(
                "  [TR] SKIP '%s' → '%s' (similarity %.0f%%, translation likely failed)",
                symptom_es, symptom_en, translation_similarity,
            )
            return None

        # Match to HPO
        match = self.phenotyper.hpo_index.match(symptom_en)
        if not match:
            logger.info("  [TR] '%s' → '%s' → no English HPO match", symptom_es, symptom_en.strip())
            return None

        hpo_id, label, match_type, confidence = match

        if confidence < self.settings.min_confidence:
            return None

        return {
            "hpo_id": hpo_id,
            "label": label,
            "label_es": self.phenotyper.hpo_index.label_es(hpo_id) or "",
            "span_es": symptom_es,
            "span_en": symptom_en,
            "start_es": -1,
            "end_es": -1,
            "start_en": -1,
            "end_en": -1,
            "negated": False,
            "confidence": round(confidence, 3),
            "matched_by": match_type,
        }

    def _extract_and_match(self, text_es: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract symptoms and match to HPO terms.

        Returns:
            Tuple of (extracted_symptoms, matched_phenotypes)
        """
        # Extract symptoms using LLM
        if self.symptom_extractor:
            symptoms = self.symptom_extractor.extract(text_es)
        else:
            # Fallback: treat entire text as one symptom
            symptoms = [text_es]

        logger.info("Extracted %d symptoms from text", len(symptoms))

        # Match each symptom to HPO
        results: Dict[str, Dict[str, Any]] = {}  # De-duplicate by HPO ID
        spanish_matches = 0
        translated_matches = 0

        unmatched = []

        for symptom in symptoms:
            if not symptom or len(symptom.strip()) < 3:
                continue

            # Try Spanish match first
            result = self._match_symptom_spanish(symptom)
            if result:
                hpo_id = result["hpo_id"]
                if hpo_id not in results or result["confidence"] > results[hpo_id]["confidence"]:
                    results[hpo_id] = result
                    spanish_matches += 1
                logger.info("  [ES] '%s' → %s (%s)", symptom, result["hpo_id"], result["label_es"])
                continue

            # Fall back to translation + English match
            result = self._match_symptom_translated(symptom)
            if result:
                hpo_id = result["hpo_id"]
                if hpo_id not in results or result["confidence"] > results[hpo_id]["confidence"]:
                    results[hpo_id] = result
                    translated_matches += 1
                logger.info("  [EN] '%s' → '%s' → %s (%s)", symptom, result["span_en"], result["hpo_id"], result["label"])
            else:
                unmatched.append(symptom)
                logger.warning("  [--] '%s' → no HPO match", symptom)

        logger.info(
            "Matched %d HPO terms (%d Spanish, %d translated, %d unmatched)",
            len(results),
            spanish_matches,
            translated_matches,
            len(unmatched),
        )

        return symptoms, list(results.values())

    def extract_with_meta(
        self, text_es: str, locale: str = "es-AR", normalized_text: str | None = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract HPO terms with metadata."""
        normalized = normalized_text or self.normalize_text(text_es, locale)
        cache_key = self._cache_key(normalized, locale)

        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return cached, {"normalized_text": normalized, "cache_hit": True, "duration_ms": 0}

        started = time.time()

        # Extract symptoms and match to HPO
        symptoms, phenotypes = self._extract_and_match(normalized)

        response = {
            "symptoms_extracted": symptoms,
            "model": {
                "symptom_extractor": self.symptom_extractor.model if self.symptom_extractor else "none",
                "translator": self.translator.model_name,
                "phenotyper": self.phenotyper.version,
            },
            "phenotypes": phenotypes,
        }

        self.cache.set(cache_key, response)
        duration_ms = int((time.time() - started) * 1000)
        logger.info("Extract complete in %sms", duration_ms)

        return response, {
            "normalized_text": normalized,
            "cache_hit": False,
            "duration_ms": duration_ms,
        }

    def extract(self, text_es: str, locale: str = "es-AR") -> Dict[str, Any]:
        """Extract HPO terms from Spanish medical text."""
        response, _ = self.extract_with_meta(text_es, locale)
        return response
