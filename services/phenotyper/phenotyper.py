from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import spacy
from spacy.tokens import Doc, Span

from .index import HpoIndex
from .negation import is_negated

logger = logging.getLogger(__name__)


@dataclass
class CandidateSpan:
    text: str
    start_char: int
    end_char: int


class Phenotyper:
    version = "baseline-v1"

    def __init__(
        self,
        hpo_obo_path: str,
        hpo_index_path: str,
        hpo_download_url: str,
        hpo_es_path: str | None = None,
        spacy_model: str,
        min_confidence: float,
        enable_span_backtranslation: bool,
    ) -> None:
        self.hpo_obo_path = hpo_obo_path
        self.hpo_index_path = hpo_index_path
        self.hpo_download_url = hpo_download_url
        self.hpo_es_path = hpo_es_path
        self.spacy_model = spacy_model
        self.min_confidence = min_confidence
        self.enable_span_backtranslation = enable_span_backtranslation
        self.hpo_index_loaded = False
        self._nlp = None
        self._hpo_index: Optional[HpoIndex] = None

    def ensure_ready(self) -> None:
        if self._hpo_index is None:
            self._hpo_index = HpoIndex.load_or_build(
                self.hpo_obo_path,
                self.hpo_index_path,
                self.hpo_download_url,
                self.hpo_es_path,
            )
            self.hpo_index_loaded = True
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.spacy_model)
            except Exception as exc:
                logger.warning("spaCy model load failed: %s", exc)
                self._nlp = spacy.blank("en")

    @property
    def nlp(self):
        if self._nlp is None:
            self.ensure_ready()
        return self._nlp

    @property
    def hpo_index(self):
        if self._hpo_index is None:
            self.ensure_ready()
        return self._hpo_index

    def _extract_candidates(self, doc: Doc) -> List[CandidateSpan]:
        seen = set()
        spans: List[CandidateSpan] = []
        try:
            for chunk in doc.noun_chunks:
                text = chunk.text.strip()
                if len(text) < 3:
                    continue
                key = (chunk.start_char, chunk.end_char)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(CandidateSpan(text, chunk.start_char, chunk.end_char))
        except Exception:
            pass

        for ent in doc.ents:
            text = ent.text.strip()
            if len(text) < 3:
                continue
            key = (ent.start_char, ent.end_char)
            if key in seen:
                continue
            seen.add(key)
            spans.append(CandidateSpan(text, ent.start_char, ent.end_char))

        if not spans:
            for token in doc:
                if token.is_punct or token.is_space:
                    continue
                if len(token.text) < 4:
                    continue
                start = token.idx
                end = start + len(token.text)
                key = (start, end)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(CandidateSpan(token.text, start, end))

        return spans

    def _align_span_es(self, text_es: str, span_en: str, translator) -> Tuple[str, int, int]:
        if not self.enable_span_backtranslation or not translator:
            return "", -1, -1
        try:
            span_es_guess = translator.translate(
                span_en, src_lang="eng_Latn", tgt_lang="spa_Latn"
            )
        except Exception:
            return "", -1, -1
        if not span_es_guess:
            return "", -1, -1
        lower_text = text_es.lower()
        lower_guess = span_es_guess.lower()
        idx = lower_text.find(lower_guess)
        if idx >= 0:
            return text_es[idx : idx + len(span_es_guess)], idx, idx + len(span_es_guess)

        # Best-effort fallback using longest common substring
        import difflib

        matcher = difflib.SequenceMatcher(None, lower_text, lower_guess)
        match = matcher.find_longest_match(0, len(lower_text), 0, len(lower_guess))
        if match.size >= max(3, int(len(lower_guess) * 0.6)):
            start = match.a
            end = match.a + match.size
            return text_es[start:end], start, end
        return span_es_guess, -1, -1

    def extract(self, text_en: str, text_es: str, translator=None) -> List[Dict]:
        self.ensure_ready()
        doc = self.nlp(text_en)
        candidates = self._extract_candidates(doc)
        results: Dict[str, Dict] = {}

        for cand in candidates:
            match = self.hpo_index.match(cand.text)
            if not match:
                continue
            hpo_id, label, matched_by, confidence = match
            if confidence < self.min_confidence:
                continue
            negated = is_negated(text_en, cand.start_char)
            span_es, start_es, end_es = self._align_span_es(text_es, cand.text, translator)
            item = {
                "hpo_id": hpo_id,
                "label": label,
                "label_es": self.hpo_index.label_es(hpo_id),
                "span_es": span_es,
                "span_en": cand.text,
                "start_es": start_es,
                "end_es": end_es,
                "start_en": cand.start_char,
                "end_en": cand.end_char,
                "negated": negated,
                "confidence": round(confidence, 3),
                "matched_by": matched_by,
            }
            existing = results.get(hpo_id)
            if not existing or item["confidence"] > existing["confidence"]:
                results[hpo_id] = item

        return list(results.values())
