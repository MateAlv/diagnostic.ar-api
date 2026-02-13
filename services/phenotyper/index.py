from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HpoTerm:
    hpo_id: str
    label: str
    synonyms: List[str]


def normalize_key(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_key_es(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return normalize_key(text)


def parse_obo(path: Path) -> List[HpoTerm]:
    terms: List[HpoTerm] = []
    current_id = None
    current_label = None
    synonyms: List[str] = []
    is_obsolete = False
    in_term = False

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line == "[Term]":
                if current_id and current_label and not is_obsolete:
                    terms.append(HpoTerm(current_id, current_label, synonyms))
                current_id = None
                current_label = None
                synonyms = []
                is_obsolete = False
                in_term = True
                continue
            if line == "[Typedef]":
                if current_id and current_label and not is_obsolete:
                    terms.append(HpoTerm(current_id, current_label, synonyms))
                current_id = None
                current_label = None
                synonyms = []
                is_obsolete = False
                in_term = False
                continue
            if not in_term:
                continue
            if line.startswith("id:"):
                current_id = line.split("id:", 1)[1].strip()
                continue
            if line.startswith("name:"):
                current_label = line.split("name:", 1)[1].strip()
                continue
            if line.startswith("synonym:"):
                match = re.search(r'"(.+?)"', line)
                if match:
                    synonyms.append(match.group(1))
                continue
            if line.startswith("is_obsolete:") and "true" in line:
                is_obsolete = True

    if current_id and current_label and not is_obsolete:
        terms.append(HpoTerm(current_id, current_label, synonyms))

    return terms


def download_hpo(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading HPO data from %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)


def build_hpo_index(obo_path: Path, index_path: Path) -> List[HpoTerm]:
    terms = parse_obo(obo_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [term.__dict__ for term in terms]
    index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("HPO index built with %s terms", len(terms))
    return terms


class HpoIndex:
    def __init__(self, terms: List[HpoTerm], label_es_by_id: Optional[Dict[str, str]] = None) -> None:
        self.terms = terms
        self.terms_by_id: Dict[str, HpoTerm] = {term.hpo_id: term for term in terms}
        self.label_map: Dict[str, str] = {}
        self.synonym_map: Dict[str, str] = {}
        for term in terms:
            label_key = normalize_key(term.label)
            if label_key:
                self.label_map[label_key] = term.hpo_id
            for synonym in term.synonyms:
                syn_key = normalize_key(synonym)
                if syn_key:
                    self.synonym_map[syn_key] = term.hpo_id
        self.keys = list({*self.label_map.keys(), *self.synonym_map.keys()})
        self.label_es_by_id: Dict[str, str] = label_es_by_id or {}
        self._label_es_entries: List[Dict[str, str]] = []
        self._label_es_norms: List[str] = []
        for hpo_id, label_es in self.label_es_by_id.items():
            normalized = normalize_key_es(label_es)
            if not normalized:
                continue
            self._label_es_entries.append(
                {"hpo_id": hpo_id, "label_es": label_es, "label_es_norm": normalized}
            )
            self._label_es_norms.append(normalized)

    @classmethod
    def load_or_build(
        cls,
        obo_path: str,
        index_path: str,
        download_url: str,
        hpo_es_path: Optional[str] = None,
    ) -> "HpoIndex":
        obo = Path(obo_path)
        index = Path(index_path)
        label_es_by_id = load_hpo_es_json(Path(hpo_es_path)) if hpo_es_path else {}
        if not obo.exists():
            download_hpo(download_url, obo)
        if index.exists():
            payload = json.loads(index.read_text(encoding="utf-8"))
            terms = [HpoTerm(**item) for item in payload]
            return cls(terms, label_es_by_id)
        terms = build_hpo_index(obo, index)
        return cls(terms, label_es_by_id)

    def match(self, text: str, fuzzy_cutoff: int = 85) -> Optional[Tuple[str, str, str, float]]:
        key = normalize_key(text)
        if not key:
            return None
        if key in self.label_map:
            hpo_id = self.label_map[key]
            label = self.terms_by_id[hpo_id].label
            return hpo_id, label, "exact", 0.95
        if key in self.synonym_map:
            hpo_id = self.synonym_map[key]
            label = self.terms_by_id[hpo_id].label
            return hpo_id, label, "synonym", 0.85
        if not self.keys:
            return None
        match = process.extractOne(
            key,
            self.keys,
            scorer=fuzz.WRatio,
            score_cutoff=fuzzy_cutoff,
        )
        if not match:
            # Log best match below cutoff for debugging
            debug_match = process.extractOne(key, self.keys, scorer=fuzz.WRatio)
            if debug_match:
                logger.info("  match_en REJECTED '%s' → best='%s' score=%s (cutoff=%s)", key, debug_match[0][:40], debug_match[1], fuzzy_cutoff)
            return None
        matched_key, score, _ = match
        hpo_id = self.label_map.get(matched_key) or self.synonym_map.get(matched_key)
        if not hpo_id:
            return None
        label = self.terms_by_id[hpo_id].label
        confidence = 0.5 + (score / 100.0) * 0.35
        return hpo_id, label, "fuzzy", confidence

    def label_es(self, hpo_id: str) -> Optional[str]:
        return self.label_es_by_id.get(hpo_id)

    def match_es(self, text: str, fuzzy_cutoff: int = 90) -> Optional[Tuple[str, str, str, str, float]]:
        """
        Match Spanish text directly to HPO terms using Spanish labels.

        Args:
            text: Spanish symptom phrase to match
            fuzzy_cutoff: Minimum fuzzy match score (0-100)

        Returns:
            Tuple of (hpo_id, label_en, label_es, match_type, confidence) or None
        """
        if not self._label_es_entries:
            return None

        key = normalize_key_es(text)
        if not key or len(key) < 3:
            return None

        # Build Spanish label map on first use
        if not hasattr(self, "_label_es_map"):
            self._label_es_map: Dict[str, str] = {}
            for entry in self._label_es_entries:
                self._label_es_map[entry["label_es_norm"]] = entry["hpo_id"]

        # Exact match
        if key in self._label_es_map:
            hpo_id = self._label_es_map[key]
            label_en = self.terms_by_id[hpo_id].label if hpo_id in self.terms_by_id else ""
            label_es = self.label_es_by_id.get(hpo_id, "")
            return hpo_id, label_en, label_es, "exact_es", 0.95

        # Fuzzy match
        if not self._label_es_norms:
            return None

        match = process.extractOne(
            key,
            self._label_es_norms,
            scorer=fuzz.WRatio,
            score_cutoff=fuzzy_cutoff,
        )
        if not match:
            # Log best match below cutoff for debugging
            debug_match = process.extractOne(key, self._label_es_norms, scorer=fuzz.WRatio)
            if debug_match:
                debug_entry = self._label_es_entries[debug_match[2]]
                logger.info("  match_es REJECTED '%s' → best='%s' (%s) score=%s (cutoff=%s)", key, debug_match[0][:40], debug_entry["hpo_id"], debug_match[1], fuzzy_cutoff)
            return None

        matched_key, score, idx = match
        entry = self._label_es_entries[idx]
        hpo_id = entry["hpo_id"]
        label_en = self.terms_by_id[hpo_id].label if hpo_id in self.terms_by_id else ""
        label_es = entry["label_es"]
        confidence = 0.5 + (score / 100.0) * 0.40  # Slightly higher confidence for Spanish direct match
        return hpo_id, label_en, label_es, "fuzzy_es", confidence

    def search_es(self, query: str, limit: int = 20) -> List[Dict[str, str]]:
        normalized = normalize_key_es(query)
        if not normalized or not self._label_es_entries:
            return []
        results: List[Dict[str, str]] = []
        seen = set()

        prefix_matches = [
            entry for entry in self._label_es_entries if entry["label_es_norm"].startswith(normalized)
        ]
        prefix_matches.sort(
            key=lambda entry: (len(entry["label_es"]), _hpo_numeric_id(entry["hpo_id"]))
        )
        for entry in prefix_matches:
            if entry["hpo_id"] in seen:
                continue
            results.append(_format_es_result(entry, self.terms_by_id))
            seen.add(entry["hpo_id"])
            if len(results) >= limit:
                return results

        fuzzy_matches = process.extract(
            normalized,
            self._label_es_norms,
            scorer=fuzz.WRatio,
            score_cutoff=78,
            limit=max(limit * 2, 20),
        )
        scored_entries: List[tuple[float, Dict[str, str]]] = []
        for _, score, idx in fuzzy_matches:
            entry = self._label_es_entries[idx]
            if entry["hpo_id"] in seen:
                continue
            scored_entries.append((score, entry))
        scored_entries.sort(
            key=lambda item: (-item[0], _hpo_numeric_id(item[1]["hpo_id"]))
        )
        for _, entry in scored_entries:
            results.append(_format_es_result(entry, self.terms_by_id))
            seen.add(entry["hpo_id"])
            if len(results) >= limit:
                break
        return results


def load_hpo_es_json(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(k): str(v) for k, v in payload.items()}
    return {}


def _hpo_numeric_id(hpo_id: str) -> int:
    match = re.search(r"HP:(\\d+)", hpo_id)
    return int(match.group(1)) if match else 10_000_000


def _format_es_result(entry: Dict[str, str], terms_by_id: Dict[str, HpoTerm]) -> Dict[str, str]:
    hpo_id = entry["hpo_id"]
    label_en = terms_by_id[hpo_id].label if hpo_id in terms_by_id else ""
    return {"hpo_id": hpo_id, "label_es": entry["label_es"], "label_en": label_en}
