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

    def match(self, text: str, fuzzy_cutoff: int = 88) -> Optional[Tuple[str, str, str, float]]:
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
        """Autocomplete for the symptom search box. Not used by extraction.

        Labels that start with the query come first, as before. The rest must
        contain every content word of the query, as a word or the start of
        one: "retraso del des" finds "Retraso del desarrollo neurológico" and
        nothing about the urinary tract, which the old whole-string fuzzy score
        let in on "del" alone. A word that starts no label at all is allowed a
        typo ("convulsines"). Obsolete terms are left out.
        """
        normalized = normalize_key_es(query)
        if not normalized or not self._label_es_entries:
            return []
        search = self._search_es_index()
        results: List[Dict[str, str]] = []
        seen = set()

        for entry in search.synonym_equal_to(_search_words(query)):
            results.append(_format_es_result(entry, self.terms_by_id))
            seen.add(entry["hpo_id"])
        if len(results) >= limit:
            return results[:limit]

        prefix_matches = [
            entry
            for entry in self._label_es_entries
            if entry["label_es_norm"].startswith(normalized) and search.offers(entry)
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

        for entry in search.by_words(query):
            if entry["hpo_id"] in seen:
                continue
            results.append(_format_es_result(entry, self.terms_by_id))
            seen.add(entry["hpo_id"])
            if len(results) >= limit:
                break
        return results

    def _search_es_index(self) -> "_SpanishSearchIndex":
        cached = getattr(self, "_search_es_cache", None)
        if cached is None:
            cached = _SpanishSearchIndex(self._label_es_entries, set(self.terms_by_id))
            self._search_es_cache = cached
        return cached


# Words that carry no meaning on their own in an HPO label.
_SEARCH_STOPWORDS = frozenset(
    "a al con de del el en la las lo los o para por sin su sus u un una y".split()
)

# Everyday Spanish that has no label of its own. Each one is an EXACT or
# RELATED English synonym of the same term in hp.obo, which hp-es.json does
# not translate. Shown under the term's official Spanish label.
_SEARCH_SYNONYMS_ES: Dict[str, Tuple[str, ...]] = {
    "HP:0000365": ("sordera",),  # Deafness
    "HP:0001250": ("epilepsia",),  # Epilepsy
    "HP:0001263": ("retraso global del desarrollo", "retraso del desarrollo"),  # Developmental delay
}


def _all_search_words(text: str) -> List[str]:
    """Split on anything that is not a letter or a digit, so
    "auditiva/hipoacusia" is two words and not "auditivahipoacusia"."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", text)


def _search_words(text: str) -> List[str]:
    return [word for word in _all_search_words(text) if word not in _SEARCH_STOPWORDS]


def _with_or_without(text: str) -> frozenset:
    """"con" and "sin" are too common to search by, but they reverse a label."""
    return frozenset(_all_search_words(text)) & {"con", "sin"}


class _SpanishSearchIndex:
    """Word index over the Spanish labels, built on the first search."""

    TYPO_MIN_LENGTH = 5
    TYPO_CUTOFF = 85

    def __init__(self, entries: List[Dict[str, str]], current_ids: set) -> None:
        self._current_ids = current_ids
        self._docs: List[Tuple[Dict[str, str], frozenset, int]] = []
        self._synonyms: Dict[frozenset, List[Dict[str, str]]] = {}
        by_id = {entry["hpo_id"]: entry for entry in entries}
        for entry in entries:
            words = frozenset(_search_words(entry["label_es"]))
            if words and self.offers(entry):
                self._docs.append((entry, words, len(words)))
        for hpo_id, phrases in _SEARCH_SYNONYMS_ES.items():
            entry = by_id.get(hpo_id)
            if entry is None or not self.offers(entry):
                continue
            for phrase in phrases:
                words = frozenset(_search_words(phrase))
                self._docs.append((entry, words, len(words)))
                self._synonyms.setdefault(words, []).append(entry)
        self._docs_by_word: Dict[str, List[int]] = {}
        for position, (_, words, _) in enumerate(self._docs):
            for word in words:
                self._docs_by_word.setdefault(word, []).append(position)
        self._vocabulary = sorted(self._docs_by_word)

    def offers(self, entry: Dict[str, str]) -> bool:
        # Every id in hp-es.json that hp.obo does not list as a term is obsolete
        # or an alt_id there (with no ontology loaded, keep them all). A few
        # current terms still carry a translation that starts "obsoleto".
        if self._current_ids and entry["hpo_id"] not in self._current_ids:
            return False
        return not entry["label_es_norm"].startswith("obsolet")

    def synonym_equal_to(self, words: List[str]) -> List[Dict[str, str]]:
        """The terms one of whose synonyms is exactly the query."""
        return self._synonyms.get(frozenset(words), []) if words else []

    def _matching_words(self, word: str, relaxed: bool) -> Dict[str, float]:
        """The label words `word` stands for, each with how close it is (0-100)."""
        if len(word) < 3:
            return {word: 100.0} if word in self._docs_by_word else {}
        found = {candidate: 100.0 for candidate in self._vocabulary if candidate.startswith(word)}
        # A correct word never turns into a neighbour: "microcefalia" must not
        # reach "macrocefalia", nor "hipotonia" "hipertonia". Only a word that
        # starts no label word gets a typo allowance, and on a retry a word that
        # is only the start of some ("anomal" for "anormal"), never a whole one.
        allow_typo = not found or (relaxed and word not in self._docs_by_word)
        if allow_typo and len(word) >= self.TYPO_MIN_LENGTH:
            close = process.extract(
                word, self._vocabulary, scorer=fuzz.ratio, score_cutoff=self.TYPO_CUTOFF, limit=None
            )
            for candidate, score, _ in close:
                found.setdefault(candidate, score)
        return found

    def _ranked(self, words: List[str], query_says: frozenset, relaxed: bool) -> List[Tuple]:
        closeness: Dict[int, List[float]] = {}
        for word in words:
            best: Dict[int, float] = {}
            for candidate, score in self._matching_words(word, relaxed).items():
                for position in self._docs_by_word[candidate]:
                    best[position] = max(best.get(position, 0.0), score)
            for position, score in best.items():
                closeness.setdefault(position, []).append(score)
        # Every content word must be there. A long query may miss one, below
        # the labels that have them all.
        needed = len(words) if len(words) < 3 else len(words) - 1
        ranked = []
        for position, scores in closeness.items():
            if len(scores) < needed:
                continue
            entry, _, size = self._docs[position]
            label_says = _with_or_without(entry["label_es"])
            ranked.append(
                (
                    len(scores) < len(words),
                    bool(query_says and label_says and query_says != label_says),
                    # The closest spelling first, so "hiperrflexia" gives
                    # Hiperreflexia ahead of Hiporreflexia.
                    -sum(scores),
                    size - len(scores),
                    len(entry["label_es"]),
                    _hpo_numeric_id(entry["hpo_id"]),
                    entry,
                )
            )
        ranked.sort(key=lambda item: item[:6])
        return ranked

    def by_words(self, query: str) -> List[Dict[str, str]]:
        words = list(dict.fromkeys(_search_words(query)))
        if not words:
            return []
        query_says = _with_or_without(query)
        ranked = self._ranked(words, query_says, relaxed=False)
        if not ranked or ranked[0][0]:
            retried = self._ranked(words, query_says, relaxed=True)
            if retried and not retried[0][0]:
                ranked = retried
        return [item[-1] for item in ranked]


def load_hpo_es_json(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(k): str(v) for k, v in payload.items()}
    return {}


def _hpo_numeric_id(hpo_id: str) -> int:
    match = re.search(r"HP:(\d+)", hpo_id)
    return int(match.group(1)) if match else 10_000_000


def _format_es_result(entry: Dict[str, str], terms_by_id: Dict[str, HpoTerm]) -> Dict[str, str]:
    hpo_id = entry["hpo_id"]
    label_en = terms_by_id[hpo_id].label if hpo_id in terms_by_id else ""
    return {"hpo_id": hpo_id, "label_es": entry["label_es"], "label_en": label_en}
