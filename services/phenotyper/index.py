from __future__ import annotations

import json
import logging
import re
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
    def __init__(self, terms: List[HpoTerm]) -> None:
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

    @classmethod
    def load_or_build(
        cls,
        obo_path: str,
        index_path: str,
        download_url: str,
    ) -> "HpoIndex":
        obo = Path(obo_path)
        index = Path(index_path)
        if not obo.exists():
            download_hpo(download_url, obo)
        if index.exists():
            payload = json.loads(index.read_text(encoding="utf-8"))
            terms = [HpoTerm(**item) for item in payload]
            return cls(terms)
        terms = build_hpo_index(obo, index)
        return cls(terms)

    def match(self, text: str, fuzzy_cutoff: int = 75) -> Optional[Tuple[str, str, str, float]]:
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
            return None
        matched_key, score, _ = match
        hpo_id = self.label_map.get(matched_key) or self.synonym_map.get(matched_key)
        if not hpo_id:
            return None
        label = self.terms_by_id[hpo_id].label
        confidence = 0.5 + (score / 100.0) * 0.35
        return hpo_id, label, "fuzzy", confidence
