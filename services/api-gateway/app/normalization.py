from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import yaml


@dataclass(frozen=True)
class NormalizationRule:
    pattern: re.Pattern
    replace: str


class Normalizer:
    def __init__(self, rules_path: str) -> None:
        self.rules = self._load_rules(rules_path)

    def _load_rules(self, rules_path: str) -> List[NormalizationRule]:
        path = Path(rules_path)
        if not path.exists():
            return []
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rules = payload.get("rules", [])
        compiled: List[NormalizationRule] = []
        for rule in rules:
            pattern = rule.get("pattern")
            replace = rule.get("replace", "")
            if not pattern:
                continue
            compiled.append(
                NormalizationRule(re.compile(pattern, re.IGNORECASE), replace)
            )
        return compiled

    def apply(self, text: str) -> str:
        normalized = text
        for rule in self.rules:
            normalized = rule.pattern.sub(rule.replace, normalized)
        return normalized
