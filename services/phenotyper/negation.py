from __future__ import annotations

import re
from typing import Iterable

NEGATION_CUES = [
    "no",
    "not",
    "denies",
    "denied",
    "without",
    "negative for",
    "absence of",
    "no evidence of",
    "free of",
]


def is_negated(text: str, span_start: int) -> bool:
    window_start = max(0, span_start - 80)
    window = text[window_start:span_start].lower()
    for cue in NEGATION_CUES:
        if re.search(rf"\b{re.escape(cue)}\b", window):
            return True
    return False
