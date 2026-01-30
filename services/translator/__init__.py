from __future__ import annotations

from typing import Protocol


class Translator(Protocol):
    """Protocol defining the translator interface."""

    model_name: str

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text from source language to target language."""
        ...


from .nllb import NllbTranslator

__all__ = [
    "Translator",
    "NllbTranslator",
]
