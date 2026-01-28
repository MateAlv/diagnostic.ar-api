from __future__ import annotations

import logging
from typing import Protocol, Union

logger = logging.getLogger(__name__)


class Translator(Protocol):
    """Protocol defining the translator interface."""

    model_name: str

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text from source language to target language."""
        ...


def create_translator(
    translator_type: str = "tower",
    # NLLB settings
    nllb_model_name: str = "auto",
    nllb_device: str = "auto",
    nllb_max_length: int = 512,
    # Tower settings
    tower_model_name: str = "auto",
    tower_fallback_model: str = "auto",
    tower_device: str = "auto",
    tower_max_length: int = 1024,
    tower_load_in_4bit: bool = True,
    tower_use_flash_attention: bool = True,
) -> Translator:
    """
    Factory function to create a translator based on configuration.

    Args:
        translator_type: "tower" (recommended for medical) or "nllb"
        nllb_*: Settings for NLLB translator
        tower_*: Settings for Tower translator

    Returns:
        A translator instance implementing the Translator protocol
    """
    translator_type = translator_type.lower().strip()

    if translator_type == "tower":
        from .tower import TowerTranslator

        logger.info("Creating Tower translator (medical-optimized)")
        return TowerTranslator(
            model_name=tower_model_name,
            fallback_model_name=tower_fallback_model,
            device=tower_device,
            max_length=tower_max_length,
            load_in_4bit=tower_load_in_4bit,
            use_flash_attention=tower_use_flash_attention,
        )
    elif translator_type == "nllb":
        from .nllb import NllbTranslator

        logger.info("Creating NLLB translator")
        return NllbTranslator(
            model_name=nllb_model_name,
            device=nllb_device,
            max_length=nllb_max_length,
        )
    else:
        raise ValueError(
            f"Unknown translator type: {translator_type}. "
            "Supported types: 'tower' (recommended for medical), 'nllb'"
        )


# Re-export individual translators for direct import
from .nllb import NllbTranslator
from .tower import TowerTranslator

__all__ = [
    "Translator",
    "create_translator",
    "NllbTranslator",
    "TowerTranslator",
]
