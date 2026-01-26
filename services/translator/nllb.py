from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


DEFAULT_MODEL_LARGE = "facebook/nllb-200-1.3B"
DEFAULT_MODEL_SMALL = "facebook/nllb-200-distilled-600M"


def resolve_device(device_pref: str) -> str:
    if device_pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_pref.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable, falling back to cpu")
        return "cpu"
    return device_pref


def resolve_model_name(model_name: str, device: str) -> str:
    if model_name != "auto":
        return model_name
    if device == "cuda":
        return DEFAULT_MODEL_LARGE
    return DEFAULT_MODEL_SMALL


class NllbTranslator:
    def __init__(self, model_name: str, device: str, max_length: int = 512) -> None:
        self.device = resolve_device(device)
        self.model_name = resolve_model_name(model_name, self.device)
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    def _load(self) -> None:
        logger.info("loading translator model=%s device=%s", self.model_name, self.device)
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, torch_dtype=torch_dtype
        ).to(self.device)
        self._model.eval()

    def _resolve_lang_id(self, tokenizer, lang_code: str) -> int:
        lang_map = getattr(tokenizer, "lang_code_to_id", None)
        if isinstance(lang_map, dict):
            lang_id = lang_map.get(lang_code)
            if isinstance(lang_id, int):
                return lang_id
        get_lang_id = getattr(tokenizer, "get_lang_id", None)
        if callable(get_lang_id):
            lang_id = get_lang_id(lang_code)
            if isinstance(lang_id, int):
                return lang_id
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            lang_id = convert(lang_code)
            if isinstance(lang_id, int) and getattr(tokenizer, "unk_token_id", None) != lang_id:
                return lang_id
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            vocab = get_vocab()
            if isinstance(vocab, dict) and lang_code in vocab:
                return vocab[lang_code]
        raise ValueError(f"Unsupported target language code: {lang_code}")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text:
            return ""
        tokenizer = self.tokenizer
        model = self.model

        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        forced_bos_token_id = self._resolve_lang_id(tokenizer, tgt_lang)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.max_length,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded[0] if decoded else ""
