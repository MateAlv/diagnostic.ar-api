from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


MODEL_9B = "Unbabel/TowerInstruct-7B-v0.2"
MODEL_2B = "Unbabel/TowerBase-2B-v0.1"

# Tower-Plus models (newer, better quality)
MODEL_PLUS_9B = "Unbabel/Tower-Plus-9B"
MODEL_PLUS_2B = "Unbabel/Tower-Plus-2B"


def resolve_device(device_pref: str) -> str:
    if device_pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_pref.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable, falling back to cpu")
        return "cpu"
    return device_pref


def get_gpu_memory_gb() -> float:
    """Returns available GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return 0.0


class TowerTranslator:
    """
    Medical-optimized translator using Unbabel's Tower models.

    Uses Tower-Plus-9B as primary model with automatic fallback to Tower-Plus-2B
    when GPU memory is insufficient or on explicit configuration.

    Tower models outperform GPT-4o, DeepL, and Google Translate for medical,
    legal, technical, and financial domain translations.
    """

    TRANSLATION_PROMPT_TEMPLATE = """Translate the following text from {src_lang} to {tgt_lang}.
Preserve all medical terminology accurately. Do not add explanations or notes.

Source text:
{text}

Translation:"""

    LANG_MAP = {
        "spa_Latn": "Spanish",
        "eng_Latn": "English",
        "por_Latn": "Portuguese",
        "fra_Latn": "French",
        "deu_Latn": "German",
        "ita_Latn": "Italian",
        "rus_Cyrl": "Russian",
        "zho_Hans": "Chinese",
        "kor_Hang": "Korean",
        "nld_Latn": "Dutch",
        # Common shortcuts
        "es": "Spanish",
        "en": "English",
        "pt": "Portuguese",
        "fr": "French",
        "de": "German",
        "it": "Italian",
    }

    def __init__(
        self,
        model_name: str = "auto",
        fallback_model_name: str = "auto",
        device: str = "auto",
        max_length: int = 1024,
        load_in_4bit: bool = True,
        use_flash_attention: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        self.max_length = max_length
        self.load_in_4bit = load_in_4bit and self.device == "cuda"
        self.use_flash_attention = use_flash_attention and self.device == "cuda"

        # Resolve model names
        self._primary_model_name = self._resolve_model_name(model_name, primary=True)
        self._fallback_model_name = self._resolve_model_name(fallback_model_name, primary=False)

        # Track which model is currently loaded
        self._current_model_name: Optional[str] = None
        self._tokenizer = None
        self._model = None
        self._load_attempted = False
        self._using_fallback = False

    def _resolve_model_name(self, model_name: str, primary: bool) -> str:
        if model_name != "auto":
            return model_name
        if primary:
            return MODEL_PLUS_9B
        return MODEL_PLUS_2B

    @property
    def model_name(self) -> str:
        """Returns the currently active model name."""
        if self._current_model_name:
            return self._current_model_name
        return self._primary_model_name

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
        """Load the model with automatic fallback on OOM."""
        if self._load_attempted and self._model is not None:
            return

        self._load_attempted = True

        # Try primary model first
        try:
            self._load_model(self._primary_model_name)
            self._using_fallback = False
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                logger.warning(
                    "Failed to load primary model %s due to OOM, falling back to %s",
                    self._primary_model_name,
                    self._fallback_model_name,
                )
                # Clear GPU memory
                if self._model is not None:
                    del self._model
                    self._model = None
                if self._tokenizer is not None:
                    del self._tokenizer
                    self._tokenizer = None
                torch.cuda.empty_cache()

                # Load fallback model
                self._load_model(self._fallback_model_name)
                self._using_fallback = True
            else:
                raise

    def _load_model(self, model_name: str) -> None:
        """Load a specific model."""
        logger.info(
            "Loading Tower translator model=%s device=%s 4bit=%s flash_attn=%s",
            model_name,
            self.device,
            self.load_in_4bit,
            self.use_flash_attention,
        )

        # Determine torch dtype
        if self.device == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Model loading kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        # 4-bit quantization for GPU
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("Using 4-bit quantization")
            except ImportError:
                logger.warning("bitsandbytes not available, loading without quantization")

        # Flash attention 2 for faster inference (requires flash_attn package)
        if self.use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except ImportError:
                logger.warning("flash_attn package not installed, using default attention (sdpa)")
                self.use_flash_attention = False

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Move to device if not using device_map
        if self.device != "cuda" or "device_map" not in model_kwargs:
            self._model = self._model.to(self.device)

        self._model.eval()
        self._current_model_name = model_name

        # Log memory usage
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info("GPU memory: allocated=%.2fGB reserved=%.2fGB", allocated, reserved)

    def _get_lang_name(self, lang_code: str) -> str:
        """Convert language code to human-readable name."""
        return self.LANG_MAP.get(lang_code, lang_code)

    def translate(self, text: str, src_lang: str = "spa_Latn", tgt_lang: str = "eng_Latn") -> str:
        """
        Translate text from source language to target language.

        Args:
            text: Text to translate
            src_lang: Source language code (NLLB format or short code)
            tgt_lang: Target language code (NLLB format or short code)

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""

        tokenizer = self.tokenizer
        model = self.model

        src_lang_name = self._get_lang_name(src_lang)
        tgt_lang_name = self._get_lang_name(tgt_lang)

        # Build the translation prompt
        prompt = self.TRANSLATION_PROMPT_TEMPLATE.format(
            src_lang=src_lang_name,
            tgt_lang=tgt_lang_name,
            text=text.strip(),
        )

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Generate translation (greedy decoding for speed, can use num_beams>1 for quality)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Reasonable limit for translation
                num_beams=1,  # Greedy decoding (fast), increase for better quality
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (exclude input)
        generated_tokens = outputs[0][input_length:]
        translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up the translation (remove any trailing artifacts)
        translation = translation.strip()

        # Remove common artifacts from instruction-tuned models
        for marker in ["\n\nSource text:", "\n\nTranslation:", "<|endoftext|>", "</s>"]:
            if marker in translation:
                translation = translation.split(marker)[0].strip()

        return translation

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str = "spa_Latn",
        tgt_lang: str = "eng_Latn",
        batch_size: int = 4,
    ) -> list[str]:
        """
        Translate multiple texts in batches for efficiency.

        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
            batch_size: Number of texts to process at once

        Returns:
            List of translated texts
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                results.append(self.translate(text, src_lang, tgt_lang))
        return results

    def is_using_fallback(self) -> bool:
        """Returns True if currently using the fallback model."""
        return self._using_fallback

    def get_model_info(self) -> dict:
        """Returns information about the loaded model."""
        return {
            "model_name": self.model_name,
            "primary_model": self._primary_model_name,
            "fallback_model": self._fallback_model_name,
            "using_fallback": self._using_fallback,
            "device": self.device,
            "load_in_4bit": self.load_in_4bit,
            "flash_attention": self.use_flash_attention,
        }
