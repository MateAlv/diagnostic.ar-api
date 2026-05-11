from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

TRANSLATION_PROMPT = """Translate the following Spanish medical symptom or term to English.
Return ONLY the English translation — no explanations, no punctuation around it, no extra text.

Spanish: {text}
English:"""


class OllamaTranslator:
    """Translate Spanish medical terms to English via Ollama LLM."""

    def __init__(
        self,
        ollama_url: str = "http://llm-ollama:11434",
        model: str = "qwen2.5:7b",
        timeout: float = 30.0,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def translate(self, text: str, src_lang: str = "spa_Latn", tgt_lang: str = "eng_Latn") -> str:
        if not text or not text.strip():
            return ""

        prompt = TRANSLATION_PROMPT.format(text=text.strip())
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 64},
        }

        try:
            response = self.client.post(f"{self.ollama_url}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "").strip()
            logger.debug("OllamaTranslator: '%s' → '%s'", text, content)
            return content
        except Exception as e:
            logger.warning("OllamaTranslator failed for '%s': %s", text, e)
            return ""
