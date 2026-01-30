from __future__ import annotations

import logging
import os
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

# Strict extraction prompt - clinical symptoms/signs only, HPO-style
EXTRACTION_PROMPT_ES = """Eres un experto en terminología médica y fenotipos clínicos (Human Phenotype Ontology, HPO).

Tu tarea es EXTRAER ÚNICAMENTE síntomas y signos clínicos presentes en el paciente,
mencionados explícitamente en el texto.

Reglas obligatorias:
- Extrae SOLO síntomas o signos clínicos.
- Usa frases clínicas CORTAS y CANÓNICAS (2-6 palabras).
- No expliques, no reformules, no agregues contexto.
- No inventes síntomas.
- Si un síntoma aparece varias veces, inclúyelo UNA sola vez.
- Si un síntoma tiene una forma específica, NO incluyas la forma genérica
  (ej.: incluir "vómitos proyectiles" y NO "vómitos").

INCLUIR (ejemplos válidos):
- cefalea holocraneal
- fiebre alta
- vómitos proyectiles
- convulsiones tónico-clónicas generalizadas
- rigidez de nuca
- edema bipalpebral
- edema con fóvea
- hepatomegalia
- esplenomegalia
- ictericia escleral
- dificultad respiratoria
- cianosis perioral
- diarrea acuosa
- pérdida de peso
- hipotonía muscular
- retraso del desarrollo psicomotor
- microcefalia
- hipoacusia neurosensorial bilateral
- ataxia de la marcha
- nistagmo horizontal
- disartria
- temblor intencional

NO INCLUIR (ignorar completamente):
- Datos demográficos: edad, sexo, familiares ("8 años", "masculino", "la madre")
- Duración o frecuencia: "3 días", "2 minutos", "episodios"
- Valores numéricos o medidas: "39.5°C", "3 cm", "88%"
- Localizaciones aisladas sin fenotipo: "cabeza", "cuello", "extremidades"
- Negaciones explícitas o implícitas: "no presenta", "niega", "sin"
- Diagnósticos o sospechas diagnósticas: "meningitis", "diabetes"
- Tratamientos o fármacos: "ibuprofeno", "antibióticos"
- Procedimientos o estudios: "examen físico", "resonancia"
- Verbos narrativos o relleno clínico: "paciente", "refiere", "presenta", "se observa"

Texto médico:
{text}

Formato de salida OBLIGATORIO:
- Una lista plana
- Un síntoma por línea
- Sin numeración
- Sin guiones
- Sin texto adicional"""


class SymptomExtractor:
    """
    Extracts clinical symptoms from Spanish medical text using a local LLM.

    Uses Ollama server with Qwen2.5-7B-Instruct for high-quality Spanish
    medical understanding.
    """

    def __init__(
        self,
        ollama_url: str = "http://llm-ollama:11434",
        model: str = "qwen2.5:7b",
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def extract(self, text: str) -> List[str]:
        """
        Extract symptom phrases from Spanish medical text.

        Args:
            text: Spanish medical text (clinical note, patient history, etc.)

        Returns:
            List of symptom phrases in Spanish
        """
        if not text or not text.strip():
            return []

        prompt = EXTRACTION_PROMPT_ES.format(text=text.strip())

        for attempt in range(self.max_retries + 1):
            try:
                response = self._call_ollama(prompt)
                symptoms = self._parse_response(response)
                logger.info(
                    "Extracted %d symptoms from text (attempt %d)",
                    len(symptoms),
                    attempt + 1,
                )
                return symptoms
            except Exception as e:
                logger.warning(
                    "Symptom extraction attempt %d failed: %s",
                    attempt + 1,
                    str(e),
                )
                if attempt == self.max_retries:
                    logger.error("All extraction attempts failed, returning empty list")
                    return []

        return []

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for chat completion."""
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent extraction
                "num_predict": 1024,  # Max tokens for response
            },
        }

        response = self.client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        return data.get("message", {}).get("content", "")

    def _parse_response(self, response: str) -> List[str]:
        """Parse LLM response into list of symptom phrases."""
        symptoms = []

        for line in response.strip().split("\n"):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Remove common list prefixes
            for prefix in ["- ", "• ", "* ", "· "]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break

            # Remove numbered prefixes (1. 2. etc)
            if len(line) > 2 and line[0].isdigit() and line[1] in ".)":
                line = line[2:].strip()
            elif len(line) > 3 and line[:2].isdigit() and line[2] in ".)":
                line = line[3:].strip()

            # Skip if too short or looks like a header
            if len(line) < 3:
                continue
            if line.endswith(":"):
                continue

            # Clean up quotes
            line = line.strip('"\'')

            if line:
                symptoms.append(line)

        return symptoms

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = self.client.get(f"{self.ollama_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def get_info(self) -> dict:
        """Get extractor configuration info."""
        return {
            "ollama_url": self.ollama_url,
            "model": self.model,
            "available": self.is_available(),
        }


def create_symptom_extractor(
    ollama_url: str = None,
    model: str = None,
) -> SymptomExtractor:
    """
    Factory function to create a symptom extractor.

    Args:
        ollama_url: Ollama server URL (default from env OLLAMA_URL)
        model: Model name (default from env SYMPTOM_EXTRACTOR_MODEL)

    Returns:
        Configured SymptomExtractor instance
    """
    return SymptomExtractor(
        ollama_url=ollama_url or os.getenv("OLLAMA_URL", "http://llm-ollama:11434"),
        model=model or os.getenv("SYMPTOM_EXTRACTOR_MODEL", "qwen2.5:7b"),
    )
