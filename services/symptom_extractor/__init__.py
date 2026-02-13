from __future__ import annotations

import logging
import os
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

# Strict extraction prompt - clinical symptoms/signs only, HPO-style atomic
EXTRACTION_PROMPT_ES = """Eres un experto en terminología HPO (Human Phenotype Ontology).

TAREA: Extraer TODOS los síntomas y signos clínicos del texto. NO OMITAS NINGUNO.

REGLAS CRÍTICAS:
1. Un síntoma por línea, ATÓMICO (1-3 palabras, máximo 4 si es imprescindible).
2. Usa el NOMBRE CLÍNICO CANÓNICO más corto posible.
3. ELIMINA todo modificador: localización, duración, grado, medida, descripción física.
4. Si hay dos síntomas unidos por "con"/"y"/"e", listarlos SEPARADOS.
5. Si existe una forma ESPECÍFICA reconocida, usa esa (ej: "vómitos en proyectil", NO "vómitos").
6. Incluye antecedentes clínicos del paciente.
7. NO inventes. NO expliques. NO reformules. SOLO lista.

ATOMIZACIÓN — transforma así:
"hepatomegalia palpable a 3 cm del reborde costal derecho" → hepatomegalia
"dificultad respiratoria con tiraje intercostal" → dificultad respiratoria + tiraje intercostal (DOS líneas)
"edema de miembros inferiores con fóvea positiva" → edema de miembros inferiores
"hipotonía muscular desde el nacimiento" → hipotonía muscular
"temblor intencional en miembros superiores" → temblor intencional
"dolor abdominal difuso tipo cólico" → dolor abdominal
"diarrea acuosa sin sangre" → diarrea
"sordera neurosensorial bilateral diagnosticada a los 2 años" → sordera neurosensorial
"convulsiones tónico-clónicas generalizadas de 2 min" → convulsiones tónico-clónicas
"cefalea intensa holocraneana" → cefalea
"ictericia escleral leve" → ictericia
"cianosis perioral" → cianosis
"edema bipalpebral" → edema palpebral
"retraso del desarrollo psicomotor" → retraso del desarrollo psicomotor
"estrabismo convergente" → estrabismo
"ataxia de la marcha" → ataxia
"nistagmo horizontal" → nistagmo
"pérdida de peso de 4 kg en el último mes" → pérdida de peso
"pérdida de conocimiento" → pérdida de conocimiento
"hipoglucemia sintomática" → hipoglucemia
"saturación de oxígeno del 88%" → desaturación de oxígeno
"fiebre alta persistente de 39.5°C" → fiebre

NO INCLUIR (ignorar completamente):
- Datos demográficos (edad, sexo, familiares)
- Valores numéricos aislados (39.5°C, 3 cm, 88%, 4 kg)
- Localizaciones sin fenotipo (cabeza, cuello)
- Diagnósticos (meningitis, diabetes)
- Tratamientos y fármacos
- Procedimientos y estudios
- Verbos narrativos (presenta, refiere, se observa)

Texto médico:
{text}

LISTA (un síntoma por línea, sin guiones, sin números):"""


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
                "num_predict": 2048,  # Max tokens for response (increased for long symptom lists)
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
