from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    text_es: str = Field(..., min_length=1, max_length=10000)
    patient_locale: str = Field(default="es-AR")


class ModelInfo(BaseModel):
    symptom_extractor: str
    translator: str
    phenotyper: str


class PhenotypeItem(BaseModel):
    hpo_id: str
    label: str
    label_es: str | None = None
    span_es: str
    span_en: str
    start_es: int
    end_es: int
    start_en: int
    end_en: int
    negated: bool
    confidence: float
    matched_by: str


class ExtractResponse(BaseModel):
    symptoms_extracted: List[str]
    model: ModelInfo
    phenotypes: List[PhenotypeItem]
