"""GenPhenAI gateway service.

This service exposes ``/rank-genes`` for the diagnostic.ar frontend and delegates
gene ranking to the real GenPhenia inference API (``POST /predict``).
It preserves the existing response schema expected by the UI and computes
specialties/confidence from the returned score distribution.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("genphenai")

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
_DATA_DIR = Path(os.getenv("GENPHENIA_DATA_DIR", "/app/data"))
_PANELAPP_PATH = _DATA_DIR / "panelapp_panels.csv"
_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
_INFERENCE_URL = os.getenv("GENPHENIA_INFERENCE_URL", "http://host.docker.internal:8002/predict")
_INFERENCE_TIMEOUT_SECONDS = float(os.getenv("GENPHENIA_INFERENCE_TIMEOUT_SECONDS", "30"))
_POSTPROCESS_TOP_K = int(os.getenv("GENPHENIA_POSTPROCESS_TOP_K", "200"))
_INFERENCE_MAX_TOP_K = int(os.getenv("GENPHENIA_INFERENCE_MAX_TOP_K", "5229"))

# ---------------------------------------------------------------------------
# The genphenia post-processing scripts are copied next to this file in the
# Docker image. They import each other with bare names, so we add their
# directory to sys.path.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).parent.parent  # /app  (confidence.py, specialist_recommendation.py)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
app = FastAPI(title="GenPhenAI", version="0.2.0-inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level singletons populated on startup
_gene_to_specialties: Optional[Dict[str, Any]] = None
_recommend_specialties = None
_confidence_from_distribution = None
_http_client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def startup() -> None:
    global _gene_to_specialties, _recommend_specialties, _confidence_from_distribution, _http_client

    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(_INFERENCE_TIMEOUT_SECONDS))
    logger.info("Configured inference endpoint: %s", _INFERENCE_URL)

    # Load genphenia post-processing scripts
    try:
        from specialist_recommendation import (  # type: ignore
            load_panelapp_gene_to_specialties,
            recommend_specialties,
        )
        from confidence import confidence_from_distribution  # type: ignore

        _gene_to_specialties = load_panelapp_gene_to_specialties(_PANELAPP_PATH)
        _recommend_specialties = recommend_specialties
        _confidence_from_distribution = confidence_from_distribution
        logger.info("PanelApp loaded: %d genes with specialty mappings", len(_gene_to_specialties))
    except Exception as exc:
        logger.warning("genphenia post-processing unavailable: %s", exc)


@app.on_event("shutdown")
async def shutdown() -> None:
    if _http_client is not None:
        await _http_client.aclose()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RankRequest(BaseModel):
    hpo_ids: List[str] = Field(..., min_length=1, description="Patient HPO IDs")
    top_k: int = Field(10, ge=1, le=200)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coerce_score_map(payload: Any) -> Dict[str, float]:
    """Normalize inference payload into ``{gene_symbol: probability}``."""
    if not isinstance(payload, dict):
        raise ValueError("Inference response is not a JSON object")

    # Current inference API default shape:
    # {"GENE1": 0.12, "GENE2": 0.08, ...}
    if payload and all(isinstance(v, (int, float)) for v in payload.values()):
        score_map: Dict[str, float] = {}
        for gene, raw_score in payload.items():
            gene_symbol = str(gene).strip().upper()
            if not gene_symbol:
                continue
            score_map[gene_symbol] = float(raw_score)
        if score_map:
            return score_map

    # Compatibility fallback in case inference is asked for full payload:
    # {"top_predictions": [{"gene": "...", "probability": ...}, ...], ...}
    top_predictions = payload.get("top_predictions")
    if isinstance(top_predictions, list):
        score_map = {}
        for item in top_predictions:
            if not isinstance(item, dict):
                continue
            gene_symbol = str(item.get("gene", "")).strip().upper()
            probability = item.get("probability")
            if not gene_symbol or not isinstance(probability, (int, float)):
                continue
            score_map[gene_symbol] = float(probability)
        if score_map:
            return score_map

    raise ValueError("Inference response does not contain gene probability scores")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "mode": "inference-http",
        "inference_url": _INFERENCE_URL,
        "postprocess_top_k": _POSTPROCESS_TOP_K,
        "inference_max_top_k": _INFERENCE_MAX_TOP_K,
        "specialties_available": _gene_to_specialties is not None,
    }


@app.post("/rank-genes")
async def rank_genes(request: RankRequest) -> Dict[str, Any]:
    if _http_client is None:
        raise HTTPException(status_code=503, detail="Service not ready — HTTP client not initialized")

    input_count = len(request.hpo_ids)
    # Always fetch the full distribution so confidence_from_distribution and
    # recommend_specialties operate on all 5229 genes, not just the user's top_k.
    # The visible results are sliced to request.top_k below.
    inference_payload = {"hpo_ids": request.hpo_ids, "top_k": _INFERENCE_MAX_TOP_K}

    try:
        inference_response = await _http_client.post(_INFERENCE_URL, json=inference_payload)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="genphenia-inference-timeout") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="genphenia-inference-unreachable") from exc

    if inference_response.status_code >= 400:
        detail: Any = "genphenia-inference-error"
        try:
            payload = inference_response.json()
            detail = payload.get("detail", payload)
        except Exception:
            detail = inference_response.text or detail
        raise HTTPException(status_code=502, detail=detail)

    try:
        score_map = _coerce_score_map(inference_response.json())
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Sort genes by probability descending, take user top_k for visible results
    sorted_genes = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    top_genes = sorted_genes[: request.top_k]

    results = [
        {
            "gene_id": gene_symbol,       # no NCBI ID in inference payload — use symbol
            "gene_symbol": gene_symbol,
            "score": round(score, 6),
            "matched_hpo_ids": [],        # model does not return per-gene matched HPOs
            "matched_count": 0,
            "input_count": input_count,
            "coverage": 0.0,
            "precision": 0.0,
        }
        for gene_symbol, score in top_genes
    ]

    specialties: Optional[Dict[str, Any]] = None
    if _recommend_specialties is not None and _gene_to_specialties is not None:
        try:
            specialties = _recommend_specialties(score_map, _gene_to_specialties)
        except Exception as exc:
            logger.warning("recommend_specialties failed: %s", exc)

    confidence: Optional[float] = None
    if _confidence_from_distribution is not None and score_map:
        try:
            confidence = _confidence_from_distribution(score_map)
        except Exception as exc:
            logger.warning("confidence_from_distribution failed: %s", exc)

    return {
        "results": results,
        "specialties": specialties,
        "confidence": confidence,
        "input_hpo_count": input_count,
        "beta": False,
    }
