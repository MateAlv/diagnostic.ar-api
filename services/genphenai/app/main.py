"""GenPhenAI beta service — returns static genphenia_results.json as gene ranking.

This is a placeholder until the real lightweight model is available.
The ranking is always the same pre-computed gene probability distribution;
it is NOT filtered by input HPO IDs. Specialist recommendations and confidence
are computed from that distribution using the existing genphenia scripts.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("genphenai")

# ---------------------------------------------------------------------------
# Paths (all files live in the genphenia data dir, mounted at /app/data)
# ---------------------------------------------------------------------------
_DATA_DIR = Path(os.getenv("GENPHENIA_DATA_DIR", "/app/data"))
_RESULTS_PATH = _DATA_DIR / "genphenia_results.json"
_PANELAPP_PATH = _DATA_DIR / "panelapp_panels.csv"
_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# ---------------------------------------------------------------------------
# The genphenia post-processing scripts are copied next to this file in the
# Docker image. They import each other with bare names, so we add their
# directory to sys.path.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).parent.parent  # /app  (confidence.py, specialist_recommendation.py)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="GenPhenAI (beta)", version="0.1.0-static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level singletons populated on startup
_scores: Dict[str, float] = {}          # gene_symbol → probability (full distribution)
_gene_to_specialties: Optional[Dict[str, Any]] = None
_recommend_specialties = None
_confidence_from_distribution = None


@app.on_event("startup")
async def startup() -> None:
    global _scores, _gene_to_specialties, _recommend_specialties, _confidence_from_distribution

    # Load static gene probability scores
    logger.info("Loading gene scores from %s", _RESULTS_PATH)
    _scores = json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d gene scores", len(_scores))

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


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RankRequest(BaseModel):
    hpo_ids: List[str] = Field(..., description="Patient HPO IDs (used in future model; ignored in beta)")
    top_k: int = Field(10, ge=1, le=200)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "mode": "beta-static",
        "gene_count": len(_scores),
        "specialties_available": _gene_to_specialties is not None,
    }


@app.post("/rank-genes")
async def rank_genes(request: RankRequest) -> Dict[str, Any]:
    if not _scores:
        raise HTTPException(status_code=503, detail="Service not ready — gene scores not loaded")

    input_count = len(request.hpo_ids)

    # Sort genes by probability descending, take top_k
    sorted_genes = sorted(_scores.items(), key=lambda x: x[1], reverse=True)
    top_genes = sorted_genes[: request.top_k]

    results = [
        {
            "gene_id": gene_symbol,       # no NCBI ID in static data — use symbol
            "gene_symbol": gene_symbol,
            "score": round(score, 6),
            "matched_hpo_ids": [],        # no real matching in beta
            "matched_count": 0,
            "input_count": input_count,
            "coverage": 0.0,
            "precision": 0.0,
        }
        for gene_symbol, score in top_genes
    ]

    # Build score map for post-processing (use full distribution for meaningful specialties)
    score_map = dict(top_genes)

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
        "beta": True,
    }
