"""GenPhenAI gateway service.

This service exposes ``/rank-genes`` for the diagnostic.ar frontend. The
GenPhenia v3.0 model runs in this process from the bundle under ``model/``,
exported to ONNX so it needs neither torch nor a GPU. The response keeps the
schema the UI already consumes, and specialties/confidence are computed from
the full score distribution.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .errors import UnhandledErrorsAsJson

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("genphenai")

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
_SERVICE_DIR = Path(__file__).parent.parent  # confidence.py, specialist_recommendation.py, inference_engine/
_DATA_DIR = Path(os.getenv("GENPHENIA_DATA_DIR", str(_SERVICE_DIR / "data")))
_PANELAPP_PATH = _DATA_DIR / "panelapp_panels.csv"
_BUNDLE_DIR = Path(os.getenv("GENPHENIA_BUNDLE_DIR", str(_SERVICE_DIR / "model" / "genphenia-v3.0")))
_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
_POSTPROCESS_TOP_K = int(os.getenv("GENPHENIA_POSTPROCESS_TOP_K", "200"))
_INFERENCE_MAX_TOP_K = int(os.getenv("GENPHENIA_INFERENCE_MAX_TOP_K", "5229"))

_HPO_ID_RE = re.compile(r"^HP:\d+$")

# The post-processing scripts and the vendored engine import each other with
# bare names, so their directory goes on sys.path.
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
# Module-level singletons populated on startup
_engine = None
_gene_to_specialties: Optional[Dict[str, Any]] = None
_recommend_specialties = None
_confidence_from_distribution = None
# The engine's preprocessing cache is a plain OrderedDict; requests run in a
# thread pool, so predictions are serialised. One takes a few milliseconds.
_predict_lock = threading.Lock()


def load_engine(bundle_dir: Path):
    from inference_engine.engine import GenpheniaEngine

    # The bundle is baked into the image, so there is nothing to hot-reload;
    # an infinite interval turns off the engine's per-request registry check.
    return GenpheniaEngine.load(
        bundle_dir,
        auto_refresh_seconds=float("inf"),
        auto_activate_latest=True,
    )


def startup() -> None:
    global _engine, _gene_to_specialties, _recommend_specialties, _confidence_from_distribution

    # Without the model the service has nothing to answer, so this is fatal.
    _engine = load_engine(_BUNDLE_DIR)
    logger.info(
        "GenPhenia %s loaded from %s: %d genes",
        _engine.record.model_version,
        _BUNDLE_DIR,
        len(_engine.genes),
    )

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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    startup()
    yield


app = FastAPI(title="GenPhenAI", version="0.3.0-in-process", lifespan=lifespan)

# Before CORS, so that CORS wraps it: see UnhandledErrorsAsJson.
app.add_middleware(UnhandledErrorsAsJson)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RankRequest(BaseModel):
    hpo_ids: List[str] = Field(..., min_length=1, description="Patient HPO IDs")
    top_k: int = Field(50, ge=1, le=_INFERENCE_MAX_TOP_K)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def split_known_ids(engine, hpo_ids: List[str]) -> tuple[List[str], List[str]]:
    """Separate the IDs the model's HPO release knows from the ones it does not.

    The search index can be on a newer HPO release than the model. A term the
    model has never seen used to fail the whole ranking; it is now left out
    and reported, and only a request with nothing usable is rejected.
    """
    from inference_engine.preprocess import normalize_hpo_id

    known_ids = engine.preprocessor.hpo_dag.id_to_idx
    known: List[str] = []
    unknown: List[str] = []
    for raw_id in hpo_ids:
        normalized = normalize_hpo_id(raw_id)
        if not _HPO_ID_RE.match(normalized):
            raise HTTPException(status_code=422, detail=f"Invalid HPO ID: {raw_id!r}")
        (known if normalized in known_ids else unknown).append(normalized)
    return known, unknown


def predict_scores(engine, hpo_ids: List[str]) -> Dict[str, float]:
    """``{gene_symbol: probability}`` over every gene the model outputs."""
    with _predict_lock:
        result = engine.predict(hpo_ids=hpo_ids, top_k=_INFERENCE_MAX_TOP_K)
    return {
        str(item["gene"]).strip().upper(): float(item["probability"])
        for item in result["top_predictions"]
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {
        "status": "ok" if _engine is not None else "loading",
        "mode": "in-process",
        "model_version": _engine.record.model_version if _engine is not None else None,
        "checkpoint_sha256": _engine.record.checkpoint_sha256 if _engine is not None else None,
        "postprocess_top_k": _POSTPROCESS_TOP_K,
        "inference_max_top_k": _INFERENCE_MAX_TOP_K,
        "specialties_available": _gene_to_specialties is not None,
    }


@app.post("/rank-genes")
def rank_genes(request: RankRequest) -> Dict[str, Any]:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Service not ready — model not loaded")

    known_ids, unknown_ids = split_known_ids(_engine, request.hpo_ids)
    if unknown_ids:
        logger.info("ignoring HPO IDs unknown to the model: %s", ", ".join(unknown_ids))
    if not known_ids:
        raise HTTPException(
            status_code=422,
            detail={"error": "no-known-hpo-ids", "unknown_hpo_ids": unknown_ids},
        )

    input_count = len(request.hpo_ids)
    # The full distribution, so confidence_from_distribution and
    # recommend_specialties operate on all 5229 genes, not just the user's top_k.
    # The visible results are sliced to request.top_k below.
    score_map = predict_scores(_engine, known_ids)

    sorted_genes = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    top_genes = sorted_genes[: request.top_k]

    results = [
        {
            "gene_id": gene_symbol,       # the model has no NCBI IDs — use the symbol
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
        "unknown_hpo_ids": unknown_ids,
        "beta": False,
    }
