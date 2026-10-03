"""GenPhenAI gateway service.

This service exposes ``/rank-genes`` for the diagnostic.ar frontend. The
GenPhenia v3.0 model runs in this process from the bundle under ``model/``,
exported to ONNX so it needs neither torch nor a GPU. The response keeps the
schema the UI already consumes, and specialties/confidence are computed from
the full score distribution.

When model.lock pins a re-ranker (rr-v1), a request may name the HPO terms the
patient does not have (``excluded_hpo_ids``, the question agent's "no"
answers): the whole distribution is re-ranked before anything is read from it.
Without them the output is the model's own, bit for bit.
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
_MODEL_LOCK = Path(os.getenv("GENPHENIA_MODEL_LOCK", str(_SERVICE_DIR / "model.lock")))
_RERANK_DIR = Path(os.getenv("RERANK_BUNDLE_DIR", str(_SERVICE_DIR / "model" / "rerank-v1.0")))

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
_reranker = None
_rerank_name: Optional[str] = None
_rerank_pin: Dict[str, str] = {}
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


def read_lock(path: Path) -> Dict[str, str]:
    """model.lock as a dict; empty when there is none (a checkout without it)."""
    if not path.is_file():
        return {}
    pairs: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            pairs[key] = value
    return pairs


def load_reranker(bundle_dir: Path, genes):
    """rr-v1 over the engine's genes, after every bundle file matches its sha256."""
    from rerank import load
    from rerank.bundle import read_config

    return load(bundle_dir, genes), read_config(bundle_dir)["name"]


def startup() -> None:
    global _engine, _gene_to_specialties, _recommend_specialties, _confidence_from_distribution
    global _reranker, _rerank_name, _rerank_pin

    # Without the model the service has nothing to answer, so this is fatal.
    _engine = load_engine(_BUNDLE_DIR)
    logger.info(
        "GenPhenia %s loaded from %s: %d genes",
        _engine.record.model_version,
        _BUNDLE_DIR,
        len(_engine.genes),
    )

    # Pinned means required: a missing or altered bundle stops the service
    # rather than serving the model without the re-ranker it was released with.
    lock = read_lock(_MODEL_LOCK)
    if lock.get("RERANK_TAG"):
        _reranker, _rerank_name = load_reranker(_RERANK_DIR, _engine.genes)
        _rerank_pin = {"release": lock["RERANK_TAG"], "sha256": lock.get("RERANK_SHA256", "")}
        logger.info("re-ranker %s (%s) loaded from %s", _rerank_name, lock["RERANK_TAG"], _RERANK_DIR)

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
    excluded_hpo_ids: List[str] = Field(
        default_factory=list,
        description=(
            "HPO IDs the patient does not have: the question agent's effective 'no' answers. "
            "Never negations from free text, never 'no sé'. Ignored when no re-ranker is pinned."
        ),
    )


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


def predict_reranked(engine, reranker, hpo_ids: List[str], excluded: List[str]) -> Dict[str, float]:
    """The same distribution, re-ranked with the excluded terms before anything reads it.

    The re-ranker works on the engine's distribution, in the engine's gene
    order and spelling; only then are the symbols upper-cased, so results,
    specialties and confidence all come from the one re-ranked distribution.
    """
    with _predict_lock:
        result = engine.predict(hpo_ids=hpo_ids, top_k=1, return_distribution=True)
    distribution = reranker.rerank_distribution(result["distribution"], excluded)
    return {str(gene).strip().upper(): float(p) for gene, p in distribution.items()}


def split_excluded_ids(reranker, excluded_ids: List[str], present: List[str]) -> tuple[List[str], List[str]]:
    """The excluded terms the re-ranker can use, and the ones its ontology does not know.

    A term also given as present is dropped: the patient cannot both have and
    not have it, and the present one wins. Unknown terms are reported and move
    nothing.
    """
    from inference_engine.preprocess import normalize_hpo_id

    present_set = set(present)
    seen: set = set()
    usable: List[str] = []
    for raw_id in excluded_ids:
        normalized = normalize_hpo_id(raw_id)
        if not _HPO_ID_RE.match(normalized):
            raise HTTPException(status_code=422, detail=f"Invalid HPO ID: {raw_id!r}")
        if normalized in present_set or normalized in seen:
            continue
        seen.add(normalized)
        usable.append(normalized)
    unknown = reranker.unknown_terms(usable)
    return [t for t in usable if t not in unknown], unknown


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
        "rerank": (
            {"name": _rerank_name, "release": _rerank_pin.get("release"), "sha256": _rerank_pin.get("sha256")}
            if _reranker is not None
            else None
        ),
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
    excluded: List[str] = []
    unknown_excluded: List[str] = []
    if _reranker is not None and request.excluded_hpo_ids:
        excluded, unknown_excluded = split_excluded_ids(_reranker, request.excluded_hpo_ids, known_ids)
    # The full distribution, so confidence_from_distribution and
    # recommend_specialties operate on all 5229 genes, not just the user's top_k.
    # The visible results are sliced to request.top_k below.
    if excluded:
        score_map = predict_reranked(_engine, _reranker, known_ids, excluded)
    else:
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
        # Set only when excluded terms reached the re-ranker: the question agent
        # tells people their "no" answers moved the list only when it is.
        "rerank": _rerank_name if excluded else None,
        "excluded_hpo_count": len(excluded),
        "unknown_excluded_hpo_ids": unknown_excluded,
        "beta": False,
    }
