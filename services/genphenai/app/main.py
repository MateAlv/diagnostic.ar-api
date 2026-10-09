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

When model.lock also pins ft-v1.1 and FT_V11_ENABLED is set, a request may
give the patient's sex, age and age at onset as well: ft-v1.1 lowers the genes
those, or the excluded terms, contradict. While it is on it takes rr-v1's
place, since E0 plus ft-v1.1 is the system that was measured. Off, or without
any of those fields, the output is again the model's own, bit for bit.
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
_FT_DIR = Path(os.getenv("FT_BUNDLE_DIR", str(_SERVICE_DIR / "model" / "ft-v1.1-confirmed")))

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
_ft_model = None
_ft_name: Optional[str] = None
_ft_pin: Dict[str, str] = {}
_ft_enabled = False
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


def load_ft(package_dir: Path, genes):
    """ft-v1.1 over the engine's genes, after ft_prod.py and the bundle match ft.json."""
    import ft

    return ft.load(package_dir, genes), ft.read_config(package_dir)["name"]


def startup() -> None:
    global _engine, _gene_to_specialties, _recommend_specialties, _confidence_from_distribution
    global _reranker, _rerank_name, _rerank_pin, _ft_model, _ft_name, _ft_pin, _ft_enabled

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

    # Loaded whenever it is pinned, so that the flag alone turns it on or off.
    if lock.get("FT_TAG"):
        import ft

        _ft_model, _ft_name = load_ft(_FT_DIR, _engine.genes)
        config = ft.read_config(_FT_DIR)
        _ft_pin = {
            "release": lock["FT_TAG"],
            "sha256": lock.get("FT_SHA256", ""),
            "bundle_sha256": next(e["sha256"] for n, e in config["files"].items() if n.endswith(".json.gz")),
        }
        _ft_enabled = ft.enabled()
        logger.info(
            "%s (%s) loaded from %s, %s",
            _ft_name,
            lock["FT_TAG"],
            _FT_DIR,
            "enabled" if _ft_enabled else "disabled (FT_V11_ENABLED)",
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
    excluded_hpo_ids: List[str] = Field(
        default_factory=list,
        description=(
            "HPO IDs the patient does not have: the question agent's effective 'no' answers. "
            "Never negations from free text, never 'no sé'. Ignored when no re-ranker is pinned."
        ),
    )
    # Read only by ft-v1.1, and only when FT_V11_ENABLED is set. Anything the
    # model cannot use counts as unknown rather than failing the request.
    sex: Optional[str] = Field(
        None, description="'male' or 'female'; anything else is unknown. Read only by ft-v1.1."
    )
    age_years: Optional[float] = Field(None, description="Current age in years. Read only by ft-v1.1.")
    onset_years: Optional[float] = Field(
        None,
        description="Age at onset in years; used instead of the age when given. Read only by ft-v1.1.",
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


def predict_ft(engine, model, hpo_ids: List[str], inputs: Dict[str, Any]) -> Dict[str, float]:
    """v3.0's distribution with ft-v1.1's residual, before anything reads it.

    The residual works on the engine's probabilities in the engine's gene order
    (ft.load checked the bundle follows it); only then are the symbols
    upper-cased, so results, specialties and confidence all come from it.
    """
    import ft
    import numpy as np

    with _predict_lock:
        result = engine.predict(hpo_ids=hpo_ids, top_k=1, return_distribution=True)
    distribution = result["distribution"]
    genes = list(distribution)
    if tuple(genes) != model.genes:
        raise ValueError("the engine's distribution does not follow ft-v1.1's gene order")
    probs = ft.apply(model, np.fromiter(distribution.values(), dtype=np.float64, count=len(genes)), inputs, True)
    return {str(gene).strip().upper(): float(p) for gene, p in zip(genes, probs)}


def normalize_excluded_ids(excluded_ids: List[str], present: List[str]) -> List[str]:
    """Excluded terms, normalised and deduplicated, without any also given as present."""
    from inference_engine.preprocess import normalize_hpo_id

    present_set = set(present)
    usable: List[str] = []
    for raw_id in excluded_ids:
        normalized = normalize_hpo_id(raw_id)
        if not _HPO_ID_RE.match(normalized):
            raise HTTPException(status_code=422, detail=f"Invalid HPO ID: {raw_id!r}")
        if normalized not in present_set and normalized not in usable:
            usable.append(normalized)
    return usable


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
        "ft": (
            {
                "name": _ft_name,
                "release": _ft_pin.get("release"),
                "sha256": _ft_pin.get("sha256"),
                "bundle_sha256": _ft_pin.get("bundle_sha256"),
                "enabled": _ft_enabled,
            }
            if _ft_model is not None
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
    ft_applied = False
    applied_to_excluded: Optional[str] = None
    # The full distribution, so confidence_from_distribution and
    # recommend_specialties operate on all 5229 genes, not just the user's top_k.
    # The visible results are sliced to request.top_k below.
    if _ft_model is not None and _ft_enabled:
        import ft

        excluded = normalize_excluded_ids(request.excluded_hpo_ids, known_ids)
        inputs = ft.case_inputs(request.sex, request.age_years, request.onset_years, excluded)
        if ft.has_inputs(inputs):
            score_map = predict_ft(_engine, _ft_model, known_ids, inputs)
            ft_applied = True
            applied_to_excluded = _ft_name if excluded else None
        else:
            score_map = predict_scores(_engine, known_ids)
    else:
        if _reranker is not None and request.excluded_hpo_ids:
            excluded, unknown_excluded = split_excluded_ids(_reranker, request.excluded_hpo_ids, known_ids)
        if excluded:
            score_map = predict_reranked(_engine, _reranker, known_ids, excluded)
            applied_to_excluded = _rerank_name
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
        # Set only when excluded terms reached a re-ranker (rr-v1, or ft-v1.1
        # while it is on): the question agent tells people their "no" answers
        # moved the list only when it is.
        "rerank": applied_to_excluded,
        "ft": _ft_name if ft_applied else None,
        "excluded_hpo_count": len(excluded),
        "unknown_excluded_hpo_ids": unknown_excluded,
        "beta": False,
    }
