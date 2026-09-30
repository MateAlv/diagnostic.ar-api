from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .audit import AuditLogger, AuditRecord
from .auth import AuthenticatedUser, require_user
from .config import settings
from .errors import UnhandledErrorsAsJson
from .logging import configure_logging
from .pipeline import Pipeline
from .schemas import ExtractRequest, ExtractResponse
from .throttle import (
    ConcurrencyLimiter,
    GlobalExtractionBudget,
    PerIpRateLimiter,
    PerUserRateLimiter,
    client_identifier,
)
from .utils import hash_text

# ---------------------------------------------------------------------------
# Specialist recommendations + confidence, shared with services/genphenai:
# its scripts and PanelApp table. They use bare imports between each other, so
# their directory goes on sys.path once.
# ---------------------------------------------------------------------------
def _find_genphenai_dir() -> Path:
    """services/genphenai, in a checkout or inside an image.

    In a checkout this file is services/api-gateway/app/main.py. The images
    copy services/ to /app/services but this package to /app/app, so walking
    up from here does not reach it there.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "genphenai",
        Path("/app/services/genphenai"),
    ]
    return next((c for c in candidates if c.is_dir()), candidates[0])


_GENPHENIA_DIR = _find_genphenai_dir()

_recommend_specialties = None
_confidence_from_distribution = None
_gene_to_specialties: Optional[Dict[str, Any]] = None


def _init_genphenia() -> None:
    global _recommend_specialties, _confidence_from_distribution, _gene_to_specialties
    genphenia_path = str(_GENPHENIA_DIR)
    if genphenia_path not in sys.path:
        sys.path.insert(0, genphenia_path)
    try:
        from specialist_recommendation import (  # type: ignore
            load_panelapp_gene_to_specialties,
            recommend_specialties,
        )
        from confidence import confidence_from_distribution  # type: ignore

        panelapp_path = _GENPHENIA_DIR / "data" / "panelapp_panels.csv"
        _gene_to_specialties = load_panelapp_gene_to_specialties(panelapp_path)
        _recommend_specialties = recommend_specialties
        _confidence_from_distribution = confidence_from_distribution
        logging.getLogger("api").info(
            "genphenia specialties loaded: %d genes mapped", len(_gene_to_specialties)
        )
    except Exception as exc:
        logging.getLogger("api").warning(
            "genphenia post-processing unavailable: %s", exc
        )

configure_logging(settings.log_level)
logger = logging.getLogger("api")

app = FastAPI(title=settings.app_name)

# Before CORS, so that CORS wraps it: see UnhandledErrorsAsJson.
app.add_middleware(UnhandledErrorsAsJson)
if settings.cors_origins_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

# The incumbent always serves search and the health checks; the flag only
# decides who answers /extract-hpo. Swapping the whole pipeline left search,
# /info, /healthz, startup and the audit record calling attributes the
# recognizer does not have, so the flag could not be turned on at all.
pipeline = Pipeline(settings)
if settings.use_recognizer:
    from .recognizer_pipeline import RecognizerPipeline

    logger.info("extract-hpo is served by the Spanish symptom recognizer")
    extraction = RecognizerPipeline(settings)
else:
    extraction = pipeline

audit_logger = AuditLogger(settings.audit_database_url, settings.store_requests)

# All default to disabled, so none of this changes anything until configured.
extract_rate_limiter = PerIpRateLimiter(settings.rate_limit_per_minute)
user_rate_limiter = PerUserRateLimiter(
    settings.user_extractions_per_hour, settings.user_extractions_per_day
)
extraction_budget = GlobalExtractionBudget(
    settings.global_extractions_per_5h,
    settings.global_extractions_per_week,
    settings.extraction_budget_state,
)
extraction_slots = ConcurrencyLimiter(
    settings.max_concurrent_extractions, settings.max_extraction_queue
)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("warming up pipeline")
    pipeline.phenotyper.ensure_ready()
    audit_logger.ensure_table()
    if extraction is not pipeline:
        # The translator only serves the incumbent's extraction, so it is not
        # warmed here; the recognizer is, and its credentials are checked now
        # rather than on the first clinical history.
        extraction.warm_up()
    else:
        # Pre-load NLLB translator so first request isn't slow
        logger.info("pre-loading translator model...")
        try:
            pipeline.translator.translate("test", src_lang="spa_Latn", tgt_lang="eng_Latn")
            logger.info("translator model ready")
        except Exception as exc:
            logger.warning("translator warmup failed (will retry on first request): %s", exc)
    # Load genphenia specialist/confidence post-processing
    _init_genphenia()


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "symptom_extractor": pipeline.symptom_extractor.model if pipeline.symptom_extractor else "disabled",
        "translator_model": pipeline.translator.model_name,
        "phenotyper": pipeline.phenotyper.version,
        "hpo_index_loaded": pipeline.phenotyper.hpo_index_loaded,
        "extract_hpo": extraction.status() if extraction is not pipeline else "incumbent",
        "extraction_budget": extraction_budget.status(),
    }


@app.get("/info")
async def info():
    """Detailed information about loaded models and configuration."""
    return {
        "app_name": settings.app_name,
        "symptom_extractor": {
            "enabled": settings.enable_symptom_extraction,
            "model": pipeline.symptom_extractor.model if pipeline.symptom_extractor else "none",
            "ollama_url": settings.ollama_url,
        },
        "translator": {
            "model_name": pipeline.translator.model_name,
            # Only the NLLB translator has a device; the Ollama one runs remotely.
            "device": getattr(pipeline.translator, "device", None),
        },
        "phenotyper": {
            "version": pipeline.phenotyper.version,
            "hpo_index_loaded": pipeline.phenotyper.hpo_index_loaded,
            "spacy_model": settings.spacy_model,
            "min_confidence": settings.min_confidence,
            "spanish_hpo_matching": settings.enable_spanish_hpo_matching,
        },
        "cache": {
            "enabled": settings.cache_enabled,
            "ttl_seconds": settings.cache_ttl_seconds,
        },
    }


@app.get("/hpo/es/search")
async def search_hpo_es(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=50)):
    results = pipeline.phenotyper.hpo_index.search_es(q, limit)
    return {"results": results}


@app.post("/extract-hpo", response_model=ExtractResponse)
async def extract_hpo(
    request: ExtractRequest,
    http_request: Request,
    user: AuthenticatedUser | None = Depends(require_user),
):
    # Checked before any work: rejecting an over-budget caller should be
    # cheap, and the cache lookup below is not free.
    extract_rate_limiter.check(
        client_identifier(http_request, settings.trusted_proxies_list)
    )
    user_rate_limiter.check(user.user_id if user else None)

    started = time.time()
    text_hash = hash_text(request.text_es)
    if settings.log_text:
        logger.info("extract requested hash=%s text=%s", text_hash, request.text_es)
    else:
        logger.info("extract requested hash=%s length=%s", text_hash, len(request.text_es))
    normalized_text = extraction.normalize_text(request.text_es, request.patient_locale)
    response = None
    error_message = None
    cache_hit = False
    duration_ms = 0
    try:
        # Held only around the extraction itself. This is what bounds cost
        # when extraction calls a paid model: a per-IP limit is evaded with
        # more addresses, a semaphore is not.
        with extraction_slots:
            # Counted only once a slot is held, right before the model is
            # called: a caller shed by the queue has spent nothing.
            extraction_budget.consume()
            response, meta = extraction.extract_with_meta(
                request.text_es, request.patient_locale, normalized_text
            )
        cache_hit = bool(meta.get("cache_hit"))
        duration_ms = int(meta.get("duration_ms") or 0)
    except HTTPException as exc:
        # Load shedding is a deliberate answer, not a failure: turning a 503
        # into a 500 would tell the caller to stop retrying rather than to
        # wait, and would log it as a fault.
        duration_ms = int((time.time() - started) * 1000)
        error_message = f"{exc.status_code} {exc.detail}"
        raise
    except Exception as exc:
        duration_ms = int((time.time() - started) * 1000)
        error_message = str(exc)
        logger.exception("extract failed hash=%s", text_hash)
        raise HTTPException(status_code=500, detail="extraction-failed") from exc
    finally:
        if settings.store_requests:
            model_info = (response or {}).get("model", {})
            audit_logger.log_request(
                AuditRecord(
                    request_hash=text_hash,
                    patient_locale=request.patient_locale,
                    text_es_raw=request.text_es,
                    text_es_normalized=normalized_text,
                    symptoms_extracted=(response or {}).get("symptoms_extracted", []),
                    phenotypes=(response or {}).get("phenotypes", []),
                    model_symptom_extractor=model_info.get("symptom_extractor", "none"),
                    model_translator=model_info.get("translator", "none"),
                    model_phenotyper=model_info.get("phenotyper", "none"),
                    cache_hit=cache_hit,
                    duration_ms=duration_ms,
                    error=error_message,
                    source_ip=http_request.client.host if http_request.client else None,
                    user_agent=http_request.headers.get("user-agent"),
                )
            )
    logger.info("extract completed hash=%s duration_ms=%s", text_hash, duration_ms)
    return response


class SpecialtiesRequest(BaseModel):
    gene_scores: Dict[str, float]


@app.post("/specialties")
async def get_specialties(request: SpecialtiesRequest) -> Dict[str, Any]:
    """
    Given a mapping of gene_symbol → score (from GenPhenAI /rank-genes),
    returns specialist recommendations and model confidence using the
    genphenia post-processing pipeline.
    """
    if not request.gene_scores:
        raise HTTPException(status_code=422, detail="gene_scores must not be empty")

    if _recommend_specialties is None or _gene_to_specialties is None:
        raise HTTPException(
            status_code=503,
            detail="Specialist recommendation unavailable — genphenia data not loaded",
        )

    try:
        specialties = _recommend_specialties(request.gene_scores, _gene_to_specialties)
        confidence = _confidence_from_distribution(request.gene_scores) if _confidence_from_distribution else None
    except Exception as exc:
        logger.warning("specialties computation failed: %s", exc)
        raise HTTPException(status_code=500, detail="specialties-failed") from exc

    return {"specialties": specialties, "confidence": confidence}
