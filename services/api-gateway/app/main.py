from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .logging import configure_logging
from .pipeline import Pipeline
from .schemas import ExtractRequest, ExtractResponse
from .utils import hash_text

configure_logging(settings.log_level)
logger = logging.getLogger("api")

app = FastAPI(title=settings.app_name)

if settings.cors_origins_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

pipeline = Pipeline(settings)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("warming up pipeline")
    pipeline.phenotyper.ensure_ready()


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "translator_model": pipeline.translator.model_name,
        "phenotyper": pipeline.phenotyper.version,
        "hpo_index_loaded": pipeline.phenotyper.hpo_index_loaded,
    }


@app.post("/extract-hpo", response_model=ExtractResponse)
async def extract_hpo(request: ExtractRequest):
    started = time.time()
    text_hash = hash_text(request.text_es)
    if settings.log_text:
        logger.info("extract requested hash=%s text=%s", text_hash, request.text_es)
    else:
        logger.info("extract requested hash=%s length=%s", text_hash, len(request.text_es))
    try:
        response = pipeline.extract(request.text_es, request.patient_locale)
    except Exception as exc:
        logger.exception("extract failed hash=%s", text_hash)
        raise HTTPException(status_code=500, detail="extraction-failed") from exc
    duration_ms = int((time.time() - started) * 1000)
    logger.info("extract completed hash=%s duration_ms=%s", text_hash, duration_ms)
    return response
