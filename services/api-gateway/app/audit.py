from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import psycopg
from psycopg.types.json import Json

logger = logging.getLogger("audit")


@dataclass
class AuditRecord:
    request_hash: str
    patient_locale: str
    text_es_raw: str
    text_es_normalized: str
    symptoms_extracted: Iterable[str]
    phenotypes: Iterable[dict]
    model_symptom_extractor: str
    model_translator: str
    model_phenotyper: str
    cache_hit: bool
    duration_ms: int
    error: Optional[str]
    source_ip: Optional[str]
    user_agent: Optional[str]


class AuditLogger:
    def __init__(self, dsn: str, enabled: bool = True) -> None:
        self.dsn = dsn
        self.enabled = enabled

    def ensure_table(self) -> None:
        if not self.enabled:
            return
        try:
            with psycopg.connect(self.dsn) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS extraction_requests (
                        id uuid PRIMARY KEY,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        request_hash text NOT NULL,
                        patient_locale text,
                        text_es_raw text,
                        text_es_normalized text,
                        symptoms_extracted jsonb,
                        phenotypes_json jsonb,
                        model_symptom_extractor text,
                        model_translator text,
                        model_phenotyper text,
                        cache_hit boolean,
                        duration_ms integer,
                        error text,
                        source_ip text,
                        user_agent text
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS extraction_requests_created_at_idx
                    ON extraction_requests (created_at DESC);
                    """
                )
                conn.commit()
        except Exception as exc:
            logger.error("audit table init failed: %s", exc)

    def log_request(self, record: AuditRecord) -> None:
        if not self.enabled:
            return
        try:
            with psycopg.connect(self.dsn) as conn:
                conn.execute(
                    """
                    INSERT INTO extraction_requests (
                        id,
                        request_hash,
                        patient_locale,
                        text_es_raw,
                        text_es_normalized,
                        symptoms_extracted,
                        phenotypes_json,
                        model_symptom_extractor,
                        model_translator,
                        model_phenotyper,
                        cache_hit,
                        duration_ms,
                        error,
                        source_ip,
                        user_agent
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    );
                    """,
                    (
                        str(uuid.uuid4()),
                        record.request_hash,
                        record.patient_locale,
                        record.text_es_raw,
                        record.text_es_normalized,
                        Json(list(record.symptoms_extracted)),
                        Json(list(record.phenotypes)),
                        record.model_symptom_extractor,
                        record.model_translator,
                        record.model_phenotyper,
                        record.cache_hit,
                        record.duration_ms,
                        record.error,
                        record.source_ip,
                        record.user_agent,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error("audit log insert failed: %s", exc)
