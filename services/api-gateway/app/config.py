from __future__ import annotations

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "diagnostic-ar-api"
    log_level: str = "INFO"
    log_text: bool = False

    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        validation_alias="CORS_ORIGINS",
    )

    redis_url: str = Field(default="redis://redis:6379/0", validation_alias="REDIS_URL")
    cache_enabled: bool = Field(default=True, validation_alias="CACHE_ENABLED")
    cache_ttl_seconds: int = Field(default=3600, validation_alias="CACHE_TTL_SECONDS")

    normalization_rules_path: str = Field(
        default="/app/data/normalization/es_ar.yml",
        validation_alias="NORMALIZATION_RULES_PATH",
    )

    hpo_obo_path: str = Field(
        default="/app/data/hpo/hp.obo",
        validation_alias="HPO_OBO_PATH",
    )
    hpo_index_path: str = Field(
        default="/app/data/hpo/hpo_index.json",
        validation_alias="HPO_INDEX_PATH",
    )
    hpo_es_path: str = Field(
        default="/app/data/hpo/hp-es.json",
        validation_alias="HPO_ES_PATH",
    )
    hpo_download_url: str = Field(
        default="https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo",
        validation_alias="HPO_DOWNLOAD_URL",
    )

    min_confidence: float = Field(default=0.6, validation_alias="MIN_CONFIDENCE")
    spacy_model: str = Field(default="en_core_web_sm", validation_alias="SPACY_MODEL")

    nllb_model_name: str = Field(default="auto", validation_alias="NLLB_MODEL_NAME")
    nllb_device: str = Field(default="auto", validation_alias="NLLB_DEVICE")
    nllb_max_length: int = Field(default=512, validation_alias="NLLB_MAX_LENGTH")

    enable_span_backtranslation: bool = Field(
        default=True, validation_alias="ENABLE_SPAN_BACKTRANSLATION"
    )

    audit_database_url: str = Field(
        default="postgresql://diagnostic_audit:diagnostic_audit@audit-db:5432/diagnostic_audit",
        validation_alias="AUDIT_DATABASE_URL",
    )
    store_requests: bool = Field(default=True, validation_alias="STORE_REQUESTS")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> List[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


settings = Settings()
