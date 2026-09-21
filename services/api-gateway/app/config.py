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

    # Symptom extraction settings (uses Ollama LLM)
    enable_symptom_extraction: bool = Field(default=True, validation_alias="ENABLE_SYMPTOM_EXTRACTION")
    ollama_url: str = Field(default="http://llm-ollama:11434", validation_alias="OLLAMA_URL")
    symptom_extractor_model: str = Field(default="qwen2.5:7b", validation_alias="SYMPTOM_EXTRACTOR_MODEL")

    # Direct Spanish HPO matching (before translation)
    enable_spanish_hpo_matching: bool = Field(default=True, validation_alias="ENABLE_SPANISH_HPO_MATCHING")

    # Ollama translation settings (for symptoms that don't match Spanish HPO)
    translator_model: str = Field(default="qwen2.5:7b", validation_alias="TRANSLATOR_MODEL")

    audit_database_url: str = Field(
        default="postgresql://diagnostic_audit:diagnostic_audit@audit-db:5432/diagnostic_audit",
        validation_alias="AUDIT_DATABASE_URL",
    )
    store_requests: bool = Field(default=True, validation_alias="STORE_REQUESTS")

    # Access-token verification. The backend signs with the private half of an
    # RSA keypair; only the public half belongs here, so this service can
    # check a token but never issue one.
    require_auth: bool = Field(default=False, validation_alias="REQUIRE_AUTH")
    jwt_public_key_raw: str = Field(default="", validation_alias="JWT_PUBLIC_KEY")

    # Per-IP request budget for /extract-hpo. 0 disables it.
    rate_limit_per_minute: int = Field(default=0, validation_alias="RATE_LIMIT_PER_MINUTE")
    # Addresses whose X-Forwarded-For may be believed, comma separated. Empty
    # means trust none, and the peer address is used instead.
    trusted_proxies: str = Field(default="", validation_alias="TRUSTED_PROXIES")

    # Extractions allowed to run at once. This, not the per-IP limit, is what
    # bounds spend when extraction calls a paid model: it holds however many
    # callers and addresses are involved. 0 disables it.
    max_concurrent_extractions: int = Field(
        default=0, validation_alias="MAX_CONCURRENT_EXTRACTIONS"
    )
    # Callers queued behind that limit before the service sheds load. Waiting
    # forever behind a slow model is worse than a prompt 503.
    max_extraction_queue: int = Field(default=8, validation_alias="MAX_EXTRACTION_QUEUE")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> List[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def jwt_public_key(self) -> str:
        """The PEM public key, with `\\n` escapes expanded.

        A PEM carries newlines and an .env file generally cannot, so the key
        is usually supplied escaped.
        """
        raw = self.jwt_public_key_raw.strip()
        return raw.replace("\\n", "\n") if "\\n" in raw else raw

    @property
    def trusted_proxies_list(self) -> List[str]:
        return [item.strip() for item in self.trusted_proxies.split(",") if item.strip()]


settings = Settings()


def get_settings() -> Settings:
    """Settings accessor, so a test can override the dependency."""
    return settings
