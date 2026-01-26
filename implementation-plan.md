# Diagnostic.ar API Implementation Plan

Goal: build an open-source, $0-API-cost pipeline that converts Spanish (es-AR) clinical text into HPO terms with explainable spans and offsets, running locally with Docker Compose and GPU-accelerated translation when available.

## Scope
- Implement a new repo from scratch under `diagnostic.ar-api`.
- Provide FastAPI endpoint `POST /extract-hpo` and `GET /healthz`.
- Use NLLB-200 translation on GPU when available, with a smaller fallback model.
- Build a phenotyper service that maps English spans to HPO IDs with explainable spans/offsets, negation, and confidence.
- Provide a minimal web UI to call the API and show removable HPO chips.
- Add Redis caching, normalization rules, and tests.

## Non-Goals
- Clinical-grade accuracy or FDA-grade validation.
- UMLS-licensed tools (e.g., MedCAT with UMLS).
- End-to-end EHR integration or authentication/authorization.

## Architecture (High-Level)
1) **Normalize es-AR text** using rule-based replacements (configurable).
2) **Translate es->en** with NLLB (GPU if available).
3) **Extract candidate spans** in English via spaCy/scispaCy.
4) **Map spans to HPO terms** with exact/synonym/fuzzy matching.
5) **Detect negation** (NegEx-like window rules).
6) **Return structured results** with spans, offsets, confidence, and match method.

## Repository Structure
- `/services/api-gateway` (FastAPI service, orchestration)
- `/services/translator` (NLLB module, GPU-aware)
- `/services/phenotyper` (HPO loading/index, matcher, negation, scoring)
- `/web` (minimal React UI)
- `/data` (HPO download cache + normalization rules)
- `/scripts` (HPO download/build index)
- `/tests` (pytest)

## Data Flow
`text_es` -> normalize -> hash -> Redis cache lookup
  - hit: return cached result
  - miss: translate -> extract -> map -> negation -> score -> cache -> return

## Key Design Choices
- **Translation**: HuggingFace Transformers + PyTorch.
  - Primary model: `facebook/nllb-200-1.3B`
  - Fallback: `facebook/nllb-200-distilled-600M`
  - Auto-select with GPU detection and explicit env override.
- **Phenotyping**:
  - Load HPO .obo or JSON and build an index of labels + synonyms.
  - Use spaCy/scispaCy to propose candidate spans in EN.
  - Use exact/synonym/fuzzy (rapidfuzz) match to HPO.
  - Negation detection with a simple EN NegEx-like window.
  - Confidence heuristic: exact=0.95, synonym=0.85, fuzzy=scaled.
- **Caching**: Redis; key = SHA256(normalized text + locale + model id).
- **Privacy**: Do not log raw text by default; log only request hash and timing.

## Docs
- README with quickstart, GPU notes, example curl, screenshots (if possible).
- "Why this approach" and "How to plug MedCAT later" sections.
- Limitations: span alignment and model accuracy.

## API Contract
`POST /extract-hpo`
```
{ "text_es": "...", "patient_locale": "es-AR" }
```
Response includes:
- `text_en`, model metadata, and `phenotypes[]`
- each phenotype: `hpo_id`, `label`, `span_es`, `span_en`, offsets, negation, confidence, match type.

## Normalization
- Config file at `data/normalization/es_ar.yml` or `.json`.
- Apply ordered regex rules (e.g., "me duele mal" -> "me duele mucho").
- Keep normalized and original text for offsets.

## HPO Data
- Script to download HPO OBO from official GitHub.
- Parse terms, labels, synonyms.
- Store serialized index in `data/hpo_index.json` for fast startup.
- API startup should download/build if files are missing.

## Offset Strategy
- Track EN offsets directly from translation output.
- For ES offsets: map candidate ES spans by fuzzy search of translated spans, or by direct matching in original text when the normalized term exists.
- Document limitations in README.

## Observability
- `GET /healthz` returns model and data readiness.
- Structured logs for timings and cache hits.

## Docker & Compose
- Python 3.11 base image for API.
- Compose services: `api`, `redis`, `web`.
- GPU support via `device_requests` (NVIDIA runtime).
- Env vars: model size, confidence threshold, cache TTL, cache disable flag.

## Testing
- Unit tests for normalization rules.
- HPO parsing/indexing tests.
- Deterministic extraction test using a fixed EN translation fixture.

## Milestones
1) Scaffold repo, Dockerfiles, compose, CI-ready layout.
2) Implement HPO downloader + indexer + tests.
3) Implement normalization + tests.
4) Implement translator module with GPU auto-detect + tests (mocked).
5) Implement phenotyper matcher + negation + scoring + tests.
6) FastAPI endpoint + Redis cache.
7) Minimal React UI + integration with endpoint.
8) README + quickstart + example curl + limitations.

## Risks / Trade-offs
- NLLB 1.3B may not fit typical GPUs; fallback is required and should be configurable.
- Full HPO synonym phrase matching may be heavy; consider pre-normalized keys and limiting synonyms if needed.
- Span alignment between ES and EN is approximate; document this clearly.
- scispaCy models are large; consider smaller spaCy model if container size becomes an issue.

## Open Questions
- Expected GPU memory on target hosts?
- Minimum acceptable latency for a single request?
- Should we add a `/csrf` endpoint if the UI is served from a different domain?
- Should we include a lightweight admin endpoint for reloading normalization rules?

---

# Add-On Plan: Extraction Audit Log + DB Storage

Goal: persist every extraction request with raw ES text, normalized text, EN translation, and extracted HPOs, so we can audit outcomes and improve quality.

## Scope
- Store every `/extract-hpo` request in a database with full inputs/outputs.
- Provide a secure, read-only API for reviewing these records.
- Avoid logging raw PHI to stdout; keep it only in DB.
- Add retention & redaction controls for privacy.

## Data Model (Postgres)
Table: `extraction_requests`
- `id` (uuid, pk)
- `created_at` (timestamptz, default now)
- `request_hash` (sha256 of input)
- `patient_locale` (text)
- `text_es_raw` (text)
- `text_es_normalized` (text)
- `text_en` (text)
- `phenotypes_json` (jsonb)
- `model_translation` (text)
- `model_phenotyper` (text)
- `cache_hit` (bool)
- `duration_ms` (int)
- `error` (text, nullable)
- `source_ip` (text, nullable)
- `user_agent` (text, nullable)

Optional: `redacted` (bool) + `text_es_redacted` for future PHI masking.

## API Surface (Secure)
Add admin endpoints with token auth (env `ADMIN_API_TOKEN`):
- `GET /admin/extractions?limit=50&offset=0`
- `GET /admin/extractions/{id}`
- `GET /admin/extractions/{id}/export` (json)

No raw-text logging to stdout; only request hash + timings.

## Config Flags (env)
- `STORE_REQUESTS=true|false` (default true)
- `ADMIN_API_TOKEN=...` (required for admin endpoints)
- `RETENTION_DAYS=...` (optional cleanup job)
- `LOG_TEXT=false` (keep default false)

## Implementation Steps
1) Add Postgres service to `diagnostic.ar-api/docker-compose.yml`.
2) Add SQLAlchemy/SQLModel + migrations (Alembic) in api-gateway.
3) Implement persistence in `Pipeline.extract()` or `POST /extract-hpo` handler:
   - write record on success + on error (with error field).
4) Add admin routes with token middleware.
5) Add retention script (cron/manual) to purge old rows.
6) Add tests:
   - DB insert on success
   - DB insert on error
   - admin endpoints auth

## Security Notes
- This stores PHI; keep DB local only and disable public exposure.
- Do not log raw text in API logs.
- Use least-privileged DB user.
- Consider disk encryption if running on shared hosts.

## Open Questions (for your review)
- Should we store *all* raw input, or only a redacted copy by default?
- Do you want a basic UI page for browsing records, or API-only?
