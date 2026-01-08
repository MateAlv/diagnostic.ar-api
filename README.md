# Diagnostic.ar API 

Open-source pipeline to extract HPO terms from Spanish (es-AR) clinical text with explainable spans and offsets. Runs locally with Docker Compose and uses NLLB-200 on GPU when available.

## Quickstart

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Web UI: `http://localhost:${WEB_PORT}` (default in `.env` is `3001`)

GPU is enabled by default in `docker-compose.yml` for this repo.
Local settings are in `.env` (created for this repo). Adjust if needed.
Set `WEB_PORT` there if `3000` is already in use.

Local dev without Docker:
```bash
pip install -r services/api-gateway/requirements.txt
python -m spacy download en_core_web_sm
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Web dev:
```bash
cd web
npm install
npm run dev
```

To point the web client at a different API, set `VITE_API_URL` at build time (Docker) or in your local `.env` for Vite.

### GPU Notes
- Requires NVIDIA drivers + Container Toolkit.
- GPU mode uses `runtime: nvidia` for compatibility with older Compose builds.
- CPU-only: set `NLLB_DEVICE=cpu` and `NLLB_MODEL_NAME=facebook/nllb-200-distilled-600M`.

## API

### Health
```
GET /healthz
```

### Extract HPO
```
POST /extract-hpo
Content-Type: application/json

{
  "text_es": "...",
  "patient_locale": "es-AR"
}
```

Example:
```bash
curl -s http://localhost:8000/extract-hpo \
  -H 'Content-Type: application/json' \
  -d '{"text_es":"Paciente con fiebre y convulsiones nocturnas.","patient_locale":"es-AR"}' | jq
```

## Output
The response includes:
- `text_en` translation
- model metadata
- `phenotypes[]` with `hpo_id`, label, Spanish/English spans, offsets, negation, confidence, and match method.

## Configuration
See `.env.example` for all env vars.

Key settings:
- `NLLB_MODEL_NAME`: `auto` (default), `facebook/nllb-200-1.3B`, or `facebook/nllb-200-distilled-600M`
- `NLLB_DEVICE`: `auto`, `cuda`, or `cpu`
- `MIN_CONFIDENCE`: filter threshold
- `CACHE_TTL_SECONDS`: Redis TTL
- `ENABLE_SPAN_BACKTRANSLATION`: toggles best-effort ES span alignment

HPO data is downloaded on first startup into `data/hpo/hp.obo`, and an index is built at `data/hpo/hpo_index.json`.

## Why This Approach
- **Transparent**: spans, offsets, and match method are returned.
- **Open-source**: no paid APIs or licensed clinical resources.
- **Modular**: translator and phenotyper are separable and can be replaced.

## How To Plug MedCAT Later
- Swap the phenotyper module with a MedCAT pipeline and map output to the same response schema.
- Keep the translation and normalization stages unchanged.
- Reuse Redis cache and API wrapper.

## Security Defaults
- No raw PHI logging by default (`LOG_TEXT=false`).
- CORS is restricted to configured origins.
- Redis caching is optional and uses TTL.

## Limitations
- ES/EN span alignment is best-effort and approximate.
- HPO matching is baseline string-based; accuracy improves with better NLP components.

## Tests
```bash
pip install -r services/api-gateway/requirements.txt
pip install -r requirements-dev.txt
pytest
```
