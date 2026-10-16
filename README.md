# Diagnostic.ar API 

Open-source pipeline to extract HPO terms from Spanish (es-AR) clinical text with explainable spans and offsets. Runs locally with Docker Compose and uses NLLB-200 on GPU when available.

## Quickstart

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Web UI: `http://localhost:${WEB_PORT}` (default in `.env` is `3001`)
- GenPhenAI gateway: `http://localhost:18020`

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

### Rank Genes (GenPhenia inference-backed)
```
POST /rank-genes
Content-Type: application/json

{
  "hpo_ids": ["HP:0001530", "HP:0000938"],
  "top_k": 10,
  "sex": "female",
  "age_years": 7.5,
  "onset_years": 2.0
}
```

The `genphenai` service runs the GenPhenia v4 network (`model-v4.0.0`) in
process, on the CPU, followed by v4-layer, and returns the schema the frontend
already consumes (`results`, `specialties`, `confidence`). `sex`, `age_years`
and `onset_years` are optional; v4-layer reads them, and without them the
ranking is the network's own. HPO IDs the model's release does not know are
left out and listed in `unknown_hpo_ids`; a request with none it knows is a
422.

The model and the layer are releases of `diagnostic.ar-model`, pinned in
`services/genphenai/model.lock` (see `services/genphenai/model/README.md`).
Fetch them before building or testing; the tests compare the rankings with the
release's own fixture:

```bash
services/genphenai/fetch_model.sh
cd services/genphenai && python -m pytest tests -q
```

### HPO embeddings of the symptom extractor

`Dockerfile.recognizer` does not compute the embeddings of the HPO terms: they
are a release of `diagnostic.ar-model` too (`hpo-embeddings-vX.Y.Z`, generated
there from `hpo-embeddings/embeddings.lock`), pinned by tag in
`services/api-gateway/embeddings.lock` and checked against the sha256 the
release publishes. Fetch them before building the
extractor; the build verifies them (`services/api-gateway/check_embeddings.py`)
and refuses a package made with another encoder, other labels or another
generator script:

```bash
services/api-gateway/fetch_embeddings.sh     # -> embeddings-dist/
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
- `GENPHENIA_BUNDLE_DIR`: the model bundle (default: the one baked into the image)
- `GENPHENIA_DATA_DIR`: where `panelapp_panels.csv` is (default: baked in too)

`data/hpo/hp.obo` is the HPO release `v2026-01-08`, committed; if it is missing, startup downloads that same release (`HPO_DOWNLOAD_URL`), not whatever `master` holds that day. The index is built at `data/hpo/hpo_index.json`.

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
