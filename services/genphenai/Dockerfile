# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The published base image lags on Debian security fixes.
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/genphenai/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

RUN useradd --create-home --uid 10001 app

# The model bundle first: it is the large, rarely changing layer. It is a
# release of diagnostic.ar-model, fetched by fetch_model.sh; only the tarball
# pinned in model.lock gets in, and so does its re-ranker when one is pinned.
RUN --mount=type=bind,source=services/genphenai/model.lock,target=/tmp/model.lock \
    --mount=type=bind,source=services/genphenai/model-dist,target=/tmp/model-dist \
    . /tmp/model.lock \
    && echo "$MODEL_SHA256  /tmp/model-dist/$MODEL_ASSET" | sha256sum -c \
    && mkdir -p /app/model \
    && tar -xzf "/tmp/model-dist/$MODEL_ASSET" -C /app/model \
    && if [ -n "${RERANK_TAG:-}" ]; then \
        echo "$RERANK_SHA256  /tmp/model-dist/$RERANK_ASSET" | sha256sum -c \
        && tar -xzf "/tmp/model-dist/$RERANK_ASSET" -C /app/model; \
    fi \
    && chown -R app /app/model
COPY --chown=app services/genphenai/data /app/data
COPY --chown=app services/genphenai/inference_engine /app/inference_engine
COPY --chown=app services/genphenai/rerank /app/rerank
# Read at startup: when it pins a re-ranker, the service needs it to start.
COPY --chown=app services/genphenai/model.lock /app/model.lock
COPY --chown=app services/genphenai/confidence.py services/genphenai/specialist_recommendation.py /app/
COPY --chown=app services/genphenai/app /app/app

ENV PYTHONPATH=/app \
    GENPHENIA_BUNDLE_DIR=/app/model/genphenia-v3.0 \
    GENPHENIA_DATA_DIR=/app/data \
    GENPHENIA_MODEL_LOCK=/app/model.lock \
    RERANK_BUNDLE_DIR=/app/model/rerank-v1.0

USER app

# Loading the bundles once writes the checkpoint registry, so startup does not
# hash the checkpoint again, and a broken bundle fails the build, not a request.
RUN python -c "from app.main import startup; startup()"

EXPOSE 18020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "18020"]
