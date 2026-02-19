FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/genphenai/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY services/genphenai/app /app/app

# Post-processing scripts (copied from diagnosticar_tools)
COPY services/genphenai/confidence.py /app/confidence.py
COPY services/genphenai/specialist_recommendation.py /app/specialist_recommendation.py

ENV PYTHONPATH=/app

EXPOSE 18020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "18020"]
