# diagnostic.ar-api

Los servicios de Python que usa [diagnostic.ar](https://github.com/MateAlv/diagnosticar-plataforma) detrás de la web:

- **api-gateway**: recibe los síntomas escritos por la persona y los pasa a términos HPO. Pide iniciar sesión y pone un tope de extracciones por cuenta. El reconocedor de síntomas viene de [spanish-symptoms-recognizer](https://github.com/GMBUST-FIUBA/spanish-symptoms-recognizer).
- **genphenai**: corre el modelo que ordena los genes. El modelo no está acá: se descarga de las versiones de [diagnostic.ar-model](https://github.com/MateAlv/diagnostic.ar-model).
- **question-agent**: arma la imagen del [mecanismo de repreguntas](https://github.com/MarBL00126/diagnosticar-question-agent).

Es parte del Trabajo Profesional de Ingeniería en Informática (FIUBA) de Mateo Alvarez, Mariano Brizuela López, Gonzalo Bustamante y Lucía Lourengo Caridade.

## Cómo corre en producción

Las imágenes que corren en staging, producción y la copia en AWS no salen del `docker-compose.yml` de este repo. Las arma el pipeline de [diagnosticar-plataforma](https://github.com/MateAlv/diagnosticar-plataforma), que fija el commit de este repo, y las levanta el `docker-compose.prod.yml` de la plataforma. Ahí el api-gateway se construye con `services/api-gateway/Dockerfile.recognizer` y corre con `USE_RECOGNIZER=true`, que extrae los síntomas con Claude (`RECOGNIZER_PROVIDER=claude-cli`). Las imágenes parten de `python:3.12-slim` y corren en CPU.

## Probarlo en local

El `docker-compose.yml` de este repo levanta el extractor original, que usa Ollama (`qwen2.5:7b`) y la imagen con CUDA de `services/api-gateway/Dockerfile`. Quedó del primer despliegue, en el nodo brainy, y por eso `.env.example` todavía trae variables de GPU, de Ollama y de NLLB. El modelo no está en el repo, así que hay que bajarlo antes:

```bash
cp .env.example .env                 # completar los valores marcados CHANGE_ME
services/genphenai/fetch_model.sh    # baja model-v4.0.0 de diagnostic.ar-model (usa el CLI gh con sesión iniciada)
docker compose up --build
```

La API queda en http://localhost:8000 y el modelo en http://localhost:18020. Para extraer síntomas hace falta un servidor de Ollama en `OLLAMA_URL`.

## Licencia

GPL-3.0. La carpeta `services/genphenai/inference_engine/` es el motor de GenPhenia, de Gianlucca Colangelo, y no está bajo esta licencia.
