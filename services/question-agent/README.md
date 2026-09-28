# Agente de preguntas

Receta de la imagen `diagnosticar-question-agent`: el agente de preguntas de
seguimiento de diagnostic.ar. Recibe los hallazgos HPO de un diagnóstico, pide
el ranking de genes al modelo de diagnostic.ar y propone preguntas de sí/no que
lo refinan. Sirve `POST /v1/rounds` y `GET /healthz` en el puerto 8010.
Sugiere preguntas para conversar con un médico; no diagnostica.

## De dónde sale el código

El código vive en `MarBL00126/diagnosticar-question-agent` (privado). Acá está
solo el `Dockerfile`, que lo recibe como un contexto de build adicional llamado
`agent`, igual que `services/api-gateway/Dockerfile.recognizer` con el
reconocedor de síntomas. El commit que se publica está fijado en el repositorio
de la plataforma, en `.github/question-agent.env` (`QUESTION_AGENT_COMMIT`).

La imagen reconstruye la base de conocimiento desde las fuentes HPO fijadas
(cada una verificada por sha256), trae el CLI de Claude fijado
(`@anthropic-ai/claude-code` 2.1.270, sin Node ni npm), deja la base como
archivo de solo lectura y corre como el usuario `app` (uid 10001).

## Construir y probar

Desde la raíz de este repositorio, con el agente clonado al lado:

```sh
docker buildx build -f services/question-agent/Dockerfile \
  --build-context agent=../qa-agente-claude \
  -t diagnosticar-question-agent:dev .
```

Los tests corren dentro de la imagen, como en el release; los `live` llaman a
un LLM de verdad y quedan afuera:

```sh
docker run --rm -e HOME=/tmp -v "$PWD/../qa-agente-claude/tests:/tmp/tests:ro" \
  diagnosticar-question-agent:dev \
  sh -c "pip install -q --user pytest==8.3.2 && python -m pytest /tmp/tests -q -p no:cacheprovider -m 'not live'"
```

Montando además `../qa-agente-claude/docs` en `/tmp/docs:ro` corren también los
tests que validan el ejemplo del contrato.

## Correr

Contra un servicio del modelo de diagnostic.ar que escucha en el host:

```sh
TOKEN=$(openssl rand -hex 32)
docker run -d --name question-agent --network host \
  -e PREGUNTAS_MODEL_SERVICE_URL=http://127.0.0.1:18021 \
  -e PREGUNTAS_LLM_PROVIDER=template \
  -e PREGUNTAS_SERVICE_TOKEN="$TOKEN" \
  diagnosticar-question-agent:dev
curl -s http://127.0.0.1:8010/healthz
curl -s -X POST http://127.0.0.1:8010/v1/rounds \
  -H "X-Internal-Token: $TOKEN" -H 'Content-Type: application/json' \
  -d '{"mode":"paciente","round":1,"present_hpo_ids":["HP:0002205","HP:0004322"],"absent_hpo_ids":["HP:0001250"],"answers":[],"finish":false}'
docker rm -f question-agent
```

`template` no usa ningún LLM. Con `claude-cli` hace falta
`CLAUDE_CODE_OAUTH_TOKEN`; con `anthropic`, `ANTHROPIC_API_KEY`. Sin token de
servicio toda ronda responde 401.

Con compose, junto al modelo de este repositorio:

```sh
docker compose --profile question-agent up -d question-agent
docker compose ps question-agent
```

Toma el agente de `../qa-agente-claude` (o de `QUESTION_AGENT_SRC`), usa
`template` salvo que `PREGUNTAS_LLM_PROVIDER` diga otra cosa y el token de
`QUESTION_AGENT_TOKEN`. No publica puertos: en la plataforma solo lo llama el
backend.

Todas las variables `PREGUNTAS_*` están en el README del agente, sección
[Configuración](https://github.com/MarBL00126/diagnosticar-question-agent#configuración).

## Publicación

La imagen la publica el workflow de release de diagnostic.ar
(`.github/workflows/release.yml`, job `publish-question-agent`): construye,
corre los tests dentro de la imagen, la escanea con Trivy (HIGH y CRITICAL),
verifica que no corra como root y la sube a GHCR.
