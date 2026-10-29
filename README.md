# diagnostic.ar-api

Los servicios de Python que usa [diagnostic.ar](https://github.com/MateAlv/diagnosticar-plataforma) detrás de la web:

- **api-gateway**: recibe los síntomas escritos por la persona y los pasa a términos HPO. Pide iniciar sesión y pone un tope de extracciones por cuenta. El reconocedor de síntomas viene de [spanish-symptoms-recognizer](https://github.com/GMBUST-FIUBA/spanish-symptoms-recognizer).
- **genphenai**: corre el modelo que ordena los genes. El modelo no está acá: se descarga de las versiones de [diagnostic.ar-model](https://github.com/MateAlv/diagnostic.ar-model).
- **question-agent**: arma la imagen del [mecanismo de repreguntas](https://github.com/MarBL00126/diagnosticar-question-agent).

Es parte del Trabajo Profesional de Ingeniería en Informática (FIUBA) de Mateo Alvarez, Mariano Brizuela López, Gonzalo Bustamante y Lucía Lourengo Caridade.

## Probarlo

```bash
docker compose up --build
```

La API queda en http://localhost:8000.

## Licencia

GPL-3.0. La carpeta `services/genphenai/inference_engine/` es el motor de GenPhenia, de Gianlucca Colangelo, y no está bajo esta licencia.
