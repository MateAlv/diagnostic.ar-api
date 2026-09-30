"""An unhandled exception reaches a browser as a readable JSON 500."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.errors import UnhandledErrorsAsJson  # noqa: E402

ORIGIN = "https://staging.example.org"


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(UnhandledErrorsAsJson)
    app.add_middleware(CORSMiddleware, allow_origins=[ORIGIN], allow_credentials=True, allow_methods=["*"])

    @app.get("/boom")
    def boom():
        raise RuntimeError("internal detail that must not leak")

    @app.get("/fine")
    def fine():
        return {"ok": True}

    return app


def test_an_unhandled_error_is_a_json_500_with_cors_headers():
    client = TestClient(_app(), raise_server_exceptions=False)
    response = client.get("/boom", headers={"Origin": ORIGIN})
    assert response.status_code == 500
    assert response.json() == {"detail": {"error": "internal"}}
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert "must not leak" not in response.text


def test_a_normal_answer_is_untouched():
    client = TestClient(_app())
    response = client.get("/fine", headers={"Origin": ORIGIN})
    assert response.json() == {"ok": True}
    assert response.headers["access-control-allow-origin"] == ORIGIN


def test_the_service_puts_it_inside_cors():
    source = (Path(__file__).resolve().parent.parent / "app" / "main.py").read_text()
    assert source.index("add_middleware(UnhandledErrorsAsJson)") < source.index("CORSMiddleware,")
