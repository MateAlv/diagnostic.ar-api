"""Unhandled exceptions as a JSON 500 that browsers can read."""

from __future__ import annotations

import logging

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class UnhandledErrorsAsJson:
    """Answers an unhandled exception with ``{"detail": {"error": "internal"}}``.

    Starlette answers those from ServerErrorMiddleware, which wraps every user
    middleware, CORS included: the plain-text 500 went out without CORS
    headers and a browser on another origin reported a network failure
    instead. Added before CORSMiddleware, this one runs inside it.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            logger.exception("unhandled error on %s %s", scope.get("method"), scope.get("path"))
            if started:
                raise
            response = JSONResponse({"detail": {"error": "internal"}}, status_code=500)
            await response(scope, receive, send)
