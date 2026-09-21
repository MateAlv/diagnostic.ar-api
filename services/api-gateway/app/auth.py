"""Verify access tokens issued by the diagnostic.ar backend.

The backend signs with an RSA private key and this service holds only the
public half, so it can check a token but never mint one. That matters because
this runs on brainy: a symmetric secret here would let the GPU node issue a
token for any user, including an admin.

What this cannot do is revoke. The backend re-reads `token_version` from the
database on sensitive routes, so a logged-out or demoted user loses access
immediately there. There is no database here, so a stolen token stays valid
until it expires -- which is why access tokens are short-lived.
"""
from __future__ import annotations

import logging

import jwt
from fastapi import Depends, HTTPException, Request, status

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

ACCESS_TOKEN_ALGORITHM = "RS256"
ACCESS_TOKEN_COOKIE = "access_token"


class AuthenticatedUser:
    """The claims the backend puts in an access token."""

    __slots__ = ("user_id", "role")

    def __init__(self, user_id: str, role: str | None):
        self.user_id = user_id
        self.role = role

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"AuthenticatedUser(user_id={self.user_id!r}, role={self.role!r})"


def extract_token(request: Request) -> str | None:
    """Read the token from the Authorization header, else the cookie.

    Only the Bearer scheme, matching the backend. Splitting on whitespace
    would accept any scheme's credential, and staging sits behind a proxy that
    may set `Authorization: Basic ...` of its own -- that header is the
    proxy's, and carries no claim about who the caller is.
    """
    header = request.headers.get("authorization") or ""
    scheme, _, credential = header.partition(" ")
    if scheme.lower() == "bearer" and credential.strip():
        return credential.strip()

    cookie = request.cookies.get(ACCESS_TOKEN_COOKIE)
    return cookie.strip() if cookie else None


def decode_access_token(token: str, public_key: str) -> AuthenticatedUser:
    """Verify a token and return its claims, or raise HTTPException(401)."""
    try:
        claims = jwt.decode(
            token,
            public_key,
            # Pinned, never taken from the token's own header. A verifier that
            # honours `alg` accepts "none", and accepts an HS256 token signed
            # with this public key as if it were a shared secret -- and the
            # public key is, by design, not a secret.
            algorithms=[ACCESS_TOKEN_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
        )
    except jwt.InvalidTokenError:
        # Deliberately not logged with the token or the reason: a rejected
        # token is still a credential, and the distinction between "bad
        # signature" and "malformed" is useful to an attacker.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    subject = claims.get("sub")
    if not subject or not isinstance(subject, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    role = claims.get("role")
    return AuthenticatedUser(user_id=subject, role=role if isinstance(role, str) else None)


def require_user(
    request: Request, settings: Settings = Depends(get_settings)
) -> AuthenticatedUser | None:
    """FastAPI dependency enforcing a valid access token.

    Returns None when authentication is disabled, which is how the endpoint
    behaves today and what lets this land before the frontend sends a token.
    """
    if not settings.require_auth:
        return None

    if not settings.jwt_public_key:
        # Refusing is the safe failure: starting unauthenticated because a key
        # is missing is how an endpoint quietly ends up open.
        logger.error("REQUIRE_AUTH is set but JWT_PUBLIC_KEY is empty")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is misconfigured",
        )

    token = extract_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )

    return decode_access_token(token, settings.jwt_public_key)
