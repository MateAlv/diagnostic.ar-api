"""Tests for access-token verification.

The token shape mirrors what the diagnostic.ar backend signs: sub, role,
tokenVersion, RS256.
"""
from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from app.auth import decode_access_token, extract_token


def make_keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


PRIVATE_KEY, PUBLIC_KEY = make_keypair()


def issue(claims=None, key=PRIVATE_KEY, algorithm="RS256", **kwargs):
    payload = {"sub": "user-123", "role": "user", "tokenVersion": 1}
    payload.update(claims or {})
    payload.setdefault("exp", int(time.time()) + 900)
    return jwt.encode(payload, key, algorithm=algorithm, **kwargs)


class FakeRequest:
    def __init__(self, headers=None, cookies=None):
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.cookies = cookies or {}


class TestExtractToken:
    def test_bearer_header(self):
        assert extract_token(FakeRequest({"Authorization": "Bearer abc.def.ghi"})) == "abc.def.ghi"

    def test_scheme_is_case_insensitive(self):
        assert extract_token(FakeRequest({"Authorization": "bearer tok"})) == "tok"

    def test_cookie_is_used_when_no_header(self):
        assert extract_token(FakeRequest(cookies={"access_token": "tok"})) == "tok"

    def test_header_wins_over_cookie(self):
        request = FakeRequest({"Authorization": "Bearer from-header"}, {"access_token": "from-cookie"})
        assert extract_token(request) == "from-header"

    def test_other_schemes_are_ignored(self):
        # Staging sits behind a proxy that may set Basic auth of its own; that
        # header is the proxy's and says nothing about the caller.
        assert extract_token(FakeRequest({"Authorization": "Basic dXNlcjpwYXNz"})) is None

    def test_no_credentials_is_none(self):
        assert extract_token(FakeRequest()) is None
        assert extract_token(FakeRequest({"Authorization": "Bearer   "})) is None


class TestDecodeAccessToken:
    def test_valid_token_yields_its_claims(self):
        user = decode_access_token(issue(), PUBLIC_KEY)
        assert user.user_id == "user-123"
        assert user.role == "user"

    def test_public_key_alone_suffices(self):
        # The point of RS256 here: this service verifies without any secret.
        assert decode_access_token(issue(), PUBLIC_KEY).user_id == "user-123"

    def test_token_from_another_keypair_is_rejected(self):
        other_private, _ = make_keypair()
        with pytest.raises(HTTPException) as excinfo:
            decode_access_token(issue(key=other_private), PUBLIC_KEY)
        assert excinfo.value.status_code == 401

    def test_alg_none_is_rejected(self):
        forged = jwt.encode({"sub": "attacker", "role": "admin"}, None, algorithm="none")
        with pytest.raises(HTTPException):
            decode_access_token(forged, PUBLIC_KEY)

    def test_hs256_signed_with_the_public_key_is_rejected(self):
        # Algorithm confusion: the public key is public, so a verifier that
        # honoured the token's own alg would accept this as an HMAC secret.
        try:
            forged = jwt.encode({"sub": "attacker", "role": "admin"}, PUBLIC_KEY, algorithm="HS256")
        except Exception:
            # PyJWT refuses to build it at all, which is the same protection.
            return
        with pytest.raises(HTTPException):
            decode_access_token(forged, PUBLIC_KEY)

    def test_expired_token_is_rejected(self):
        with pytest.raises(HTTPException) as excinfo:
            decode_access_token(issue({"exp": int(time.time()) - 5}), PUBLIC_KEY)
        assert excinfo.value.status_code == 401

    def test_tampered_payload_is_rejected(self):
        header, _, signature = issue().split(".")
        import base64, json
        elevated = base64.urlsafe_b64encode(
            json.dumps({"sub": "user-123", "role": "admin"}).encode()
        ).rstrip(b"=").decode()
        with pytest.raises(HTTPException):
            decode_access_token(f"{header}.{elevated}.{signature}", PUBLIC_KEY)

    @pytest.mark.parametrize("claims", [{"sub": ""}, {"sub": None}, {"sub": 12345}])
    def test_token_without_a_usable_subject_is_rejected(self, claims):
        with pytest.raises(HTTPException):
            decode_access_token(issue(claims), PUBLIC_KEY)

    def test_missing_role_is_tolerated(self):
        # Role is read from the database for anything that depends on it, so
        # its absence here is not a reason to reject the caller.
        assert decode_access_token(issue({"role": None}), PUBLIC_KEY).role is None

    def test_garbage_is_rejected(self):
        with pytest.raises(HTTPException):
            decode_access_token("not-a-token", PUBLIC_KEY)

    def test_rejection_does_not_echo_the_token(self):
        try:
            decode_access_token(issue(key=make_keypair()[0]), PUBLIC_KEY)
        except HTTPException as exc:
            assert "eyJ" not in str(exc.detail)
