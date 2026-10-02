import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from ai_lib.security import decode_user_id
from ai_lib.security import jwt as jwt_module

ISSUER = "solaria-auth"
JWKS_URL = "https://auth.test/.well-known/jwks.json"


@pytest.fixture
def key_pair(monkeypatch):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )

    class FakeJwkClient:
        def get_signing_key_from_jwt(self, token):
            class Key:
                key = private.public_key()

            return Key()

    monkeypatch.setattr(jwt_module, "_jwk_client", lambda url: FakeJwkClient())
    return pem


def token(pem, **claims):
    payload = {"sub": "user-1", "iss": ISSUER, "exp": int(time.time()) + 60, **claims}
    return pyjwt.encode(payload, pem, algorithm="RS256")


def test_returns_subject_for_valid_token(key_pair):
    assert decode_user_id(token(key_pair), jwks_url=JWKS_URL, issuer=ISSUER) == "user-1"


def test_wrong_issuer_returns_none(key_pair):
    assert decode_user_id(token(key_pair, iss="other"), jwks_url=JWKS_URL, issuer=ISSUER) is None


def test_expired_token_returns_none(key_pair):
    expired = token(key_pair, exp=int(time.time()) - 10)
    assert decode_user_id(expired, jwks_url=JWKS_URL, issuer=ISSUER) is None


def test_missing_subject_returns_none(key_pair):
    payload = {"iss": ISSUER, "exp": int(time.time()) + 60}
    encoded = pyjwt.encode(payload, key_pair, algorithm="RS256")
    assert decode_user_id(encoded, jwks_url=JWKS_URL, issuer=ISSUER) is None


def test_garbage_and_unreachable_jwks_return_none(monkeypatch):
    monkeypatch.setattr(jwt_module, "_clients", {})
    assert decode_user_id("not-a-jwt", jwks_url="http://127.0.0.1:9/jwks", issuer=ISSUER) is None


def test_jwk_clients_are_cached_per_url(monkeypatch):
    monkeypatch.setattr(jwt_module, "_clients", {})
    first = jwt_module._jwk_client(JWKS_URL)
    assert jwt_module._jwk_client(JWKS_URL) is first
    assert jwt_module._jwk_client("https://other.test/jwks") is not first
