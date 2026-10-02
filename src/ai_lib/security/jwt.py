import jwt as pyjwt
from jwt import PyJWKClient

_clients: dict[str, PyJWKClient] = {}


def _jwk_client(jwks_url: str) -> PyJWKClient:
    client = _clients.get(jwks_url)
    if client is None:
        client = _clients[jwks_url] = PyJWKClient(jwks_url)
    return client


def decode_user_id(token: str, *, jwks_url: str, issuer: str) -> str | None:
    try:
        signing_key = _jwk_client(jwks_url).get_signing_key_from_jwt(token)
        payload = pyjwt.decode(token, signing_key.key, algorithms=["RS256"], issuer=issuer)
        return payload.get("sub")
    except Exception:
        return None
