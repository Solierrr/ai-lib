import json

import httpx
import pytest

from ai_lib.registry import RegistryClient, RegistryError


def make_client(handler):
    transport = httpx.MockTransport(handler)
    return RegistryClient(
        "https://registry.test",
        "token",
        transport=transport,
        async_transport=transport,
        sleep=lambda _: None,
    )


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_report_ignores_non_finite_retry_after(value):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(204)

    make_client(handler).report("k1", "rate_limited", value)

    assert seen == [{"outcome": "rate_limited"}]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"<html>not json</html>"),
        httpx.Response(200, json={"unexpected": "shape"}),
    ],
)
def test_invalid_lease_body_is_a_registry_error(response):
    with pytest.raises(RegistryError):
        make_client(lambda request: response).lease()


async def test_invalid_lease_body_is_a_registry_error_async():
    with pytest.raises(RegistryError):
        await make_client(lambda request: httpx.Response(200, content=b"nope")).alease()
