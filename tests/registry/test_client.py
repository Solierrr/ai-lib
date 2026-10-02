import json

import httpx
import pytest

from ai_lib.registry import (
    RegistryAuthError,
    RegistryClient,
    RegistryError,
    RegistryKeysNotConfigured,
    RegistryKeysUnavailable,
    RegistryUnreachable,
    RetryPolicy,
)

LEASE = {
    "provider": "gemini",
    "key_id": "k1",
    "api_key": "secret",
    "base_url": "https://generativelanguage.googleapis.com/v1beta",
    "auth_header": {"name": "x-goog-api-key", "value": "secret"},
}


class Script:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(script: Script, sleeps: list[float], **policy) -> RegistryClient:
    async def async_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    transport = httpx.MockTransport(script)
    return RegistryClient(
        "https://registry.test",
        "token",
        policy=RetryPolicy(**policy) if policy else None,
        transport=transport,
        async_transport=transport,
        sleep=sleeps.append,
        async_sleep=async_sleep,
    )


def ok() -> httpx.Response:
    return httpx.Response(200, json=LEASE)


def test_lease_returns_key_and_sends_token_and_filters():
    script = Script(ok())
    client = make_client(script, [])

    lease = client.lease(provider="gemini", purpose="vision", exclude=["a", "b"])

    assert lease.key_id == "k1"
    assert lease.api_key == "secret"
    request = script.requests[0]
    assert request.headers["authorization"] == "Bearer token"
    assert request.url.params["provider"] == "gemini"
    assert request.url.params["purpose"] == "vision"
    assert request.url.params.get_list("exclude") == ["a", "b"]


def test_lease_repr_hides_secret():
    lease = make_client(Script(ok()), []).lease()
    assert "secret" not in repr(lease)


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_is_immediate(status):
    script = Script(httpx.Response(status))
    sleeps: list[float] = []
    with pytest.raises(RegistryAuthError):
        make_client(script, sleeps).lease()
    assert len(script.requests) == 1
    assert sleeps == []


def test_not_configured_is_immediate():
    script = Script(httpx.Response(404, json={"code": "x"}))
    with pytest.raises(RegistryKeysNotConfigured):
        make_client(script, []).lease()
    assert len(script.requests) == 1


def test_unexpected_client_error_is_immediate():
    with pytest.raises(RegistryError):
        make_client(Script(httpx.Response(422)), []).lease()


def test_503_without_retry_after_backs_off_then_succeeds():
    script = Script(httpx.Response(503), httpx.Response(503), ok())
    sleeps: list[float] = []

    assert make_client(script, sleeps).lease().key_id == "k1"
    assert sleeps == [1.0, 2.0]


def test_503_without_retry_after_gives_up_after_max_attempts():
    script = Script(*[httpx.Response(503) for _ in range(3)])
    sleeps: list[float] = []

    with pytest.raises(RegistryKeysUnavailable):
        make_client(script, sleeps).lease()
    assert len(script.requests) == 3
    assert sleeps == [1.0, 2.0]


def test_503_with_retry_after_waits_once_and_retries():
    script = Script(httpx.Response(503, headers={"Retry-After": "7"}), ok())
    sleeps: list[float] = []

    assert make_client(script, sleeps).lease().key_id == "k1"
    assert sleeps == [7.0]


def test_retry_after_is_capped():
    script = Script(httpx.Response(503, headers={"Retry-After": "600"}), ok())
    sleeps: list[float] = []

    make_client(script, sleeps).lease()
    assert sleeps == [30.0]


def test_second_503_with_retry_after_raises_with_hint():
    script = Script(
        httpx.Response(503, headers={"Retry-After": "5"}),
        httpx.Response(503, headers={"Retry-After": "9"}),
    )
    with pytest.raises(RegistryKeysUnavailable) as raised:
        make_client(script, []).lease()
    assert raised.value.retry_after == 9.0


def test_transport_errors_retry_then_raise_unreachable():
    error = httpx.ConnectError("down")
    script = Script(error, error, error)
    sleeps: list[float] = []

    with pytest.raises(RegistryUnreachable):
        make_client(script, sleeps).lease()
    assert sleeps == [1.0, 2.0]


def test_cold_start_timeout_then_success():
    script = Script(httpx.ReadTimeout("sleeping"), ok())
    sleeps: list[float] = []

    assert make_client(script, sleeps).lease().key_id == "k1"
    assert sleeps == [1.0]


def test_total_budget_stops_retries():
    script = Script(httpx.Response(503), httpx.Response(503))
    with pytest.raises(RegistryKeysUnavailable):
        make_client(script, [], max_total_seconds=0.5).lease()
    assert len(script.requests) == 1


def test_report_sends_outcome_and_clamps_retry_after():
    script = Script(httpx.Response(204))
    make_client(script, []).report("k1", "rate_limited", 120.7)

    request = script.requests[0]
    assert request.url.path == "/v1/llm/keys/k1/report"
    assert json.loads(request.content) == {"outcome": "rate_limited", "retry_after_seconds": 120}


def test_report_ok_has_no_retry_after():
    script = Script(httpx.Response(204))
    make_client(script, []).report("k1", "ok", 30)
    assert json.loads(script.requests[0].content) == {"outcome": "ok"}


def test_report_never_raises():
    client = make_client(Script(httpx.ConnectError("down")), [])
    client.report("k1", "invalid")
    client = make_client(Script(httpx.Response(404)), [])
    client.report("k1", "invalid")


async def test_alease_retries_and_succeeds():
    script = Script(httpx.Response(503, headers={"Retry-After": "2"}), ok())
    sleeps: list[float] = []

    lease = await make_client(script, sleeps).alease(provider="groq")
    assert lease.key_id == "k1"
    assert sleeps == [2.0]


async def test_alease_auth_failure():
    with pytest.raises(RegistryAuthError):
        await make_client(Script(httpx.Response(401)), []).alease()


async def test_areport_never_raises():
    script = Script(httpx.Response(204), httpx.ConnectError("down"))
    client = make_client(script, [])
    await client.areport("k1", "ok")
    await client.areport("k1", "invalid")


def test_from_env_requires_both_variables(monkeypatch):
    monkeypatch.delenv("REGISTRY_URL", raising=False)
    monkeypatch.delenv("REGISTRY_CONSUMER_TOKEN", raising=False)
    with pytest.raises(RegistryError):
        RegistryClient.from_env()

    monkeypatch.setenv("REGISTRY_URL", "https://registry.test/")
    monkeypatch.setenv("REGISTRY_CONSUMER_TOKEN", "token")
    assert isinstance(RegistryClient.from_env(), RegistryClient)
