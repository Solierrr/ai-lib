import pytest

from ai_lib.llm import chat as chat_module
from ai_lib.llm import classify_key_failure
from ai_lib.registry import KeyLease


class HttpError(Exception):
    def __init__(self, status_code, message="boom"):
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize(
    "message",
    [
        "Request 4291 failed: model returned 429 tokens",
        "context length 24290 tokens exceeded",
        "ReadTimeout while waiting for 429 bytes",
    ],
)
def test_numbers_in_messages_are_not_rate_limits(message):
    assert classify_key_failure(Exception(message)) is None


@pytest.mark.parametrize(
    "error",
    [
        HttpError(429),
        Exception("RESOURCE_EXHAUSTED: quota exceeded"),
        Exception("Too Many Requests"),
        Exception("rate limit reached for model"),
    ],
)
def test_rate_limit_signals(error):
    failure = classify_key_failure(error)
    assert failure is not None and failure.outcome == "rate_limited"


@pytest.mark.parametrize(
    "error",
    [
        HttpError(401),
        Exception("API key not valid. Please pass a valid API key."),
        Exception("Your API key was reported as leaked."),
    ],
)
def test_invalid_key_signals(error):
    failure = classify_key_failure(error)
    assert failure is not None and failure.outcome == "invalid"


def test_forbidden_is_not_blamed_on_the_key():
    assert classify_key_failure(HttpError(403, "region not supported")) is None


def test_non_numeric_retry_after_is_ignored():
    class Response:
        status_code = 429
        headers = {"retry-after": "soon"}

    class Limited(Exception):
        response = Response()

    failure = classify_key_failure(Limited("quota"))
    assert failure is not None and failure.retry_after is None


def test_provider_sdks_do_not_retry_with_the_same_key_for_long():
    lease = KeyLease(
        provider="groq",
        key_id="k",
        api_key="gsk_test",
        base_url="https://api.groq.com/openai/v1",
        auth_header={"name": "Authorization", "value": "Bearer gsk_test"},
    )

    groq = chat_module._build_groq(lease, "m", 0.2, {})
    gemini = chat_module._build_gemini(lease, "m", 0.2, {})
    overridden = chat_module._build_groq(lease, "m", 0.2, {"max_retries": 4})

    assert groq.max_retries == 0
    assert gemini.max_retries == 1
    assert overridden.max_retries == 4
