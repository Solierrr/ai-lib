from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import BaseModel

from ai_lib.llm import chat as chat_module
from ai_lib.llm import get_chat_model, get_chat_model_with_fallback
from ai_lib.llm.errors import classify_key_failure
from ai_lib.registry import KeyLease, RegistryKeysUnavailable


class FakeRegistry:
    def __init__(self, keys: dict[str, list[str]]):
        self.keys = {provider: list(ids) for provider, ids in keys.items()}
        self.leases: list[dict[str, Any]] = []
        self.reports: list[tuple] = []

    def _lease(self, provider, purpose, exclude) -> KeyLease:
        self.leases.append({"provider": provider, "purpose": purpose, "exclude": list(exclude)})
        for key_id in self.keys.get(provider, []):
            if key_id not in exclude:
                return KeyLease(
                    provider=provider,
                    key_id=key_id,
                    api_key=f"secret-{key_id}",
                    base_url="https://example.test",
                    auth_header={"name": "h", "value": f"secret-{key_id}"},
                )
        raise RegistryKeysUnavailable("none left")

    def lease(self, *, provider=None, purpose=None, exclude=()):
        return self._lease(provider, purpose, exclude)

    async def alease(self, *, provider=None, purpose=None, exclude=()):
        return self._lease(provider, purpose, exclude)

    def report(self, key_id, outcome, retry_after_seconds=None):
        self.reports.append((key_id, outcome, retry_after_seconds))

    async def areport(self, key_id, outcome, retry_after_seconds=None):
        self.report(key_id, outcome, retry_after_seconds)


class RateLimited(Exception):
    status_code = 429


class FakeInner(BaseChatModel):
    api_key: str
    bad_keys: Any = None
    tools: Any = None

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tools": [t.name for t in tools]})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.bad_keys and self.api_key in self.bad_keys:
            raise RateLimited(f"limit for {self.api_key}")
        text = f"{self.api_key}|tools={self.tools}"
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=text, response_metadata={"model_name": "fake-model"})
                )
            ]
        )

    def with_structured_output(self, schema, **kwargs):
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(lambda _: schema(answer=self.api_key))


@pytest.fixture(autouse=True)
def fake_builders(monkeypatch):
    state = {"bad": set()}

    def builder(lease, model, temperature, kwargs):
        return FakeInner(api_key=lease.api_key, bad_keys=state["bad"])

    monkeypatch.setitem(chat_module.BUILDERS, "gemini", builder)
    monkeypatch.setitem(chat_module.BUILDERS, "groq", builder)
    return state


def test_invoke_uses_leased_key_and_reports_ok():
    registry = FakeRegistry({"gemini": ["a"]})
    model = get_chat_model("gemini", client=registry)

    result = model.invoke([HumanMessage(content="hi")])

    assert result.content.startswith("secret-a|")
    assert result.response_metadata["model_name"] == "fake-model"
    assert registry.leases == [{"provider": "gemini", "purpose": "chat", "exclude": []}]
    assert registry.reports == [("a", "ok", None)]


def test_rate_limit_reports_and_retries_with_exclude(fake_builders):
    fake_builders["bad"].add("secret-a")
    registry = FakeRegistry({"gemini": ["a", "b"]})

    result = get_chat_model("gemini", client=registry).invoke("hi")

    assert result.content.startswith("secret-b|")
    assert registry.leases[1]["exclude"] == ["a"]
    assert registry.reports == [("a", "rate_limited", None), ("b", "ok", None)]


def test_exhausted_keys_raise_registry_error(fake_builders):
    fake_builders["bad"].update({"secret-a", "secret-b"})
    registry = FakeRegistry({"gemini": ["a", "b"]})

    with pytest.raises(RegistryKeysUnavailable):
        get_chat_model("gemini", client=registry).invoke("hi")


def test_max_key_attempts_reraises_last_provider_error(fake_builders):
    fake_builders["bad"].update({"secret-a", "secret-b"})
    registry = FakeRegistry({"gemini": ["a", "b", "c"]})

    with pytest.raises(RateLimited):
        get_chat_model("gemini", client=registry, max_key_attempts=2).invoke("hi")
    assert [r[1] for r in registry.reports] == ["rate_limited", "rate_limited"]


def test_unrelated_errors_propagate_without_report(monkeypatch):
    def builder(lease, model, temperature, kwargs):
        raise ValueError("boom")

    monkeypatch.setitem(chat_module.BUILDERS, "groq", builder)
    registry = FakeRegistry({"groq": ["g"]})

    with pytest.raises(ValueError):
        get_chat_model("groq", client=registry).invoke("hi")
    assert registry.reports == []


async def test_ainvoke_leases_asynchronously(fake_builders):
    fake_builders["bad"].add("secret-a")
    registry = FakeRegistry({"groq": ["a", "b"]})

    result = await get_chat_model("groq", client=registry).ainvoke("hi")

    assert result.content.startswith("secret-b|")
    assert registry.reports == [("a", "rate_limited", None), ("b", "ok", None)]


def test_bind_tools_is_applied_to_each_inner_model():
    @tool
    def add(a: int, b: int) -> int:
        """Add numbers."""
        return a + b

    registry = FakeRegistry({"gemini": ["a"]})
    model = get_chat_model("gemini", client=registry).bind_tools([add])

    assert model.invoke("hi").content == "secret-a|tools=['add']"


def test_with_structured_output_leases_per_call():
    class Answer(BaseModel):
        answer: str

    registry = FakeRegistry({"gemini": ["a"]})
    structured = get_chat_model("gemini", client=registry).with_structured_output(Answer)

    assert structured.invoke("hi") == Answer(answer="secret-a")
    assert registry.reports == [("a", "ok", None)]


async def test_with_structured_output_async():
    class Answer(BaseModel):
        answer: str

    registry = FakeRegistry({"gemini": ["a"]})
    structured = get_chat_model("gemini", client=registry).with_structured_output(Answer)

    assert (await structured.ainvoke("hi")).answer == "secret-a"


def test_fallback_switches_provider_when_primary_has_no_keys():
    registry = FakeRegistry({"groq": ["g"]})
    model = get_chat_model_with_fallback(client=registry)

    assert model.invoke("hi").content.startswith("secret-g|")
    assert [lease["provider"] for lease in registry.leases] == ["gemini", "groq"]


def test_purpose_and_default_models_are_forwarded(monkeypatch):
    seen = {}

    def builder(lease, model, temperature, kwargs):
        seen.update(model=model, temperature=temperature, kwargs=kwargs)
        return FakeInner(api_key=lease.api_key)

    monkeypatch.setitem(chat_module.BUILDERS, "gemini", builder)
    registry = FakeRegistry({"gemini": ["a"]})

    get_chat_model("gemini", purpose="vision", temperature=0.1, client=registry, timeout=60).invoke(
        "hi"
    )

    assert registry.leases[0]["purpose"] == "vision"
    assert seen == {"model": "gemini-2.5-flash", "temperature": 0.1, "kwargs": {"timeout": 60}}


def test_real_builders_construct_provider_models_without_network():
    lease = KeyLease(
        provider="groq",
        key_id="k",
        api_key="gsk_test",
        base_url="https://api.groq.com/openai/v1",
        auth_header={"name": "Authorization", "value": "Bearer gsk_test"},
    )
    groq = chat_module._build_groq(lease, "openai/gpt-oss-120b", 0.2, {})
    gemini = chat_module._build_gemini(lease, "gemini-2.5-flash", 0.2, {})

    assert type(groq).__name__ == "ChatGroq"
    assert type(gemini).__name__ == "ChatGoogleGenerativeAI"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RateLimited("x"), ("rate_limited", None)),
        (Exception("429 RESOURCE_EXHAUSTED quota"), ("rate_limited", None)),
        (Exception("API key not valid. Please pass a valid API key."), ("invalid", None)),
        (ValueError("nothing relevant"), None),
    ],
)
def test_classify_key_failure(error, expected):
    failure = classify_key_failure(error)
    assert (failure and (failure.outcome, failure.retry_after)) == expected


def test_classify_follows_the_cause_chain_and_reads_retry_after():
    class Response:
        status_code = 429
        headers = {"retry-after": "12"}

    class Inner(Exception):
        response = Response()

    try:
        try:
            raise Inner("quota")
        except Inner as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as wrapped:
        failure = classify_key_failure(wrapped)

    assert failure and failure.outcome == "rate_limited" and failure.retry_after == 12.0
