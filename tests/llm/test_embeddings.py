import langchain_google_genai

from ai_lib.llm import LeasedEmbeddings
from tests.llm.test_chat import FakeRegistry, RateLimited


class FakeGenaiEmbeddings:
    bad_keys: set[str] = set()

    def __init__(self, model, api_key, **kwargs):
        self.api_key = api_key
        self.model = model

    def _vector(self, text):
        if self.api_key in self.bad_keys:
            raise RateLimited("quota")
        return [float(len(text)), float(len(self.api_key))]

    def embed_query(self, text):
        return self._vector(text)

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    async def aembed_query(self, text):
        return self._vector(text)

    async def aembed_documents(self, texts):
        return [self._vector(t) for t in texts]


def make(monkeypatch, bad=()):
    FakeGenaiEmbeddings.bad_keys = set(bad)
    monkeypatch.setattr(langchain_google_genai, "GoogleGenerativeAIEmbeddings", FakeGenaiEmbeddings)
    registry = FakeRegistry({"gemini": ["a", "b"]})
    return LeasedEmbeddings(client=registry), registry


def test_embeddings_lease_with_embedding_purpose(monkeypatch):
    embeddings, registry = make(monkeypatch)

    assert embeddings.embed_query("abc") == [3.0, 8.0]
    assert embeddings.embed_documents(["a", "bb"]) == [[1.0, 8.0], [2.0, 8.0]]
    assert registry.leases[0]["purpose"] == "embedding"
    assert registry.leases[0]["provider"] == "gemini"


def test_embeddings_retry_with_another_key_on_rate_limit(monkeypatch):
    embeddings, registry = make(monkeypatch, bad={"secret-a"})

    assert embeddings.embed_query("abc") == [3.0, 8.0]
    assert registry.reports[0] == ("a", "rate_limited", None)
    assert registry.leases[1]["exclude"] == ["a"]


async def test_async_embeddings(monkeypatch):
    embeddings, registry = make(monkeypatch)

    assert await embeddings.aembed_query("abc") == [3.0, 8.0]
    assert await embeddings.aembed_documents(["a"]) == [[1.0, 8.0]]
    assert registry.reports[-1] == ("a", "ok", None)
