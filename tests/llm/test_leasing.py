import pytest

from ai_lib.llm import leasing
from ai_lib.registry import RegistryClient, RegistryError


@pytest.fixture(autouse=True)
def clean_default_client():
    leasing.reset_default_client()
    yield
    leasing.reset_default_client()


def test_default_client_is_built_from_the_environment_once(monkeypatch):
    monkeypatch.setenv("REGISTRY_URL", "https://registry.test")
    monkeypatch.setenv("REGISTRY_CONSUMER_TOKEN", "token")

    first = leasing.default_client()

    assert isinstance(first, RegistryClient)
    assert leasing.default_client() is first


def test_default_client_requires_configuration(monkeypatch):
    monkeypatch.delenv("REGISTRY_URL", raising=False)
    monkeypatch.delenv("REGISTRY_CONSUMER_TOKEN", raising=False)

    with pytest.raises(RegistryError):
        leasing.default_client()


def test_reset_drops_the_cached_client(monkeypatch):
    monkeypatch.setenv("REGISTRY_URL", "https://registry.test")
    monkeypatch.setenv("REGISTRY_CONSUMER_TOKEN", "token")
    first = leasing.default_client()

    leasing.reset_default_client()

    assert leasing.default_client() is not first
