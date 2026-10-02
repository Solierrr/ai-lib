from typing import Any

from langchain_core.embeddings import Embeddings

from ai_lib.llm.leasing import KeyLeaser
from ai_lib.registry import KeyLease, RegistryClient

DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


class LeasedEmbeddings(Embeddings):
    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        *,
        client: RegistryClient | None = None,
        max_key_attempts: int = 3,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._kwargs = kwargs
        self._leaser = KeyLeaser(
            provider="gemini",
            purpose="embedding",
            client=client,
            max_key_attempts=max_key_attempts,
        )

    def _inner(self, lease: KeyLease) -> Any:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=self._model, api_key=lease.api_key, **self._kwargs
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._leaser.run(lambda lease: self._inner(lease).embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._leaser.run(lambda lease: self._inner(lease).embed_query(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        async def operation(lease: KeyLease) -> list[list[float]]:
            return await self._inner(lease).aembed_documents(texts)

        return await self._leaser.arun(operation)

    async def aembed_query(self, text: str) -> list[float]:
        async def operation(lease: KeyLease) -> list[float]:
            return await self._inner(lease).aembed_query(text)

        return await self._leaser.arun(operation)
