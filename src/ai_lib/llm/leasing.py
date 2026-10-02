from collections.abc import Awaitable, Callable
from typing import TypeVar

from ai_lib.llm.errors import classify_key_failure
from ai_lib.registry import KeyLease, Provider, Purpose, RegistryClient

T = TypeVar("T")

_default_client: RegistryClient | None = None


def default_client() -> RegistryClient:
    global _default_client
    if _default_client is None:
        _default_client = RegistryClient.from_env()
    return _default_client


def reset_default_client() -> None:
    global _default_client
    _default_client = None


class KeyLeaser:
    def __init__(
        self,
        *,
        provider: Provider,
        purpose: Purpose | None = None,
        client: RegistryClient | None = None,
        max_key_attempts: int = 3,
        report_success: bool = True,
    ) -> None:
        self.provider = provider
        self.purpose = purpose
        self.client = client
        self.max_key_attempts = max_key_attempts
        self.report_success = report_success

    def _client(self) -> RegistryClient:
        return self.client or default_client()

    def run(self, operation: Callable[[KeyLease], T]) -> T:
        client = self._client()
        excluded: list[str] = []
        last_error: BaseException | None = None
        for _ in range(self.max_key_attempts):
            lease = client.lease(provider=self.provider, purpose=self.purpose, exclude=excluded)
            try:
                result = operation(lease)
            except Exception as error:
                failure = classify_key_failure(error)
                if failure is None:
                    raise
                client.report(lease.key_id, failure.outcome, failure.retry_after)
                excluded.append(lease.key_id)
                last_error = error
                continue
            if self.report_success:
                client.report(lease.key_id, "ok")
            return result
        assert last_error is not None
        raise last_error

    async def arun(self, operation: Callable[[KeyLease], Awaitable[T]]) -> T:
        client = self._client()
        excluded: list[str] = []
        last_error: BaseException | None = None
        for _ in range(self.max_key_attempts):
            lease = await client.alease(
                provider=self.provider, purpose=self.purpose, exclude=excluded
            )
            try:
                result = await operation(lease)
            except Exception as error:
                failure = classify_key_failure(error)
                if failure is None:
                    raise
                await client.areport(lease.key_id, failure.outcome, failure.retry_after)
                excluded.append(lease.key_id)
                last_error = error
                continue
            if self.report_success:
                await client.areport(lease.key_id, "ok")
            return result
        assert last_error is not None
        raise last_error
