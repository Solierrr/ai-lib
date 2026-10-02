import asyncio
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from ai_lib.registry.errors import (
    RegistryAuthError,
    RegistryError,
    RegistryKeysNotConfigured,
    RegistryKeysUnavailable,
    RegistryUnreachable,
)
from ai_lib.registry.schemas import KeyLease, Outcome, Provider, Purpose

logger = logging.getLogger(__name__)

URL_ENV = "REGISTRY_URL"
TOKEN_ENV = "REGISTRY_CONSUMER_TOKEN"
KEYS_PATH = "/v1/llm/keys"


@dataclass(frozen=True)
class RetryPolicy:
    timeout: float = 10.0
    max_attempts: int = 3
    backoff_base: float = 1.0
    max_total_seconds: float = 60.0
    max_retry_after: float = 30.0


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return None


class RegistryClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        policy: RetryPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._policy = policy or RetryPolicy()
        self._transport = transport
        self._async_transport = async_transport
        self._sleep = sleep
        self._async_sleep = async_sleep

    @classmethod
    def from_env(cls, **kwargs: Any) -> RegistryClient:
        url = os.environ.get(URL_ENV)
        token = os.environ.get(TOKEN_ENV)
        if not url or not token:
            raise RegistryError(f"{URL_ENV} and {TOKEN_ENV} must be set")
        return cls(url, token, **kwargs)

    @staticmethod
    def _params(
        provider: Provider | None, purpose: Purpose | None, exclude: Sequence[str]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if provider:
            params["provider"] = provider
        if purpose:
            params["purpose"] = purpose
        if exclude:
            params["exclude"] = list(exclude)
        return params

    @staticmethod
    def _report_body(outcome: Outcome, retry_after_seconds: float | None) -> dict[str, Any]:
        body: dict[str, Any] = {"outcome": outcome}
        if outcome == "rate_limited" and retry_after_seconds and math.isfinite(retry_after_seconds):
            body["retry_after_seconds"] = max(1, min(int(retry_after_seconds), 86400))
        return body

    def _client_kwargs(self) -> dict[str, Any]:
        return {
            "base_url": self._base_url,
            "headers": {"Authorization": f"Bearer {self._token}"},
            "timeout": self._policy.timeout,
        }

    def _backoff(self, attempt: int) -> float:
        return self._policy.backoff_base * 2 ** (attempt - 1)

    def _decide(
        self,
        response: httpx.Response,
        attempt: int,
        elapsed: float,
        retry_after_used: bool,
    ) -> tuple[KeyLease | None, float, bool]:
        status = response.status_code
        if status == 200:
            try:
                return KeyLease.model_validate(response.json()), 0.0, retry_after_used
            except ValueError as error:
                raise RegistryError("registry returned an invalid key lease") from error
        if status in (401, 403):
            raise RegistryAuthError("registry rejected the consumer token")
        if status == 404:
            raise RegistryKeysNotConfigured("registry has no keys for this request")
        if status < 500:
            raise RegistryError(f"unexpected registry response: {status}")

        retry_after = _retry_after(response) if status == 503 else None
        if retry_after is not None:
            if retry_after_used:
                raise RegistryKeysUnavailable("no key available", retry_after)
            return None, min(retry_after, self._policy.max_retry_after), True
        wait = self._backoff(attempt)
        if attempt >= self._policy.max_attempts or elapsed + wait > self._policy.max_total_seconds:
            raise RegistryKeysUnavailable(f"registry unavailable ({status})")
        return None, wait, retry_after_used

    def _transport_wait(self, attempt: int, elapsed: float, error: Exception) -> float:
        wait = self._backoff(attempt)
        if attempt >= self._policy.max_attempts or elapsed + wait > self._policy.max_total_seconds:
            raise RegistryUnreachable("registry unreachable") from error
        return wait

    def lease(
        self,
        *,
        provider: Provider | None = None,
        purpose: Purpose | None = None,
        exclude: Sequence[str] = (),
    ) -> KeyLease:
        params = self._params(provider, purpose, exclude)
        started = time.monotonic()
        attempt = 1
        retry_after_used = False
        with httpx.Client(transport=self._transport, **self._client_kwargs()) as http:
            while True:
                elapsed = time.monotonic() - started
                try:
                    response = http.get(KEYS_PATH, params=params)
                except httpx.TransportError as error:
                    wait = self._transport_wait(attempt, elapsed, error)
                else:
                    lease, wait, retry_after_used = self._decide(
                        response, attempt, elapsed, retry_after_used
                    )
                    if lease is not None:
                        return lease
                self._sleep(wait)
                attempt += 1

    async def alease(
        self,
        *,
        provider: Provider | None = None,
        purpose: Purpose | None = None,
        exclude: Sequence[str] = (),
    ) -> KeyLease:
        params = self._params(provider, purpose, exclude)
        started = time.monotonic()
        attempt = 1
        retry_after_used = False
        async with httpx.AsyncClient(
            transport=self._async_transport, **self._client_kwargs()
        ) as http:
            while True:
                elapsed = time.monotonic() - started
                try:
                    response = await http.get(KEYS_PATH, params=params)
                except httpx.TransportError as error:
                    wait = self._transport_wait(attempt, elapsed, error)
                else:
                    lease, wait, retry_after_used = self._decide(
                        response, attempt, elapsed, retry_after_used
                    )
                    if lease is not None:
                        return lease
                await self._async_sleep(wait)
                attempt += 1

    def report(
        self, key_id: str, outcome: Outcome, retry_after_seconds: float | None = None
    ) -> None:
        try:
            with httpx.Client(transport=self._transport, **self._client_kwargs()) as http:
                http.post(
                    f"{KEYS_PATH}/{key_id}/report",
                    json=self._report_body(outcome, retry_after_seconds),
                ).raise_for_status()
        except httpx.HTTPError as error:
            logger.warning("failed to report key outcome %s: %s", outcome, type(error).__name__)

    async def areport(
        self, key_id: str, outcome: Outcome, retry_after_seconds: float | None = None
    ) -> None:
        try:
            async with httpx.AsyncClient(
                transport=self._async_transport, **self._client_kwargs()
            ) as http:
                response = await http.post(
                    f"{KEYS_PATH}/{key_id}/report",
                    json=self._report_body(outcome, retry_after_seconds),
                )
                response.raise_for_status()
        except httpx.HTTPError as error:
            logger.warning("failed to report key outcome %s: %s", outcome, type(error).__name__)
