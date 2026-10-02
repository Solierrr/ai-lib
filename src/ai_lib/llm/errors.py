from dataclasses import dataclass

from ai_lib.registry.schemas import Outcome

RATE_LIMIT_MARKERS = ("resource_exhausted", "rate limit", "rate_limit", "too many requests")
INVALID_KEY_MARKERS = (
    "api_key_invalid",
    "api key not valid",
    "invalid api key",
    "invalid_api_key",
    "reported as leaked",
)
MAX_CAUSE_DEPTH = 5


@dataclass(frozen=True)
class KeyFailure:
    outcome: Outcome
    retry_after: float | None = None


def _status_of(error: BaseException) -> int | None:
    for candidate in (
        getattr(error, "status_code", None),
        getattr(error, "code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int):
            return candidate
    return None


def _retry_after_of(error: BaseException) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def classify_key_failure(error: BaseException) -> KeyFailure | None:
    seen: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and len(seen) < MAX_CAUSE_DEPTH:
        seen.append(current)
        current = current.__cause__ or current.__context__

    for candidate in seen:
        status = _status_of(candidate)
        text = str(candidate).lower()
        if status == 429 or any(marker in text for marker in RATE_LIMIT_MARKERS):
            return KeyFailure("rate_limited", _retry_after_of(candidate))
        if status == 401 or any(marker in text for marker in INVALID_KEY_MARKERS):
            return KeyFailure("invalid")
    return None
