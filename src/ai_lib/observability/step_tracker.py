import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from httpx import ConnectError, TimeoutException
from langchain_core.callbacks import AsyncCallbackHandler

from ai_lib.llm.errors import classify_key_failure
from ai_lib.llm.pricing import compute_cost_usd

logger = logging.getLogger(__name__)

Sender = Callable[[dict[str, Any]], Awaitable[None]]


def categorize_llm_error(error: BaseException) -> str:
    failure = classify_key_failure(error)
    if failure is not None and failure.outcome == "rate_limited":
        return "rate_limited"
    if (
        isinstance(error, TimeoutException | TimeoutError)
        or "timeout" in type(error).__name__.lower()
    ):
        return "timeout"
    return "error"


def categorize_tool_error(error: BaseException) -> str:
    if isinstance(error, TimeoutException):
        return "timeout"
    if isinstance(error, ConnectError):
        return "connection_error"
    return "error"


class StepTracker(AsyncCallbackHandler):
    def __init__(self, conversation_id: str, sender: Sender) -> None:
        self.conversation_id = conversation_id
        self.sender = sender
        self.step_order = 0
        self._starts: dict[Any, float] = {}
        self._node_by_run: dict[Any, str] = {}
        self._tool_name_by_run: dict[Any, str] = {}

    def _next_order(self) -> int:
        self.step_order += 1
        return self.step_order

    def _elapsed_ms(self, run_id: Any) -> float:
        return round(
            (time.perf_counter() - self._starts.pop(run_id, time.perf_counter())) * 1000, 1
        )

    async def _save(self, node: str, doc: dict[str, Any]) -> None:
        doc = {
            "conversationId": self.conversation_id,
            "node": node,
            "stepOrder": self._next_order(),
            "timestamp": datetime.now(UTC).isoformat(),
            **doc,
        }
        try:
            await self.sender(doc)
        except Exception as error:
            logger.warning("failed to send observability (node=%s): %s", node, type(error).__name__)

    async def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs):
        self._starts[run_id] = time.perf_counter()
        self._node_by_run[run_id] = (metadata or {}).get("langgraph_node", "desconhecido")

    async def on_llm_end(self, response, *, run_id, **kwargs):
        latency_ms = self._elapsed_ms(run_id)
        node = self._node_by_run.pop(run_id, "desconhecido")
        message = getattr(response.generations[0][0], "message", None)
        usage = getattr(message, "usage_metadata", None) if message else None
        metadata = getattr(message, "response_metadata", {}) if message else {}

        model = metadata.get("model_name") or metadata.get("model") or "desconhecido"
        tokens_in = usage.get("input_tokens", 0) if usage else 0
        tokens_out = usage.get("output_tokens", 0) if usage else 0
        tokens_total = (usage.get("total_tokens") if usage else None) or tokens_in + tokens_out

        await self._save(
            node,
            {
                "stepType": "LLM_CALL",
                "model": model,
                "tokensIn": tokens_in,
                "tokensOut": tokens_out,
                "tokensTotal": tokens_total,
                "costUsd": compute_cost_usd(model, tokens_in, tokens_out),
                "latencyMs": latency_ms,
                "status": "ok",
            },
        )

    async def on_llm_error(self, error, *, run_id, **kwargs):
        latency_ms = self._elapsed_ms(run_id)
        node = self._node_by_run.pop(run_id, "desconhecido")
        await self._save(
            node,
            {
                "stepType": "LLM_CALL",
                "model": "desconhecido",
                "tokensIn": 0,
                "tokensOut": 0,
                "tokensTotal": 0,
                "costUsd": 0.0,
                "latencyMs": latency_ms,
                "status": categorize_llm_error(error),
                "error": str(error),
            },
        )

    async def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, **kwargs):
        self._starts[run_id] = time.perf_counter()
        self._node_by_run[run_id] = (metadata or {}).get("langgraph_node", "desconhecido")
        self._tool_name_by_run[run_id] = serialized.get("name", "tool_desconhecida")

    def _tool_doc(self, run_id: Any, status: str, error: BaseException | None = None) -> tuple:
        latency_ms = self._elapsed_ms(run_id)
        node = self._node_by_run.pop(run_id, "desconhecido")
        tool_name = self._tool_name_by_run.pop(run_id, "tool_desconhecida")
        doc: dict[str, Any] = {
            "stepType": "TOOL_CALL",
            "toolName": tool_name,
            "model": "n/a",
            "tokensIn": 0,
            "tokensOut": 0,
            "tokensTotal": 0,
            "costUsd": 0.0,
            "latencyMs": latency_ms,
            "status": status,
        }
        if error is not None:
            doc["error"] = str(error)
        return node, doc

    async def on_tool_end(self, output, *, run_id, **kwargs):
        node, doc = self._tool_doc(run_id, "ok")
        await self._save(node, doc)

    async def on_tool_error(self, error, *, run_id, **kwargs):
        node, doc = self._tool_doc(run_id, categorize_tool_error(error), error)
        await self._save(node, doc)
