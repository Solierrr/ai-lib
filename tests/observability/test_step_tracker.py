from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from ai_lib.observability import StepTracker
from ai_lib.observability.step_tracker import categorize_llm_error, categorize_tool_error


class Sink:
    def __init__(self, fail=False):
        self.docs: list[dict] = []
        self.fail = fail

    async def __call__(self, doc):
        if self.fail:
            raise RuntimeError("messenger down")
        self.docs.append(doc)


def llm_result(model="gemini-2.5-flash", tokens_in=1000, tokens_out=1000):
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        },
        response_metadata={"model_name": model},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


async def test_llm_call_is_recorded_with_cost_and_tokens():
    sink = Sink()
    tracker = StepTracker("conv-1", sink)
    run_id = uuid4()

    await tracker.on_llm_start({}, [], run_id=run_id, metadata={"langgraph_node": "orchestrator"})
    await tracker.on_llm_end(llm_result(), run_id=run_id)

    doc = sink.docs[0]
    assert doc["conversationId"] == "conv-1"
    assert doc["node"] == "orchestrator"
    assert doc["stepOrder"] == 1
    assert doc["stepType"] == "LLM_CALL"
    assert doc["model"] == "gemini-2.5-flash"
    assert (doc["tokensIn"], doc["tokensOut"], doc["tokensTotal"]) == (1000, 1000, 2000)
    assert doc["costUsd"] == round(0.0003 + 0.0025, 6)
    assert doc["status"] == "ok"
    assert doc["latencyMs"] >= 0


async def test_step_order_increases_and_sender_failures_are_swallowed():
    sink = Sink()
    tracker = StepTracker("conv-1", sink)
    for _ in range(2):
        run_id = uuid4()
        await tracker.on_llm_start({}, [], run_id=run_id)
        await tracker.on_llm_end(llm_result(), run_id=run_id)
    assert [d["stepOrder"] for d in sink.docs] == [1, 2]

    broken = StepTracker("conv-2", Sink(fail=True))
    run_id = uuid4()
    await broken.on_llm_start({}, [], run_id=run_id)
    await broken.on_llm_end(llm_result(), run_id=run_id)


async def test_llm_error_is_categorized():
    class RateLimited(Exception):
        status_code = 429

    sink = Sink()
    tracker = StepTracker("conv-1", sink)
    run_id = uuid4()

    await tracker.on_llm_start({}, [], run_id=run_id)
    await tracker.on_llm_error(RateLimited("slow down"), run_id=run_id)

    assert sink.docs[0]["status"] == "rate_limited"
    assert sink.docs[0]["error"] == "slow down"


async def test_tool_call_success_and_error():
    sink = Sink()
    tracker = StepTracker("conv-1", sink)

    ok_run = uuid4()
    await tracker.on_tool_start({"name": "geocode"}, "x", run_id=ok_run)
    await tracker.on_tool_end("out", run_id=ok_run)

    err_run = uuid4()
    await tracker.on_tool_start({"name": "weather"}, "x", run_id=err_run)
    await tracker.on_tool_error(httpx.ConnectError("no route"), run_id=err_run)

    ok_doc, err_doc = sink.docs
    assert (ok_doc["toolName"], ok_doc["status"], ok_doc["model"]) == ("geocode", "ok", "n/a")
    assert (err_doc["toolName"], err_doc["status"]) == ("weather", "connection_error")
    assert err_doc["error"] == "no route"


def test_error_categories():
    assert categorize_llm_error(httpx.ReadTimeout("slow")) == "timeout"
    assert categorize_llm_error(ValueError("x")) == "error"
    assert categorize_tool_error(httpx.ReadTimeout("slow")) == "timeout"
    assert categorize_tool_error(ValueError("x")) == "error"
