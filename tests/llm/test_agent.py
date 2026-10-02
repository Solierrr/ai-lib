from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from ai_lib.llm import chat as chat_module
from ai_lib.llm import get_chat_model, get_chat_model_with_fallback
from tests.llm.test_chat import FakeRegistry


class ToolCallingInner(BaseChatModel):
    api_key: str
    bound: Any = None

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"bound": [t.name for t in tools]})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if any(isinstance(m, ToolMessage) for m in messages):
            message = AIMessage(content=f"result={messages[-1].content} key={self.api_key}")
        else:
            message = AIMessage(
                content="",
                tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "call-1"}],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@pytest.fixture(autouse=True)
def tool_calling_builders(monkeypatch):
    def builder(lease, model, temperature, kwargs):
        return ToolCallingInner(api_key=lease.api_key)

    monkeypatch.setitem(chat_module.BUILDERS, "gemini", builder)
    monkeypatch.setitem(chat_module.BUILDERS, "groq", builder)


def final_text(result):
    return result["messages"][-1].content


def test_create_agent_runs_a_tool_loop_with_a_leased_model():
    registry = FakeRegistry({"gemini": ["a"]})
    agent = create_agent(get_chat_model("gemini", client=registry), tools=[add])

    result = agent.invoke({"messages": [("user", "2+3?")]})

    assert final_text(result) == "result=5 key=secret-a"
    assert len(registry.leases) == 2
    assert [report[1] for report in registry.reports] == ["ok", "ok"]


async def test_create_agent_async_with_fallback_runnable():
    registry = FakeRegistry({"gemini": ["a"]})
    agent = create_agent(get_chat_model_with_fallback(client=registry), tools=[add])

    result = await agent.ainvoke({"messages": [("user", "2+3?")]})

    assert final_text(result) == "result=5 key=secret-a"
