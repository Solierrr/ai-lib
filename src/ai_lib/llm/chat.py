from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import ConfigDict, Field

from ai_lib.llm.leasing import KeyLeaser
from ai_lib.registry import KeyLease, Provider, Purpose, RegistryClient

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
}


def _build_gemini(lease: KeyLease, model: str, temperature: float, kwargs: dict) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model, temperature=temperature, api_key=lease.api_key, **kwargs
    )


def _build_groq(lease: KeyLease, model: str, temperature: float, kwargs: dict) -> Any:
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, temperature=temperature, api_key=lease.api_key, **kwargs)


BUILDERS: dict[str, Callable[[KeyLease, str, float, dict], Any]] = {
    "gemini": _build_gemini,
    "groq": _build_groq,
}


class LeasedChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Provider
    model_name: str
    temperature: float = 0.7
    purpose: Purpose | None = "chat"
    max_key_attempts: int = 3
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    registry: Any = Field(default=None, exclude=True, repr=False)
    bindings: tuple[tuple[str, tuple, dict], ...] = ()

    @property
    def _llm_type(self) -> str:
        return f"solaria-leased-{self.provider}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"provider": self.provider, "model_name": self.model_name}

    @property
    def _leaser(self) -> KeyLeaser:
        return KeyLeaser(
            provider=self.provider,
            purpose=self.purpose,
            client=self.registry,
            max_key_attempts=self.max_key_attempts,
        )

    def _inner(self, lease: KeyLease) -> Any:
        builder = BUILDERS[self.provider]
        inner = builder(lease, self.model_name, self.temperature, dict(self.model_kwargs))
        for method, args, kwargs in self.bindings:
            inner = getattr(inner, method)(*args, **kwargs)
        return inner

    def _with_binding(self, method: str, args: tuple, kwargs: dict) -> LeasedChatModel:
        return self.model_copy(update={"bindings": (*self.bindings, (method, args, kwargs))})

    def bind_tools(self, tools: Any, **kwargs: Any) -> LeasedChatModel:
        return self._with_binding("bind_tools", (list(tools),), kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        leaser = self._leaser

        def invoke(value: Any, config: RunnableConfig | None = None) -> Any:
            return leaser.run(
                lambda lease: (
                    self._inner(lease)
                    .with_structured_output(schema, **kwargs)
                    .invoke(value, config)
                )
            )

        async def ainvoke(value: Any, config: RunnableConfig | None = None) -> Any:
            async def operation(lease: KeyLease) -> Any:
                return await (
                    self._inner(lease)
                    .with_structured_output(schema, **kwargs)
                    .ainvoke(value, config)
                )

            return await leaser.arun(operation)

        return RunnableLambda(invoke, afunc=ainvoke)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._leaser.run(
            lambda lease: self._inner(lease).invoke(messages, stop=stop, **kwargs)
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        async def operation(lease: KeyLease) -> Any:
            return await self._inner(lease).ainvoke(messages, stop=stop, **kwargs)

        message = await self._leaser.arun(operation)
        return ChatResult(generations=[ChatGeneration(message=message)])


def get_chat_model(
    provider: Provider,
    *,
    model: str | None = None,
    temperature: float = 0.7,
    purpose: Purpose | None = "chat",
    client: RegistryClient | None = None,
    max_key_attempts: int = 3,
    **model_kwargs: Any,
) -> LeasedChatModel:
    return LeasedChatModel(
        provider=provider,
        model_name=model or DEFAULT_MODELS[provider],
        temperature=temperature,
        purpose=purpose,
        registry=client,
        max_key_attempts=max_key_attempts,
        model_kwargs=model_kwargs,
    )


def get_chat_model_with_fallback(
    *,
    primary: Provider = "gemini",
    fallback: Provider = "groq",
    primary_model: str | None = None,
    fallback_model: str | None = None,
    temperature: float = 0.7,
    purpose: Purpose | None = "chat",
    client: RegistryClient | None = None,
) -> Runnable:
    return get_chat_model(
        primary, model=primary_model, temperature=temperature, purpose=purpose, client=client
    ).with_fallbacks(
        [
            get_chat_model(
                fallback,
                model=fallback_model,
                temperature=temperature,
                purpose=purpose,
                client=client,
            )
        ]
    )
