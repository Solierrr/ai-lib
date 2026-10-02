import logging
from collections.abc import Callable, Collection
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable

from ai_lib.guardrails.anonymize import anonymize_text, deanonymize_text
from ai_lib.guardrails.injection import matches_injection_pattern, matches_internal_data_keyword
from ai_lib.guardrails.parsers import (
    INPUT_CATEGORIES,
    parse_input_classification,
    parse_judge_verdict,
    parse_output_review,
)
from ai_lib.guardrails.prompts import INPUT_PLACEHOLDER, OUTPUT_PLACEHOLDER
from ai_lib.guardrails.state import GuardrailState
from ai_lib.llm import get_chat_model

logger = logging.getLogger(__name__)

DEFAULT_INPUT_BLOCKED = "Desculpe, não posso processar essa solicitação por políticas de segurança."
DEFAULT_OUTPUT_FALLBACK = (
    "Não foi possível processar sua solicitação no momento. Tente novamente em instantes."
)
DEFAULT_JUDGE_BLOCKED = (
    "Não foi possível gerar uma resposta confiável para essa solicitação. "
    "Poderia reformular sua pergunta com mais detalhes?"
)
OUTPUT_FORMAT_INSTRUCTION = (
    "\n\nResponda EXATAMENTE neste formato, em texto (nao chame nenhuma tool):\n"
    "STATUS: APROVADO ou CORRIGIDO\nRESPOSTA: <texto final, ja revisado>"
)
JUDGE_FORMAT_INSTRUCTION = (
    "\n\nResponda EXATAMENTE neste formato, em texto (nao chame nenhuma tool):\n"
    "STATUS: APROVADO ou REPROVADO\nJUSTIFICATIVA: <texto>"
)

LlmSource = BaseChatModel | Runnable | Callable[[], Any] | None


def _resolve_llm(source: LlmSource) -> Any:
    if source is None:
        return get_chat_model("groq")
    if isinstance(source, Runnable):
        return source
    return source()


def _append_turn_agent(state: GuardrailState, agent: str) -> list[str]:
    return [*state.get("turn_agents", []), agent]


def make_input_guardrail_node(
    *,
    prompt: str,
    llm: LlmSource = None,
    blocked_response: str = DEFAULT_INPUT_BLOCKED,
    valid_categories: Collection[str] = INPUT_CATEGORIES,
) -> Callable[..., dict]:
    def blocked_result(state: GuardrailState, category: str, pii_map: dict | None = None) -> dict:
        return {
            "messages": [
                RemoveMessage(id=state["messages"][-1].id),
                AIMessage(content=blocked_response),
            ],
            "route": "end",
            "pii_map": pii_map or {},
            "turn_agents": [f"input_guardrail_blocked_{category.lower()}"],
        }

    def input_guardrail_node(state: GuardrailState, config: Any = None) -> dict:
        last_message = state["messages"][-1].content

        if matches_injection_pattern(last_message):
            return blocked_result(state, "manipulacao_regex")
        if matches_internal_data_keyword(last_message):
            return blocked_result(state, "dados_internos_regex")

        anonymized_text, pii_map = anonymize_text(last_message)
        formatted_prompt = prompt.replace(INPUT_PLACEHOLDER, anonymized_text)

        try:
            response = _resolve_llm(llm).invoke(
                [HumanMessage(content=formatted_prompt)], config=config
            )
            classification = parse_input_classification(response.content, valid_categories)
        except Exception as error:
            logger.warning("input guardrail evaluation failed: %s", type(error).__name__)
            return blocked_result(state, "falha_avaliacao_guardrail")

        if classification.category != "APROVADO":
            return blocked_result(state, classification.category, pii_map)

        return {
            "messages": [
                RemoveMessage(id=state["messages"][-1].id),
                HumanMessage(content=anonymized_text),
            ],
            "route": "proceed",
            "pii_map": pii_map,
            "turn_agents": ["input_guardrail_approved"],
        }

    return input_guardrail_node


def make_output_guardrail_node(
    *,
    prompt: str,
    llm: LlmSource = None,
    fallback_response: str = DEFAULT_OUTPUT_FALLBACK,
    specialists: Collection[str] = (),
) -> Callable[..., dict]:
    def output_guardrail_node(state: GuardrailState, config: Any = None) -> dict:
        last_message_text = state["messages"][-1].content
        formatted_prompt = (
            prompt.replace(OUTPUT_PLACEHOLDER, last_message_text) + OUTPUT_FORMAT_INSTRUCTION
        )

        try:
            response = _resolve_llm(llm).invoke(
                [HumanMessage(content=formatted_prompt)], config=config
            )
            review = parse_output_review(response.content)
            final_text = deanonymize_text(review.revised_response, state.get("pii_map", {}))
        except Exception as error:
            logger.warning("output guardrail evaluation failed: %s", type(error).__name__)
            final_text = fallback_response

        workflow_steps = _append_turn_agent(state, "output_guardrail")

        return {
            "messages": [
                RemoveMessage(id=state["messages"][-1].id),
                AIMessage(
                    content=final_text,
                    additional_kwargs={
                        "specialists_used": [a for a in workflow_steps if a in specialists],
                        "workflow_steps": workflow_steps,
                    },
                ),
            ],
            "turn_agents": workflow_steps,
        }

    return output_guardrail_node


def make_judge_node(
    *,
    prompt: str,
    llm: LlmSource = None,
    blocked_response: str = DEFAULT_JUDGE_BLOCKED,
    max_retries: int = 1,
    extra_context: Callable[[GuardrailState], str] | None = None,
) -> Callable[..., dict]:
    def judge_node(state: GuardrailState, config: Any = None) -> dict:
        last_message = state["messages"][-1].content
        audited = f"Resposta a ser auditada:\n\n{last_message}"
        if extra_context is not None:
            audited = f"{audited}\n\n{extra_context(state)}"
        messages = [
            SystemMessage(content=prompt + JUDGE_FORMAT_INSTRUCTION),
            HumanMessage(content=audited),
        ]
        try:
            response = _resolve_llm(llm).invoke(messages, config=config)
            status = parse_judge_verdict(response.content).status
        except Exception as error:
            logger.warning("judge evaluation failed: %s", type(error).__name__)
            status = "REPROVADO"

        retries = state.get("judge_retries", 0)

        if status == "APROVADO":
            return {
                "judge_status": "approved",
                "turn_agents": _append_turn_agent(state, "judge_approved"),
            }

        if retries < max_retries:
            return {
                "messages": [RemoveMessage(id=state["messages"][-1].id)],
                "judge_status": "retry",
                "judge_retries": retries + 1,
                "turn_agents": _append_turn_agent(state, "judge_rejected"),
            }

        return {
            "messages": [
                RemoveMessage(id=state["messages"][-1].id),
                AIMessage(content=blocked_response),
            ],
            "judge_status": "blocked",
            "turn_agents": _append_turn_agent(state, "judge_blocked"),
        }

    return judge_node
