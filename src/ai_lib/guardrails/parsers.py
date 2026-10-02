from collections.abc import Collection

from pydantic import BaseModel

INPUT_CATEGORIES = frozenset(
    {
        "APROVADO",
        "REDIRECIONAR",
        "FORA_ESCOPO",
        "MANIPULACAO",
        "DADOS_INTERNOS",
        "INFORMACAO_FALSA",
    }
)


class InputClassification(BaseModel):
    category: str
    reason: str


class OutputReview(BaseModel):
    revised_response: str
    was_corrected: bool


class JudgeVerdict(BaseModel):
    status: str
    justification: str


def _require_text(text: object, label: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"{label} response is not text")
    return text


def _parse_fields(text: str, allowed: set[str], label: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"line without marker in {label}")
        key, value = line.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if key not in allowed or key in fields or not value:
            raise ValueError(f"invalid field in {label}")
        fields[key] = value
    if set(fields) != allowed:
        raise ValueError(f"missing required fields in {label}")
    return fields


def parse_input_classification(
    text: str, valid_categories: Collection[str] = INPUT_CATEGORIES
) -> InputClassification:
    fields = _parse_fields(
        _require_text(text, "classifier"), {"CATEGORIA", "JUSTIFICATIVA"}, "classifier"
    )
    category = fields["CATEGORIA"].upper()
    if category not in valid_categories:
        raise ValueError("unknown category")
    return InputClassification(category=category, reason=fields["JUSTIFICATIVA"])


def parse_output_review(text: str) -> OutputReview:
    status: str | None = None
    response_lines: list[str] | None = None
    for line in _require_text(text, "compliance").splitlines():
        content = line.strip()
        upper = content.upper()
        if not content:
            if response_lines is not None:
                response_lines.append("")
            continue
        if upper.startswith("STATUS:"):
            if status is not None or response_lines is not None:
                raise ValueError("invalid STATUS marker")
            status = content.split(":", 1)[1].strip().upper()
        elif upper.startswith("RESPOSTA:"):
            if status is None or response_lines is not None:
                raise ValueError("invalid RESPOSTA marker")
            response_lines = [content.split(":", 1)[1].strip()]
        elif response_lines is not None:
            response_lines.append(line)
        else:
            raise ValueError("text outside the compliance contract")

    revised = "\n".join(response_lines or []).strip()
    if status not in {"APROVADO", "CORRIGIDO"} or not revised:
        raise ValueError("incomplete or invalid compliance response")
    return OutputReview(revised_response=revised, was_corrected=status == "CORRIGIDO")


def parse_judge_verdict(text: str) -> JudgeVerdict:
    fields = _parse_fields(_require_text(text, "judge"), {"STATUS", "JUSTIFICATIVA"}, "judge")
    status = fields["STATUS"].upper()
    if status not in {"APROVADO", "REPROVADO"}:
        raise ValueError("unknown status")
    return JudgeVerdict(status=status, justification=fields["JUSTIFICATIVA"])
