import logging

logger = logging.getLogger(__name__)

MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"in": 0.0003, "out": 0.0025},
    "openai/gpt-oss-120b": {"in": 0.00015, "out": 0.00060},
}

DEFAULT_PRICING: dict[str, float] = {"in": 0.0, "out": 0.0}


def get_pricing(model: str) -> dict[str, float]:
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    matches = [known for known in MODEL_PRICING if model.startswith(known)]
    if matches:
        return MODEL_PRICING[max(matches, key=len)]
    logger.warning("no pricing registered for model %s, cost will be 0.0", model)
    return DEFAULT_PRICING


def compute_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    pricing = get_pricing(model)
    return round((tokens_in / 1000) * pricing["in"] + (tokens_out / 1000) * pricing["out"], 6)
