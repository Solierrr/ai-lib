from ai_lib.llm import compute_cost_usd, get_pricing, pricing


def test_exact_match():
    assert get_pricing("gemini-2.5-flash") == {"in": 0.0003, "out": 0.0025}


def test_prefix_match_prefers_the_longest_known_model(monkeypatch):
    monkeypatch.setitem(pricing.MODEL_PRICING, "gemini-2.5-flash-lite", {"in": 0.1, "out": 0.2})

    assert get_pricing("gemini-2.5-flash-lite-001") == {"in": 0.1, "out": 0.2}
    assert get_pricing("gemini-2.5-flash-001") == {"in": 0.0003, "out": 0.0025}


def test_unknown_model_costs_zero():
    assert get_pricing("mystery") == {"in": 0.0, "out": 0.0}
    assert compute_cost_usd("mystery", 1000, 1000) == 0.0


def test_cost_is_per_thousand_tokens():
    assert compute_cost_usd("gemini-2.5-flash", 2000, 1000) == round(2 * 0.0003 + 0.0025, 6)
