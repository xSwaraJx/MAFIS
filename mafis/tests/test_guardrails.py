from __future__ import annotations

# Rails under test:
# 1. Investment advice block
# 2. Price prediction block
# 3. Graceful fallback when rails=None

from unittest.mock import MagicMock, patch

import pytest

from app.guardrails.guardrails import GuardrailsWrapper, GuardedResponse


@pytest.fixture
def guardrails_wrapper():
    return GuardrailsWrapper()


# ---------------------------------------------------------------------------
# Investment advice rail (8.3.2)
# ---------------------------------------------------------------------------

def test_investment_advice_rail_triggers(guardrails_wrapper):
    if guardrails_wrapper.rails is None:
        pytest.skip("Rails not loaded")
    guardrails_wrapper.rails = MagicMock()
    guardrails_wrapper.rails.generate.return_value = (
        "I'm not able to provide investment advice. I can share factual market data and analysis, "
        "but decisions about buying or selling securities should be made with a registered financial advisor."
    )
    r = guardrails_wrapper.apply("Should I buy AAPL?", "Yes, buy AAPL immediately.")
    assert r.triggered is True
    assert "not" in r.text.lower() or "advisor" in r.text.lower()


def test_factual_query_passes(guardrails_wrapper):
    if guardrails_wrapper.rails is None:
        pytest.skip("Rails not loaded")
    original_answer = "The fed funds rate is 5.25%."
    guardrails_wrapper.rails = MagicMock()
    guardrails_wrapper.rails.generate.return_value = original_answer
    r = guardrails_wrapper.apply("What is the fed funds rate?", original_answer)
    assert r.triggered is False
    assert original_answer in r.text


# ---------------------------------------------------------------------------
# Price prediction rail (8.3.3)
# ---------------------------------------------------------------------------

def test_price_prediction_rail_triggers(guardrails_wrapper):
    if guardrails_wrapper.rails is None:
        pytest.skip("Rails not loaded")
    guardrails_wrapper.rails = MagicMock()
    guardrails_wrapper.rails.generate.return_value = (
        "I cannot make specific price predictions. I can provide current data and historical context, "
        "but future price movements are uncertain."
    )
    r = guardrails_wrapper.apply("Where will AAPL go?", "AAPL will definitely reach $300.")
    assert r.triggered is True


def test_guardrails_wrapper_graceful_fallback(guardrails_wrapper):
    guardrails_wrapper.rails = None
    r = guardrails_wrapper.apply("any query", "any response")
    assert r.triggered is False
    assert r.text == "any response"


# ---------------------------------------------------------------------------
# Guardrails integration with agent (8.3.4)
# ---------------------------------------------------------------------------

def test_guardrail_flag_propagates_to_agent_response():
    with patch("app.agents.rates_agent.ChatOpenAI"):
        from app.agents.rates_agent import RatesAgent
        agent = RatesAgent()

    declined = (
        "I'm not able to provide investment advice. I can share factual market data and analysis, "
        "but decisions about buying or selling securities should be made with a registered financial advisor."
    )
    msg = MagicMock()
    msg.content = "You should definitely buy treasury bonds."
    msg.tool_calls = []
    agent.graph = MagicMock()
    agent.graph.invoke.return_value = {"messages": [msg]}

    agent.guardrails = MagicMock()
    agent.guardrails.apply.return_value = GuardedResponse(
        text=declined, triggered=True, rail="financial"
    )

    result = agent.run("Should I buy bonds?")
    assert result.guardrail_triggered is True
