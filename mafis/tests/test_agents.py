from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.schemas import AgentResponse
from app.guardrails.guardrails import GuardedResponse

# Fixture: a valid AgentResponse for use as mock return value
mock_agent_response = AgentResponse(
    agent="TestAgent",
    query="test",
    answer="test answer",
    tools_used=["tool1"],
    data_snapshot={},
    latency_ms=100.0,
    guardrail_triggered=False,
)


# ---------------------------------------------------------------------------
# Agent construction tests (8.2.2)
# ---------------------------------------------------------------------------

def test_rates_agent_initialises():
    with patch("app.agents.rates_agent.ChatOpenAI"):
        from app.agents.rates_agent import RatesAgent
        a = RatesAgent()
    assert hasattr(a, "graph")


def test_fx_agent_initialises():
    with patch("app.agents.fx_agent.ChatOpenAI"):
        from app.agents.fx_agent import FXAgent
        a = FXAgent()
    assert hasattr(a, "graph")


def test_equities_agent_initialises():
    with patch("app.agents.equities_agent.ChatOpenAI"):
        from app.agents.equities_agent import EquitiesAgent
        a = EquitiesAgent()
    assert hasattr(a, "graph")


# ---------------------------------------------------------------------------
# Agent run() output shape tests (8.2.3)
# ---------------------------------------------------------------------------

def _make_mock_graph_result(answer: str = "Test answer"):
    msg = MagicMock()
    msg.content = answer
    msg.tool_calls = []
    return {"messages": [msg]}


def _mock_guardrails(answer: str) -> MagicMock:
    m = MagicMock()
    m.apply.return_value = GuardedResponse(text=answer, triggered=False)
    return m


def test_rates_agent_run_shape():
    with patch("app.agents.rates_agent.ChatOpenAI"):
        from app.agents.rates_agent import RatesAgent
        a = RatesAgent()
    a.graph = MagicMock()
    a.graph.invoke.return_value = _make_mock_graph_result()
    a.guardrails = _mock_guardrails("Test answer")
    r = a.run("test query")
    assert r.answer == "Test answer"
    assert r.agent == "RatesAgent"
    assert r.latency_ms >= 0
    assert isinstance(r.guardrail_triggered, bool)


def test_fx_agent_run_shape():
    with patch("app.agents.fx_agent.ChatOpenAI"):
        from app.agents.fx_agent import FXAgent
        a = FXAgent()
    a.graph = MagicMock()
    a.graph.invoke.return_value = _make_mock_graph_result()
    a.guardrails = _mock_guardrails("Test answer")
    r = a.run("test query")
    assert r.answer == "Test answer"
    assert r.agent == "FXAgent"
    assert r.latency_ms >= 0
    assert isinstance(r.guardrail_triggered, bool)


def test_equities_agent_run_shape():
    with patch("app.agents.equities_agent.ChatOpenAI"):
        from app.agents.equities_agent import EquitiesAgent
        a = EquitiesAgent()
    a.graph = MagicMock()
    a.graph.invoke.return_value = _make_mock_graph_result()
    a.guardrails = _mock_guardrails("Test answer")
    r = a.run("test query")
    assert r.answer == "Test answer"
    assert r.agent == "EquitiesAgent"
    assert r.latency_ms >= 0
    assert isinstance(r.guardrail_triggered, bool)


# ---------------------------------------------------------------------------
# Agent error handling tests (8.2.4)
# ---------------------------------------------------------------------------

def test_agent_run_returns_response_on_executor_error():
    for AgentClass, agent_name, module in [
        ("RatesAgent",    "RatesAgent",    "app.agents.rates_agent"),
        ("FXAgent",       "FXAgent",       "app.agents.fx_agent"),
        ("EquitiesAgent", "EquitiesAgent", "app.agents.equities_agent"),
    ]:
        chat_module = module.replace("_agent", "_agent").rsplit(".", 1)[0] + "." + module.rsplit(".", 1)[1]
        with patch(f"{module}.ChatOpenAI"):
            import importlib
            mod = importlib.import_module(module)
            cls = getattr(mod, AgentClass)
            a = cls()
        a.graph = MagicMock()
        a.graph.invoke.side_effect = Exception("LLM error")
        result = a.run("test query")
        assert isinstance(result, AgentResponse), f"{AgentClass} did not return AgentResponse"
        assert "error" in result.answer.lower(), f"{AgentClass} answer missing 'error'"
