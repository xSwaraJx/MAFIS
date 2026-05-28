from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas import AgentResponse, OrchestratorResponse

# ---------------------------------------------------------------------------
# Shared mock responses
# ---------------------------------------------------------------------------

_mock_agent = AgentResponse(
    agent="RatesAgent", query="test", answer="test answer",
    tools_used=[], data_snapshot={}, latency_ms=100.0, guardrail_triggered=False,
)

_mock_orch = OrchestratorResponse(
    query="test", rates=_mock_agent, fx=_mock_agent, equities=_mock_agent,
    total_latency_ms=300.0, any_guardrail_triggered=False,
)


class _MockOrchestrator:
    def run_agent(self, stream: str, query: str) -> AgentResponse:
        return _mock_agent

    async def run_all(self, query: str) -> OrchestratorResponse:
        return _mock_orch


# ---------------------------------------------------------------------------
# Client fixture (8.4.1)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client():
    app.state.orchestrator = _MockOrchestrator()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint test (8.4.2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "environment" in data


# ---------------------------------------------------------------------------
# Single agent endpoint tests (8.4.3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_rates_returns_agent_response(client):
    response = await client.post("/analyze/rates", json={"query": "What is the fed funds rate?"})
    assert response.status_code == 200
    data = response.json()
    for key in ("agent", "query", "answer", "tools_used", "latency_ms", "guardrail_triggered"):
        assert key in data


@pytest.mark.asyncio
async def test_analyze_fx_returns_agent_response(client):
    response = await client.post("/analyze/fx", json={"query": "What is USD to EUR rate?"})
    assert response.status_code == 200
    data = response.json()
    for key in ("agent", "query", "answer", "tools_used", "latency_ms", "guardrail_triggered"):
        assert key in data


@pytest.mark.asyncio
async def test_analyze_equities_returns_agent_response(client):
    response = await client.post("/analyze/equities", json={"query": "What is Apple stock price?"})
    assert response.status_code == 200
    data = response.json()
    for key in ("agent", "query", "answer", "tools_used", "latency_ms", "guardrail_triggered"):
        assert key in data


# ---------------------------------------------------------------------------
# Full orchestrator endpoint test (8.4.4)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_all_returns_orchestrator_response(client):
    response = await client.post("/analyze", json={"query": "Market overview", "stream": "all"})
    assert response.status_code == 200
    data = response.json()
    for key in ("query", "rates", "fx", "equities", "total_latency_ms", "any_guardrail_triggered"):
        assert key in data


# ---------------------------------------------------------------------------
# Validation and error tests (8.4.5)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_query_returns_422(client):
    response = await client.post("/analyze/rates", json={"query": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_stream_returns_422(client):
    response = await client.post("/analyze", json={"query": "test", "stream": "crypto"})
    assert response.status_code == 422
