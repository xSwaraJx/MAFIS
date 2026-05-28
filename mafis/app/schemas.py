from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    query: str = Field(min_length=1)
    stream: Literal["rates", "fx", "equities", "all"] = "all"


class AgentResponse(BaseModel):
    agent: str
    query: str
    answer: str
    tools_used: list[str]
    data_snapshot: dict[str, Any]
    latency_ms: float
    guardrail_triggered: bool = False


class OrchestratorResponse(BaseModel):
    query: str
    rates: AgentResponse | None = None
    fx: AgentResponse | None = None
    equities: AgentResponse | None = None
    total_latency_ms: float
    any_guardrail_triggered: bool = False


class HealthResponse(BaseModel):
    status: str
    model: str
    environment: str
