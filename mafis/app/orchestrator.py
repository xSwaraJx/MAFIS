from __future__ import annotations

import asyncio
import time
from typing import Union

from app.agents.rates_agent import RatesAgent
from app.agents.fx_agent import FXAgent
from app.agents.equities_agent import EquitiesAgent
from app.schemas import AgentResponse, OrchestratorResponse


class MAFISOrchestrator:
    def __init__(self) -> None:
        self.rates_agent = RatesAgent()
        self.fx_agent = FXAgent()
        self.equities_agent = EquitiesAgent()

    def run_agent(self, stream: str, query: str) -> Union[AgentResponse, OrchestratorResponse]:
        if stream == "all":
            return asyncio.run(self.run_all(query))
        if stream == "rates":
            return self.rates_agent.run(query)
        if stream == "fx":
            return self.fx_agent.run(query)
        if stream == "equities":
            return self.equities_agent.run(query)
        return AgentResponse(
            agent="Orchestrator",
            query=query,
            answer=f"Unknown stream: {stream}",
            tools_used=[],
            data_snapshot={},
            latency_ms=0.0,
            guardrail_triggered=False,
        )

    async def run_all(self, query: str) -> OrchestratorResponse:
        start = time.time()
        tasks = [
            asyncio.to_thread(self.rates_agent.run, query),
            asyncio.to_thread(self.fx_agent.run, query),
            asyncio.to_thread(self.equities_agent.run, query),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_latency_ms = (time.time() - start) * 1000

        rates = results[0] if not isinstance(results[0], Exception) else None
        fx = results[1] if not isinstance(results[1], Exception) else None
        equities = results[2] if not isinstance(results[2], Exception) else None

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"Agent {['rates', 'fx', 'equities'][i]} error: {r}")

        any_guardrail = any(
            r.guardrail_triggered for r in [rates, fx, equities] if r is not None
        )

        return OrchestratorResponse(
            query=query,
            rates=rates,
            fx=fx,
            equities=equities,
            total_latency_ms=total_latency_ms,
            any_guardrail_triggered=any_guardrail,
        )


__all__ = ["MAFISOrchestrator"]
