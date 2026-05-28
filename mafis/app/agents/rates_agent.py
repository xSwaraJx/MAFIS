from __future__ import annotations

import time

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from app.config import get_settings
from app.tools.fred_tools import FRED_TOOLS
from app.schemas import AgentResponse
from app.guardrails.guardrails import GuardrailsWrapper

RATES_SYSTEM_PROMPT = (
    "You are a rates analyst specialising in monetary policy and fixed income markets.\n"
    "You have access to live FRED data for interest rates, treasury yields, and inflation.\n"
    "Rules you must always follow:\n"
    "1. Cite the data source and observation date for every numerical claim.\n"
    "2. Use probabilistic language for any forward-looking statement (e.g. 'may', 'could', 'suggests').\n"
    "3. Never make a specific rate forecast presented as certainty.\n"
    "4. Never give investment advice or recommend buying/selling any security."
)


class RatesAgent:
    def __init__(self) -> None:
        llm = ChatGroq(
            model=get_settings().model_name,
            temperature=0,
            api_key=get_settings().groq_api_key,
        )
        self.graph = create_react_agent(llm, FRED_TOOLS, prompt=RATES_SYSTEM_PROMPT)
        self.guardrails = GuardrailsWrapper()

    def run(self, query: str) -> AgentResponse:
        try:
            start = time.time()
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"recursion_limit": 10},
            )
            latency_ms = (time.time() - start) * 1000
            answer = result["messages"][-1].content
            tools_used = [
                tc["name"]
                for msg in result["messages"]
                if hasattr(msg, "tool_calls") and msg.tool_calls
                for tc in msg.tool_calls
            ]
            guarded = self.guardrails.apply(query, answer)
            return AgentResponse(
                agent="RatesAgent",
                query=query,
                answer=guarded.text,
                tools_used=tools_used,
                data_snapshot={},
                latency_ms=latency_ms,
                guardrail_triggered=guarded.triggered,
            )
        except Exception as e:
            return AgentResponse(
                agent="RatesAgent",
                query=query,
                answer=f"Agent error: {e}",
                tools_used=[],
                data_snapshot={},
                latency_ms=0.0,
                guardrail_triggered=False,
            )


__all__ = ["RatesAgent"]
