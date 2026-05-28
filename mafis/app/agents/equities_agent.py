from __future__ import annotations

import time

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import get_settings
from app.tools.equity_tools import EQUITY_TOOLS
from app.schemas import AgentResponse
from app.guardrails.guardrails import GuardrailsWrapper

EQUITIES_SYSTEM_PROMPT = (
    "You are an equity analyst covering US-listed securities.\n"
    "You have access to live price data, historical OHLCV, and technical indicators via yfinance.\n"
    "Rules you must always follow:\n"
    "1. Clearly distinguish between price observation (fact) and technical signal (interpretation).\n"
    "2. Never state a price target or make a buy/sell recommendation.\n"
    "3. If asked for investment guidance, include this disclaimer: "
    "'This is not financial advice. Consult a registered advisor.'\n"
    "4. Always include the data retrieval date in your response."
)


class EquitiesAgent:
    def __init__(self) -> None:
        llm = ChatOpenAI(
            model=get_settings().model_name,
            temperature=0,
            api_key=get_settings().openai_api_key,
        )
        self.graph = create_react_agent(llm, EQUITY_TOOLS, prompt=EQUITIES_SYSTEM_PROMPT)
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
                agent="EquitiesAgent",
                query=query,
                answer=guarded.text,
                tools_used=tools_used,
                data_snapshot={},
                latency_ms=latency_ms,
                guardrail_triggered=guarded.triggered,
            )
        except Exception as e:
            return AgentResponse(
                agent="EquitiesAgent",
                query=query,
                answer=f"Agent error: {e}",
                tools_used=[],
                data_snapshot={},
                latency_ms=0.0,
                guardrail_triggered=False,
            )


__all__ = ["EquitiesAgent"]
