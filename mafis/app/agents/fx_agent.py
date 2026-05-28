from __future__ import annotations

import time

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from app.config import get_settings
from app.tools.fx_tools import FX_TOOLS
from app.schemas import AgentResponse
from app.guardrails.guardrails import GuardrailsWrapper

FX_SYSTEM_PROMPT = (
    "You are a foreign exchange analyst covering G10 currency pairs.\n"
    "You have access to live FX rate data from the Frankfurter API.\n"
    "Rules you must always follow:\n"
    "1. Always state the timestamp of the rate data you are referencing.\n"
    "2. Never recommend specific trade entry or exit points.\n"
    "3. Flag explicitly if a currency pair is not in your supported list.\n"
    "4. Use probabilistic language for any directional commentary."
)


class FXAgent:
    def __init__(self) -> None:
        llm = ChatGroq(
            model=get_settings().model_name,
            temperature=0,
            api_key=get_settings().groq_api_key,
        )
        self.graph = create_react_agent(llm, FX_TOOLS, prompt=FX_SYSTEM_PROMPT)
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
                agent="FXAgent",
                query=query,
                answer=guarded.text,
                tools_used=tools_used,
                data_snapshot={},
                latency_ms=latency_ms,
                guardrail_triggered=guarded.triggered,
            )
        except Exception as e:
            return AgentResponse(
                agent="FXAgent",
                query=query,
                answer=f"Agent error: {e}",
                tools_used=[],
                data_snapshot={},
                latency_ms=0.0,
                guardrail_triggered=False,
            )


__all__ = ["FXAgent"]
