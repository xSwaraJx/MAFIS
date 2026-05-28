from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nemoguardrails import RailsConfig, LLMRails

GUARDRAILS_DIR = Path(__file__).parent


@dataclass
class GuardedResponse:
    text: str
    triggered: bool
    rail: str | None = None


class GuardrailsWrapper:
    def __init__(self) -> None:
        try:
            config = RailsConfig.from_path(str(GUARDRAILS_DIR))
            self.rails = LLMRails(config)
        except Exception as e:
            print(f"Warning: guardrails failed to load: {e}")
            self.rails = None

    def apply(self, query: str, agent_response: str) -> GuardedResponse:
        if self.rails is None:
            return GuardedResponse(text=agent_response, triggered=False)
        try:
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": agent_response},
            ]
            raw = self.rails.generate(messages=messages)
            output = raw if isinstance(raw, str) else raw.get("content", agent_response)
            triggered = output != agent_response
            return GuardedResponse(
                text=output,
                triggered=triggered,
                rail="financial" if triggered else None,
            )
        except Exception:
            return GuardedResponse(text=agent_response, triggered=False)


__all__ = ["GuardrailsWrapper", "GuardedResponse"]
