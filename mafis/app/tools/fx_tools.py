from __future__ import annotations

# Tools to be added in subsequent chunks:
# - get_fx_rate
# - get_fx_historical
# - get_fx_change_pct

import time

import httpx
from langchain_core.tools import tool

BASE_URL = "https://api.frankfurter.dev/v1"


def _fetch(endpoint: str, params: dict) -> dict:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = httpx.get(BASE_URL + endpoint, params=params, timeout=5.0)
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_exc = e
            if attempt < 2:
                time.sleep(0.5)
        except Exception as e:
            return {"error": str(e)}
    return {"error": f"Request failed after 3 retries: {last_exc}"}


FX_TOOLS: list = []
