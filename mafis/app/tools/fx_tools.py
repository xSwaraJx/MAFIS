from __future__ import annotations

# Tools to be added in subsequent chunks:
# - get_fx_rate
# - get_fx_historical
# - get_fx_change_pct

import time
from datetime import date, timedelta

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


@tool
def get_fx_rate(base: str, target: str) -> dict:
    """Return the latest FX rate between two currency codes (e.g. USD, EUR)."""
    result = _fetch("/latest", {"from": base, "to": target})
    if "error" in result:
        return result
    try:
        return {"base": base, "target": target, "rate": float(result["rates"][target]), "date": result["date"]}
    except (KeyError, TypeError) as e:
        return {"error": str(e)}


@tool
def get_fx_historical(base: str, target: str, days: int = 30) -> dict:
    """Return a historical FX rate series for a currency pair over the specified number of days."""
    start_date = (date.today() - timedelta(days=days)).isoformat()
    end_date = date.today().isoformat()
    result = _fetch(f"/{start_date}..{end_date}", {"from": base, "to": target})
    if "error" in result:
        return result
    if "rates" not in result:
        return {"error": "No data returned"}
    series = sorted(
        [{"date": d, "rate": float(rates[target])} for d, rates in result["rates"].items() if target in rates],
        key=lambda x: x["date"],
    )
    return {"base": base, "target": target, "days": days, "series": series}


FX_TOOLS: list = [get_fx_rate, get_fx_historical]
