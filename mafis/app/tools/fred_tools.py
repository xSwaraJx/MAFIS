from __future__ import annotations

# Tools to be added in subsequent chunks:
# - get_fed_funds_rate
# - get_treasury_yield
# - get_yield_curve_spread
# - get_cpi_inflation

import fredapi
from langchain_core.tools import tool

from app.config import get_settings


def _get_fred_client() -> fredapi.Fred:
    return fredapi.Fred(api_key=get_settings().fred_api_key)


@tool
def get_fed_funds_rate() -> dict:
    """Return the latest Federal Funds Rate from FRED series FEDFUNDS."""
    try:
        series = _get_fred_client().get_series("FEDFUNDS")
        latest = series.dropna().iloc[-1]
        date = str(series.dropna().index[-1].date())
        return {"series": "FEDFUNDS", "latest_value": float(latest), "date": date, "unit": "Percent"}
    except Exception as e:
        return {"error": str(e), "series": "FEDFUNDS"}


FRED_TOOLS: list = [get_fed_funds_rate]
