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


@tool
def get_treasury_yield(maturity: str) -> dict:
    """Return the latest Treasury yield for a given maturity (2y, 10y, 30y) from FRED."""
    series_map = {"2y": "DGS2", "10y": "DGS10", "30y": "DGS30"}
    if maturity not in series_map:
        return {"error": "Invalid maturity. Use 2y, 10y, or 30y."}
    series_id = series_map[maturity]
    try:
        series = _get_fred_client().get_series(series_id)
        latest = series.dropna().iloc[-1]
        date = str(series.dropna().index[-1].date())
        return {"series": series_id, "maturity": maturity, "latest_value": float(latest), "date": date, "unit": "Percent"}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_yield_curve_spread() -> dict:
    """Return the 10Y-2Y Treasury yield spread and a curve signal from FRED."""
    try:
        client = _get_fred_client()
        s10 = client.get_series("DGS10").dropna()
        s2 = client.get_series("DGS2").dropna()
        ten_year = float(s10.iloc[-1])
        two_year = float(s2.iloc[-1])
        spread = ten_year - two_year
        date = str(s10.index[-1].date())
        if spread < 0:
            signal = "INVERTED (recession signal)"
        elif spread < 0.5:
            signal = "FLAT"
        else:
            signal = "NORMAL"
        return {"spread_bps": round(spread * 100, 2), "ten_year": ten_year, "two_year": two_year, "date": date, "signal": signal}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_cpi_inflation() -> dict:
    """Return year-over-year CPI inflation from FRED series CPIAUCSL."""
    try:
        series = _get_fred_client().get_series("CPIAUCSL", observation_start=None).dropna()
        series = series.iloc[-13:]
        latest_value = float(series.iloc[-1])
        twelve_months_ago = float(series.iloc[0])
        yoy = (latest_value / twelve_months_ago - 1) * 100
        latest_date = str(series.index[-1].date())
        return {"series": "CPIAUCSL", "yoy_change_pct": round(yoy, 4), "latest_date": latest_date, "latest_value": latest_value}
    except Exception as e:
        return {"error": str(e)}


FRED_TOOLS: list = [get_fed_funds_rate, get_treasury_yield, get_yield_curve_spread, get_cpi_inflation]
