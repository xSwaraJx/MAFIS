from __future__ import annotations

# Tools to be added in subsequent chunks:
# - get_stock_price
# - get_stock_history
# - get_technical_indicators
# - get_market_summary

from datetime import date

import yfinance
from langchain_core.tools import tool


def _get_ticker(symbol: str) -> yfinance.Ticker:
    return yfinance.Ticker(symbol)


@tool
def get_stock_price(ticker: str) -> dict:
    """Return the current price, previous close, day change %, and currency for a stock ticker."""
    try:
        info = _get_ticker(ticker).fast_info
        current_price = float(info.last_price)
        previous_close = float(info.previous_close)
        day_change_pct = (current_price / previous_close - 1) * 100
        return {
            "ticker": ticker,
            "current_price": current_price,
            "previous_close": previous_close,
            "day_change_pct": round(day_change_pct, 4),
            "currency": info.currency,
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


EQUITY_TOOLS: list = [get_stock_price]
