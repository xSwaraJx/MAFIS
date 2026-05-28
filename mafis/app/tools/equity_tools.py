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


@tool
def get_stock_history(ticker: str, period: str) -> dict:
    """Return summary statistics for a stock's OHLCV history over a given period (1mo, 3mo, 6mo, 1y)."""
    if period not in {"1mo", "3mo", "6mo", "1y"}:
        return {"error": "Invalid period. Use 1mo, 3mo, 6mo, or 1y."}
    try:
        hist = _get_ticker(ticker).history(period=period)
        if hist.empty:
            return {"error": "No data returned", "ticker": ticker}
        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        total_return_pct = (end_price / start_price - 1) * 100
        return {
            "ticker": ticker,
            "period": period,
            "data_points": len(hist),
            "start_date": str(hist.index[0].date()),
            "end_date": str(hist.index[-1].date()),
            "start_price": start_price,
            "end_price": end_price,
            "total_return_pct": round(total_return_pct, 4),
        }
    except Exception as e:
        return {"error": str(e)}


EQUITY_TOOLS: list = [get_stock_price, get_stock_history]
