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


@tool
def get_technical_indicators(ticker: str) -> dict:
    """Return SMA-20, SMA-50, RSI-14, 52-week high/low technical indicators for a ticker."""
    try:
        hist = _get_ticker(ticker).history(period="1y")
        if hist.empty or len(hist) < 50:
            return {"error": "Insufficient data for indicators"}
        closes = hist["Close"]

        sma_20 = round(float(closes.iloc[-20:].mean()), 2)
        sma_50 = round(float(closes.iloc[-50:].mean()), 2)
        week_52_high = round(float(closes.max()), 2)
        week_52_low = round(float(closes.min()), 2)

        delta = closes.diff().dropna()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = float(gains.iloc[:14].mean())
        avg_loss = float(losses.iloc[:14].mean())
        for g, l in zip(gains.iloc[14:], losses.iloc[14:]):
            avg_gain = (avg_gain * 13 + g) / 14
            avg_loss = (avg_loss * 13 + l) / 14
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        rsi_14 = round(100 - (100 / (1 + rs)), 2)

        return {
            "ticker": ticker,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "rsi_14": rsi_14,
            "week_52_high": week_52_high,
            "week_52_low": week_52_low,
            "computation_date": date.today().isoformat(),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


@tool
def get_market_summary() -> dict:
    """Return current price and day change % for SPY, QQQ, and DIA as a market summary."""
    summary = []
    for sym in ("SPY", "QQQ", "DIA"):
        result = get_stock_price.invoke({"ticker": sym})
        if "error" in result:
            summary.append({"ticker": sym, "error": result["error"]})
        else:
            summary.append({"ticker": sym, "current_price": result["current_price"], "day_change_pct": result["day_change_pct"]})
    return {"market_summary": summary, "as_of": date.today().isoformat()}


EQUITY_TOOLS: list = [get_stock_price, get_stock_history, get_technical_indicators, get_market_summary]
