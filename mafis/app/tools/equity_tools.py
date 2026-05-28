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


EQUITY_TOOLS: list = []
