from __future__ import annotations

# Tools under test:
# FRED:    get_fed_funds_rate, get_treasury_yield, get_yield_curve_spread, get_cpi_inflation
# FX:      get_fx_rate, get_fx_historical, get_fx_change_pct
# Equity:  get_stock_price, get_stock_history, get_technical_indicators, get_market_summary

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# FRED tool tests (8.1.2)
# ---------------------------------------------------------------------------

def test_get_fed_funds_rate_success():
    from app.tools.fred_tools import get_fed_funds_rate
    series = pd.Series([5.33], index=pd.to_datetime(["2024-01-01"]))
    with patch("app.tools.fred_tools._get_fred_client") as mock_client:
        mock_client.return_value.get_series.return_value = series
        result = get_fed_funds_rate.invoke({})
    assert "latest_value" in result
    assert isinstance(result["latest_value"], float)


def test_get_fed_funds_rate_error():
    from app.tools.fred_tools import get_fed_funds_rate
    with patch("app.tools.fred_tools._get_fred_client") as mock_client:
        mock_client.return_value.get_series.side_effect = Exception("API down")
        result = get_fed_funds_rate.invoke({})
    assert "error" in result
    assert "API down" in result["error"]


def test_get_treasury_yield_invalid_maturity():
    from app.tools.fred_tools import get_treasury_yield
    result = get_treasury_yield.invoke({"maturity": "5y"})
    assert "error" in result


def test_get_yield_curve_spread_inverted():
    from app.tools.fred_tools import get_yield_curve_spread
    s10 = pd.Series([3.5], index=pd.to_datetime(["2024-01-01"]))
    s2 = pd.Series([4.0], index=pd.to_datetime(["2024-01-01"]))
    with patch("app.tools.fred_tools._get_fred_client") as mock_client:
        mock_client.return_value.get_series.side_effect = [s10, s2]
        result = get_yield_curve_spread.invoke({})
    assert result["signal"] == "INVERTED (recession signal)"


# ---------------------------------------------------------------------------
# FX tool tests (8.1.3)
# ---------------------------------------------------------------------------

def test_get_fx_rate_success():
    from app.tools.fx_tools import get_fx_rate
    mock_response = MagicMock()
    mock_response.json.return_value = {"rates": {"EUR": 0.92}, "date": "2024-01-01"}
    mock_response.raise_for_status.return_value = None
    with patch("app.tools.fx_tools.httpx.get", return_value=mock_response):
        result = get_fx_rate.invoke({"base": "USD", "target": "EUR"})
    assert result["rate"] == 0.92


def test_get_fx_rate_unsupported_currency():
    from app.tools.fx_tools import get_fx_rate
    result = get_fx_rate.invoke({"base": "USD", "target": "DOGE"})
    assert "error" in result


def test_get_fx_change_pct_insufficient_data():
    from app.tools.fx_tools import get_fx_change_pct
    with patch("app.tools.fx_tools.get_fx_historical") as mock_hist:
        mock_hist.invoke.return_value = {"base": "USD", "target": "EUR", "days": 30, "series": [{"date": "2024-01-01", "rate": 0.9}]}
        result = get_fx_change_pct.invoke({"base": "USD", "target": "EUR", "days": 30})
    assert "error" in result


# ---------------------------------------------------------------------------
# Equity tool tests (8.1.4)
# ---------------------------------------------------------------------------

def test_get_stock_price_success():
    from app.tools.equity_tools import get_stock_price
    mock_fast_info = MagicMock()
    mock_fast_info.last_price = 150.0
    mock_fast_info.previous_close = 145.0
    mock_fast_info.currency = "USD"
    with patch("app.tools.equity_tools._get_ticker") as mock_ticker:
        mock_ticker.return_value.fast_info = mock_fast_info
        result = get_stock_price.invoke({"ticker": "AAPL"})
    assert result["day_change_pct"] > 0
    assert result["current_price"] == 150.0


def test_get_stock_history_invalid_period():
    from app.tools.equity_tools import get_stock_history
    result = get_stock_history.invoke({"ticker": "AAPL", "period": "2w"})
    assert "error" in result


def test_get_technical_indicators_insufficient_data():
    from app.tools.equity_tools import get_technical_indicators
    small_df = pd.DataFrame({"Close": [100.0] * 10})
    small_df.index = pd.date_range("2024-01-01", periods=10)
    with patch("app.tools.equity_tools._get_ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = small_df
        result = get_technical_indicators.invoke({"ticker": "AAPL"})
    assert "error" in result


# ---------------------------------------------------------------------------
# All tools return dict on forced error (8.1.5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool,invoke_args,patch_target,exception", [
    ("get_fed_funds_rate",        {},                                   "app.tools.fred_tools._get_fred_client",   Exception("forced")),
    ("get_treasury_yield",        {"maturity": "10y"},                  "app.tools.fred_tools._get_fred_client",   Exception("forced")),
    ("get_yield_curve_spread",    {},                                   "app.tools.fred_tools._get_fred_client",   Exception("forced")),
    ("get_cpi_inflation",         {},                                   "app.tools.fred_tools._get_fred_client",   Exception("forced")),
    ("get_stock_price",           {"ticker": "AAPL"},                  "app.tools.equity_tools._get_ticker",      Exception("forced")),
    ("get_stock_history",         {"ticker": "AAPL", "period": "1mo"}, "app.tools.equity_tools._get_ticker",      Exception("forced")),
    ("get_technical_indicators",  {"ticker": "AAPL"},                  "app.tools.equity_tools._get_ticker",      Exception("forced")),
])
def test_all_tools_return_dict_not_exception(tool, invoke_args, patch_target, exception):
    import importlib
    module_name, tool_name = patch_target.rsplit(".", 1)
    if "fred" in module_name:
        from app.tools import fred_tools as mod
        tool_fn = getattr(mod, tool)
    else:
        from app.tools import equity_tools as mod
        tool_fn = getattr(mod, tool)

    with patch(patch_target) as mock_fn:
        mock_fn.side_effect = exception
        result = tool_fn.invoke(invoke_args)

    assert isinstance(result, dict), f"{tool} did not return a dict"
    assert "error" in result, f"{tool} result missing 'error' key"


def test_get_fx_rate_network_error():
    from app.tools.fx_tools import get_fx_rate
    import httpx
    with patch("app.tools.fx_tools.httpx.get", side_effect=Exception("network error")):
        result = get_fx_rate.invoke({"base": "USD", "target": "EUR"})
    assert isinstance(result, dict)
    assert "error" in result
