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


FRED_TOOLS: list = []
