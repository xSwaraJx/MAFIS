from __future__ import annotations

import os
import pytest


def pytest_configure(config):
    os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
    os.environ.setdefault("FRED_API_KEY", "test-fred-key")
    os.environ.setdefault("MODEL_NAME", "llama-3.3-70b-versatile")
    # Clear cached settings so test env vars take effect
    from app.config import get_settings
    get_settings.cache_clear()
