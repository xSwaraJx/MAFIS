from __future__ import annotations

import functools

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    fred_api_key: str
    environment: str = "development"
    log_level: str = "INFO"
    model_name: str = "gpt-4o"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()
