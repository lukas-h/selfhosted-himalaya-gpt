from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_tokens: str = Field(default="", description="Comma-separated bearer tokens accepted on /v1/*")
    worker_token: str = Field(default="", description="Single shared secret for /internal/*")

    model_slugs: str = Field(default="himalaya-bf16,himalaya-q8,himalaya-q4")

    max_concurrent_per_worker: int = 2
    max_queue_depth: int = 32
    rate_limit_rpm: int = 60
    user_request_timeout_s: int = 300
    worker_poll_timeout_s: int = 55

    @property
    def api_token_set(self) -> set[str]:
        return {t.strip() for t in self.api_tokens.split(",") if t.strip()}

    @property
    def slugs(self) -> list[str]:
        return [s.strip() for s in self.model_slugs.split(",") if s.strip()]


settings = Settings()
