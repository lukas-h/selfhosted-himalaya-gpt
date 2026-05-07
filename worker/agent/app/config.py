from __future__ import annotations

import socket

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    master_base_url: str = Field(..., description="e.g. https://api.example.com")
    worker_token: str = Field(...)

    llama_base_url: str = Field(default="http://llama:8080")
    llama_internal_key: str = Field(default="")

    model_slugs: str = Field(default="himalaya-bf16,himalaya-q8,himalaya-q4")
    max_concurrent: int = 2
    poll_timeout_s: int = 55
    worker_id: str = Field(default_factory=lambda: socket.gethostname())

    @property
    def slugs(self) -> list[str]:
        return [s.strip() for s in self.model_slugs.split(",") if s.strip()]


settings = Settings()
