from __future__ import annotations

import socket

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    master_base_url: str = Field(..., description="e.g. https://api.example.com")
    worker_token: str = Field(...)

    # URL template for the per-quant llama-server containers. `{quant}` is
    # replaced by the slug with the `himalaya-` prefix stripped — so the
    # default expects compose services named `llama-bf16`, `llama-q8`, etc.
    # Set to a literal URL (no `{quant}`) to point every slug at the same
    # llama-server (legacy single-container deployments).
    llama_url_template: str = Field(default="http://llama-{quant}:8080")
    llama_internal_key: str = Field(default="")

    model_slugs: str = Field(default="himalaya-bf16,himalaya-q8,himalaya-q4")
    max_concurrent: int = 2
    poll_timeout_s: int = 55
    worker_id: str = Field(default_factory=lambda: socket.gethostname())

    @property
    def slugs(self) -> list[str]:
        return [s.strip() for s in self.model_slugs.split(",") if s.strip()]

    def llama_url_for(self, slug: str) -> str:
        quant = slug.removeprefix("himalaya-") or slug
        return self.llama_url_template.format(quant=quant)


settings = Settings()
