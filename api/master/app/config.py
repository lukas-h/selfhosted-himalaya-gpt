from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_tokens: str = Field(default="", description="Comma-separated bearer tokens accepted on /v1/*")
    worker_token: str = Field(default="", description="Single shared secret for /internal/*")

    model_slugs: str = Field(default="himalaya-bf16,himalaya-q8,himalaya-q4")

    max_concurrent_per_worker: int = 3
    max_queue_depth: int = 32
    # Sustained rate cap per token. 1200/min (20 req/s) is enough headroom
    # for typical agentic workloads (multiple models + retries + tool loops)
    # against a single trusted token. Lower it if you expose API_TOKENS to
    # untrusted clients.
    rate_limit_rpm: int = 1200
    # Optional burst cap per token, in requests-per-second. The bucket
    # resets every second so brief spikes are absorbed but a stuck client
    # still backs off. Set to 0 to disable the per-second cap and only
    # enforce rate_limit_rpm.
    rate_limit_burst_rps: int = 30
    # Master gives up on a queued job after this many seconds and frees the
    # slot. Keep this close to typical client timeouts (curl/httpx default
    # 30s) — longer values let abandoned-curl phantom jobs accumulate and
    # saturate the per-slug queue.
    user_request_timeout_s: int = 30
    worker_poll_timeout_s: int = 55

    @property
    def api_token_set(self) -> set[str]:
        return {t.strip() for t in self.api_tokens.split(",") if t.strip()}

    @property
    def slugs(self) -> list[str]:
        return [s.strip() for s in self.model_slugs.split(",") if s.strip()]


settings = Settings()
