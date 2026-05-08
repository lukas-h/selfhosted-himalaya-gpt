from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .config import settings
from .queue import Registry
from .ratelimit import limiter, rate_limit_envelope
from .routes import health, internal, openai


def create_app() -> FastAPI:
    app = FastAPI(title="himalaya-master", version="0.1.0")

    app.state.settings = settings
    app.state.registry = Registry(
        slugs=settings.slugs,
        max_queue_depth=settings.max_queue_depth,
        max_concurrent_per_worker=settings.max_concurrent_per_worker,
    )
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def _rl(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(status_code=429, content=rate_limit_envelope(exc))

    @app.exception_handler(Exception)
    async def _generic(request: Request, exc: Exception) -> JSONResponse:
        # last-resort handler so users always see OpenAI-shaped errors
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc) or "internal error", "type": "server_error"}},
        )

    app.include_router(health.router)
    app.include_router(openai.router)
    app.include_router(internal.router)
    return app


app = create_app()
