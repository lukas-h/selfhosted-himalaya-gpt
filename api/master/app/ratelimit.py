from __future__ import annotations

import hashlib

from fastapi import Request
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from .config import settings


def _key_func(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        return hashlib.sha256(token.encode()).hexdigest()[:16]
    return request.client.host if request.client else "anon"


limiter = Limiter(key_func=_key_func, default_limits=[])


def rate_limit_envelope(exc: RateLimitExceeded):
    return {
        "error": {
            "message": f"rate limit exceeded: {exc.detail}",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }
    }
