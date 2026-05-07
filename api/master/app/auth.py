from __future__ import annotations

from fastapi import Header, HTTPException, Request, status

from .config import settings


def _bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    return authorization.split(" ", 1)[1].strip()


async def require_user_token(
    request: Request,
    authorization: str | None = Header(default=None),
) -> str:
    token = _bearer(authorization)
    if token not in settings.api_token_set:
        raise HTTPException(status_code=401, detail="invalid api token")
    request.state.api_key = token
    return token


async def require_worker_token(
    request: Request,
    authorization: str | None = Header(default=None),
) -> str:
    token = _bearer(authorization)
    if not settings.worker_token or token != settings.worker_token:
        raise HTTPException(status_code=401, detail="invalid worker token")
    request.state.api_key = token
    return token
