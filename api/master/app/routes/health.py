from __future__ import annotations

from fastapi import APIRouter, Request, Response

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> Response:
    registry = request.app.state.registry
    settings = request.app.state.settings
    if registry.any_recent_poll(max_age_s=settings.worker_poll_timeout_s * 2):
        return Response(content='{"status":"ready"}', media_type="application/json")
    return Response(
        content='{"status":"no recent worker poll"}',
        media_type="application/json",
        status_code=503,
    )
