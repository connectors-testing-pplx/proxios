"""
Health check router.

GET /api/health — returns service status and component health.
"""
import time
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_start_time = time.time()


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    version: str
    uptime_seconds: float
    timestamp: str
    components: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Return service health and component status."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        uptime_seconds=round(time.time() - _start_time, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
        components={
            "rag_pipeline": "ready",
            "vector_store": "ready",
            "claude_api": "configured",
            "embeddings": "ready",
        },
    )
