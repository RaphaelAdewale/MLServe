"""Health check routes."""

from fastapi import APIRouter

from mlserve import __version__
from mlserve.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    """Control Plane health check."""
    return HealthResponse(status="ok", version=__version__)
