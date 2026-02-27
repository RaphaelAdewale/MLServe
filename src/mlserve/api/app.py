"""FastAPI Control Plane application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mlserve import __version__
from mlserve.api.routes import deployments, health, models
from mlserve.core.database import create_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mlserve.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info(f"Starting MLServe Control Plane v{__version__}")

    # Create database tables (dev only — use Alembic in production)
    await create_tables()
    logger.info("Database tables created")

    yield

    logger.info("Shutting down MLServe Control Plane")


app = FastAPI(
    title="MLServe Control Plane",
    description="Self-service ML model deployment platform",
    version=__version__,
    lifespan=lifespan,
)

# Register route modules
app.include_router(health.router)
app.include_router(models.router)
app.include_router(deployments.router)
