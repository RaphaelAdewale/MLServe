"""FastAPI dependency injection."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from mlserve.core.config import Settings, settings
from mlserve.core.database import async_session
from mlserve.services.model_service import ModelService


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session per request."""
    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def get_model_service(db: AsyncSession) -> ModelService:
    """Provide a ModelService instance with a DB session."""
    return ModelService(db)


def get_settings() -> Settings:
    """Provide application settings."""
    return settings
