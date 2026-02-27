"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MLSERVE_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    database_url: str = "sqlite+aiosqlite:///./mlserve.db"

    artifact_dir: Path = Path("./mlserve_artifacts")

    docker_registry: str = "localhost:5000"
    docker_network: str = "mlserve-network"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    traefik_url: str = "http://localhost"


settings = Settings()
