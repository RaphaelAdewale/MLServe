"""
Model Service — orchestrates the full deploy pipeline.

Coordinates: framework detection → artifact storage → DB records →
image build → container deploy → health check.
"""

import hashlib
import logging
import shutil
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from mlserve.core.config import settings
from mlserve.core.exceptions import (
    BuildError,
    DeploymentError,
    FrameworkDetectionError,
    ModelNotFoundError,
)
from mlserve.core.models import Deployment, DeploymentEvent, Model, ModelVersion
from mlserve.services.build_service import BuildService
from mlserve.services.deploy_service import DeployService

logger = logging.getLogger("mlserve.services.model")

# Framework detection by file extension
FRAMEWORK_EXTENSIONS: dict[str, str] = {
    ".pkl": "sklearn",
    ".joblib": "sklearn",
    ".onnx": "onnx",
}

# Supported frameworks
SUPPORTED_FRAMEWORKS = {"sklearn", "onnx"}


class ModelService:
    """Orchestrates the full model registration and deployment pipeline."""

    def __init__(self, db: AsyncSession):
        self._db = db
        self._build_service = BuildService()
        self._deploy_service = DeployService()

    async def register_and_deploy(
        self,
        name: str,
        model_path: Path,
        framework: str | None = None,
        replicas: int = 1,
    ) -> dict:
        """
        Full pipeline: detect → store → register → build → deploy → health check.

        Args:
            name: Deployment name (e.g., "fraud-detector").
            model_path: Path to the model file.
            framework: ML framework. Auto-detected if None.
            replicas: Number of replicas (future use, currently 1).

        Returns:
            Dict with deployment info (name, version, endpoint_url, status).

        Raises:
            FrameworkDetectionError: If framework can't be determined.
            BuildError: If Docker image build fails.
            DeploymentError: If container deployment fails.
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # 1. Detect framework
        if framework is None:
            framework = self.detect_framework(model_path)
        elif framework not in SUPPORTED_FRAMEWORKS:
            raise FrameworkDetectionError(
                f"Unsupported framework: '{framework}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_FRAMEWORKS))}"
            )

        logger.info(f"Deploying model '{name}' (framework={framework})")

        # 2. Compute checksum
        checksum = self._compute_checksum(model_path)
        file_size = model_path.stat().st_size

        # 3. Get or create the Model record
        model_record = await self._get_or_create_model(name, framework)

        # 4. Determine next version number
        version = await self._next_version(model_record.id)

        # 5. Store artifact on disk
        artifact_path = self._store_artifact(name, version, model_path)

        # 6. Create ModelVersion record
        model_version = ModelVersion(
            model_id=model_record.id,
            version=version,
            artifact_path=str(artifact_path),
            checksum_sha256=checksum,
            file_size_bytes=file_size,
        )
        self._db.add(model_version)
        await self._db.flush()

        # 7. Create Deployment record (status: building)
        deployment = await self._get_or_create_deployment(name, model_version.id)
        deployment.status = "building"
        await self._add_event(deployment.id, "building", "Building container image")
        await self._db.flush()

        try:
            # 8. Build Docker image
            image_tag = self._build_service.build_image(
                model_name=name,
                version=version,
                framework=framework,
                model_artifact_path=artifact_path,
            )

            deployment.container_image = image_tag
            deployment.status = "deploying"
            await self._add_event(
                deployment.id, "deploying",
                f"Starting container (image: {image_tag})",
            )
            await self._db.flush()

            # 9. Find an available port and deploy
            port = self._deploy_service.find_available_port()
            container_id = self._deploy_service.deploy(
                name=name,
                image_tag=image_tag,
                framework=framework,
                version=version,
                port=port,
            )

            deployment.container_id = container_id
            deployment.port = port
            await self._db.flush()

            # 10. Wait for healthy
            self._deploy_service.wait_for_healthy(name, timeout=120)

            # 11. Update to running
            endpoint_url = f"{settings.traefik_url}/models/{name}/predict"
            deployment.status = "running"
            deployment.endpoint_url = endpoint_url
            await self._add_event(deployment.id, "running", "Health check passed")
            await self._db.commit()

            logger.info(f"Model '{name}' v{version} deployed at {endpoint_url}")

            return {
                "name": name,
                "version": version,
                "framework": framework,
                "status": "running",
                "endpoint_url": endpoint_url,
                "container_image": image_tag,
                "port": port,
            }

        except (BuildError, DeploymentError) as e:
            deployment.status = "failed"
            await self._add_event(deployment.id, "failed", str(e))
            await self._db.commit()
            raise

    async def get_deployment(self, name: str) -> dict:
        """Get deployment details by name."""
        result = await self._db.execute(
            select(Deployment).where(Deployment.name == name)
        )
        deployment = result.scalar_one_or_none()
        if deployment is None:
            raise ModelNotFoundError(f"No deployment found with name '{name}'")

        # Get model and version info
        version_result = await self._db.execute(
            select(ModelVersion).where(ModelVersion.id == deployment.model_version_id)
        )
        model_version = version_result.scalar_one_or_none()

        model_result = await self._db.execute(
            select(Model).where(Model.id == model_version.model_id)
        )
        model_record = model_result.scalar_one_or_none()

        # Get events
        events_result = await self._db.execute(
            select(DeploymentEvent)
            .where(DeploymentEvent.deployment_id == deployment.id)
            .order_by(DeploymentEvent.created_at)
        )
        events = events_result.scalars().all()

        # Try to get live container status
        container_status = None
        try:
            container_info = self._deploy_service.get_status(name)
            container_status = {
                "container_id": container_info.container_id,
                "status": container_info.status,
                "health": container_info.health,
            }
        except DeploymentError:
            pass

        return {
            "name": deployment.name,
            "version": model_version.version if model_version else None,
            "framework": model_record.framework if model_record else None,
            "status": deployment.status,
            "endpoint_url": deployment.endpoint_url,
            "container_image": deployment.container_image,
            "port": deployment.port,
            "created_at": deployment.created_at.isoformat() if deployment.created_at else None,
            "updated_at": deployment.updated_at.isoformat() if deployment.updated_at else None,
            "container": container_status,
            "events": [
                {
                    "type": e.event_type,
                    "message": e.message,
                    "timestamp": e.created_at.isoformat() if e.created_at else None,
                }
                for e in events
            ],
        }

    async def list_deployments(self) -> list[dict]:
        """List all deployments with their current status."""
        result = await self._db.execute(select(Deployment).order_by(Deployment.created_at.desc()))
        deployments = result.scalars().all()

        items = []
        for d in deployments:
            version_result = await self._db.execute(
                select(ModelVersion).where(ModelVersion.id == d.model_version_id)
            )
            mv = version_result.scalar_one_or_none()

            model_result = await self._db.execute(
                select(Model).where(Model.id == mv.model_id)
            ) if mv else None
            model_record = model_result.scalar_one_or_none() if model_result else None

            items.append({
                "name": d.name,
                "version": mv.version if mv else None,
                "framework": model_record.framework if model_record else None,
                "status": d.status,
                "endpoint_url": d.endpoint_url,
                "port": d.port,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            })

        return items

    async def delete_deployment(self, name: str) -> None:
        """Tear down a deployment and remove DB records."""
        # Teardown container (ignore if not running)
        try:
            self._deploy_service.teardown(name)
        except DeploymentError:
            logger.warning(f"Container for '{name}' could not be removed (may not exist)")

        # Remove deployment from DB
        result = await self._db.execute(
            select(Deployment).where(Deployment.name == name)
        )
        deployment = result.scalar_one_or_none()
        if deployment:
            await self._db.delete(deployment)
            await self._db.commit()
            logger.info(f"Deployment '{name}' deleted")
        else:
            raise ModelNotFoundError(f"No deployment found with name '{name}'")

    @staticmethod
    def detect_framework(model_path: Path) -> str:
        """
        Auto-detect the ML framework from the file extension.

        Args:
            model_path: Path to the model file.

        Returns:
            Framework name (e.g., "sklearn", "onnx").

        Raises:
            FrameworkDetectionError: If the framework cannot be determined.
        """
        suffix = model_path.suffix.lower()
        framework = FRAMEWORK_EXTENSIONS.get(suffix)

        if framework is None:
            supported_exts = ", ".join(sorted(FRAMEWORK_EXTENSIONS.keys()))
            raise FrameworkDetectionError(
                f"Cannot auto-detect framework for '{suffix}' files. "
                f"Supported extensions: {supported_exts}. "
                f"Use --framework to specify explicitly."
            )

        return framework

    @staticmethod
    def _compute_checksum(file_path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _store_artifact(self, name: str, version: int, model_path: Path) -> Path:
        """Copy model artifact to the artifact store directory."""
        dest_dir = settings.artifact_dir / name / f"v{version}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / model_path.name
        shutil.copy2(model_path, dest_path)
        logger.debug(f"Artifact stored at {dest_path}")
        return dest_path

    async def _get_or_create_model(self, name: str, framework: str) -> Model:
        """Get an existing model record or create a new one."""
        result = await self._db.execute(select(Model).where(Model.name == name))
        model_record = result.scalar_one_or_none()

        if model_record is None:
            model_record = Model(name=name, framework=framework)
            self._db.add(model_record)
            await self._db.flush()
            logger.debug(f"Created new model record: {name}")
        else:
            # Update framework if re-deploying with different one
            if model_record.framework != framework:
                model_record.framework = framework
                await self._db.flush()

        return model_record

    async def _next_version(self, model_id: str) -> int:
        """Get the next version number for a model."""
        result = await self._db.execute(
            select(func.max(ModelVersion.version)).where(ModelVersion.model_id == model_id)
        )
        max_version = result.scalar_one_or_none()
        return (max_version or 0) + 1

    async def _get_or_create_deployment(self, name: str, model_version_id: str) -> Deployment:
        """Get existing deployment or create a new one."""
        result = await self._db.execute(select(Deployment).where(Deployment.name == name))
        deployment = result.scalar_one_or_none()

        if deployment is None:
            deployment = Deployment(
                name=name,
                model_version_id=model_version_id,
            )
            self._db.add(deployment)
            await self._db.flush()
        else:
            deployment.model_version_id = model_version_id
            await self._db.flush()

        return deployment

    async def _add_event(self, deployment_id: str, event_type: str, message: str) -> None:
        """Add a deployment event."""
        event = DeploymentEvent(
            deployment_id=deployment_id,
            event_type=event_type,
            message=message,
        )
        self._db.add(event)
