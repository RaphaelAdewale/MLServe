"""
Build Service — generates Docker images for model serving.

Assembles a build context directory with:
  - Rendered Dockerfile (from Jinja2 template)
  - Serving runtime code (server.py, model_loader.py)
  - Framework requirements file
  - Model artifact

Then builds the image using the Docker SDK.
"""

import logging
import shutil
import tempfile
from pathlib import Path

import docker
import docker.errors
from jinja2 import Environment, FileSystemLoader

from mlserve.core.config import settings
from mlserve.core.exceptions import BuildError

logger = logging.getLogger("mlserve.services.build")

# Paths relative to this package
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_RUNTIME_DIR = _PACKAGE_DIR / "runtime"
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"


class BuildService:
    """Builds Docker images for ML model serving."""

    def __init__(self):
        try:
            self._client = docker.from_env()
            self._client.ping()
        except docker.errors.DockerException as e:
            raise BuildError(
                f"Cannot connect to Docker daemon. Is Docker running? Error: {e}"
            )

        self._jinja_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
        )

    def build_image(
        self,
        model_name: str,
        version: int,
        framework: str,
        model_artifact_path: Path,
    ) -> str:
        """
        Build a Docker image for a model.

        Args:
            model_name: Name of the model (e.g., "fraud-detector").
            version: Version number (e.g., 1).
            framework: ML framework ("sklearn" or "onnx").
            model_artifact_path: Path to the model file on disk.

        Returns:
            The full image tag (e.g., "localhost:5000/mlserve/fraud-detector:v1").

        Raises:
            BuildError: If the build fails.
        """
        image_tag = f"{settings.docker_registry}/mlserve/{model_name}:v{version}"
        logger.info(f"Building image: {image_tag}")

        build_context = None
        try:
            # 1. Create temporary build context directory
            build_context = Path(tempfile.mkdtemp(prefix="mlserve-build-"))
            self._assemble_context(
                build_context, model_name, version, framework, model_artifact_path
            )

            # 2. Build the image
            image, build_logs = self._client.images.build(
                path=str(build_context),
                tag=image_tag,
                rm=True,
                forcerm=True,
            )

            # 3. Log build output
            for chunk in build_logs:
                if "stream" in chunk:
                    line = chunk["stream"].strip()
                    if line:
                        logger.debug(f"[docker build] {line}")
                if "error" in chunk:
                    raise BuildError(f"Docker build error: {chunk['error']}")

            logger.info(f"Image built successfully: {image_tag}")
            return image_tag

        except BuildError:
            raise
        except docker.errors.BuildError as e:
            raise BuildError(f"Docker build failed: {e}") from e
        except Exception as e:
            raise BuildError(f"Unexpected build error: {e}") from e
        finally:
            # Clean up temp directory
            if build_context and build_context.exists():
                shutil.rmtree(build_context, ignore_errors=True)

    def _assemble_context(
        self,
        build_dir: Path,
        model_name: str,
        version: int,
        framework: str,
        model_artifact_path: Path,
    ) -> None:
        """
        Assemble all files needed for docker build into build_dir.

        Layout:
            build_dir/
            ├── Dockerfile
            ├── requirements.txt
            ├── server.py
            ├── model_loader.py
            └── model/
                └── <model_file>
        """
        model_filename = model_artifact_path.name

        # 1. Render Dockerfile from template
        template = self._jinja_env.get_template("base.Dockerfile.j2")
        dockerfile_content = template.render(
            model_name=model_name,
            version=version,
            framework=framework,
            model_filename=model_filename,
        )
        (build_dir / "Dockerfile").write_text(dockerfile_content)

        # 2. Copy framework requirements
        requirements_src = _RUNTIME_DIR / "requirements" / f"{framework}.txt"
        if not requirements_src.exists():
            raise BuildError(
                f"No requirements file for framework '{framework}' at {requirements_src}"
            )
        shutil.copy2(requirements_src, build_dir / "requirements.txt")

        # 3. Copy serving runtime code
        shutil.copy2(_RUNTIME_DIR / "server.py", build_dir / "server.py")
        shutil.copy2(_RUNTIME_DIR / "model_loader.py", build_dir / "model_loader.py")

        # 4. Copy model artifact
        model_dir = build_dir / "model"
        model_dir.mkdir()
        shutil.copy2(model_artifact_path, model_dir / model_filename)

        logger.debug(f"Build context assembled at {build_dir}")
