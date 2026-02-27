"""Model deployment routes — the core API."""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from mlserve.api.dependencies import get_db, get_model_service
from mlserve.api.schemas import DeployResponse, ErrorResponse
from mlserve.core.exceptions import BuildError, DeploymentError, FrameworkDetectionError

logger = logging.getLogger("mlserve.api.routes.models")

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.post(
    "/deploy",
    response_model=DeployResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Deploy a model",
    description="Upload a model file and deploy it as a REST API.",
)
async def deploy_model(
    name: str = Form(..., description="Deployment name (lowercase, hyphens allowed)"),
    file: UploadFile = File(..., description="Model file (.pkl, .joblib, .onnx)"),
    framework: str | None = Form(None, description="ML framework (auto-detected if omitted)"),
    replicas: int = Form(1, description="Number of replicas", ge=1, le=10),
    db: AsyncSession = Depends(get_db),
):
    """
    Deploy an ML model.

    Upload a model file and MLServe will:
    1. Auto-detect the framework (or use the one you specify)
    2. Build a Docker container with the serving runtime
    3. Deploy the container and route traffic to it
    4. Return the prediction endpoint URL
    """
    # Validate name format
    import re

    if not re.match(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$", name) and len(name) > 1:
        raise HTTPException(
            status_code=400,
            detail="Name must be lowercase alphanumeric with hyphens (e.g., 'fraud-detector')",
        )
    if len(name) == 1 and not name.isalnum():
        raise HTTPException(status_code=400, detail="Single-character name must be alphanumeric")

    # Save uploaded file to a temp location
    suffix = Path(file.filename).suffix if file.filename else ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=f"mlserve-{name}-")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        model_path = Path(tmp.name)

        # Run the full pipeline
        model_service = await get_model_service(db)
        result = model_service.register_and_deploy(
            name=name,
            model_path=model_path,
            framework=framework,
            replicas=replicas,
        )

        # register_and_deploy is async
        result = await result

        return DeployResponse(**result)

    except FrameworkDetectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except BuildError as e:
        raise HTTPException(status_code=500, detail=f"Build failed: {e}")
    except DeploymentError as e:
        raise HTTPException(status_code=500, detail=f"Deployment failed: {e}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temp file
        Path(tmp.name).unlink(missing_ok=True)
