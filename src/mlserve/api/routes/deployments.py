"""Deployment management routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from mlserve.api.dependencies import get_db, get_model_service
from mlserve.api.schemas import (
    DeploymentDetailResponse,
    DeploymentListResponse,
    ErrorResponse,
)
from mlserve.core.exceptions import ModelNotFoundError

logger = logging.getLogger("mlserve.api.routes.deployments")

router = APIRouter(prefix="/api/v1/deployments", tags=["deployments"])


@router.get(
    "",
    response_model=DeploymentListResponse,
    summary="List all deployments",
)
async def list_deployments(db: AsyncSession = Depends(get_db)):
    """List all model deployments with their current status."""
    model_service = await get_model_service(db)
    deployments = await model_service.list_deployments()
    return DeploymentListResponse(deployments=deployments)


@router.get(
    "/{name}",
    response_model=DeploymentDetailResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get deployment details",
)
async def get_deployment(name: str, db: AsyncSession = Depends(get_db)):
    """Get detailed status of a specific deployment."""
    try:
        model_service = await get_model_service(db)
        detail = await model_service.get_deployment(name)
        return DeploymentDetailResponse(**detail)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete(
    "/{name}",
    responses={404: {"model": ErrorResponse}},
    summary="Delete a deployment",
)
async def delete_deployment(name: str, db: AsyncSession = Depends(get_db)):
    """Stop and remove a deployment."""
    try:
        model_service = await get_model_service(db)
        await model_service.delete_deployment(name)
        return {"message": f"Deployment '{name}' deleted successfully"}
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
