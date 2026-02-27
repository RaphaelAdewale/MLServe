"""Pydantic schemas for API request/response models."""


from pydantic import BaseModel, Field

# --- Requests ---

class DeployRequest(BaseModel):
    """Request body for deploying a model (used with file upload as form fields)."""

    name: str = Field(..., min_length=1, max_length=255, pattern=r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$")
    framework: str | None = Field(None, description="ML framework. Auto-detected if omitted.")
    replicas: int = Field(1, ge=1, le=10)


# --- Responses ---

class DeployResponse(BaseModel):
    """Response after a successful deploy."""

    name: str
    version: int
    framework: str
    status: str
    endpoint_url: str | None = None
    container_image: str | None = None
    port: int | None = None


class DeploymentEventResponse(BaseModel):
    """A single deployment event."""

    type: str
    message: str | None = None
    timestamp: str | None = None


class ContainerStatusResponse(BaseModel):
    """Live container status."""

    container_id: str | None = None
    status: str | None = None
    health: str | None = None


class DeploymentDetailResponse(BaseModel):
    """Detailed deployment status."""

    name: str
    version: int | None = None
    framework: str | None = None
    status: str
    endpoint_url: str | None = None
    container_image: str | None = None
    port: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    container: ContainerStatusResponse | None = None
    events: list[DeploymentEventResponse] = []


class DeploymentListItem(BaseModel):
    """Summary of a deployment for list view."""

    name: str
    version: int | None = None
    framework: str | None = None
    status: str
    endpoint_url: str | None = None
    port: int | None = None
    created_at: str | None = None


class DeploymentListResponse(BaseModel):
    """List of all deployments."""

    deployments: list[DeploymentListItem]


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    version: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
