"""Custom exception hierarchy."""


class MLServeError(Exception):
    """Base exception for MLServe."""


class FrameworkDetectionError(MLServeError):
    """Could not auto-detect the ML framework."""


class ModelNotFoundError(MLServeError):
    """Model not found in the registry."""


class BuildError(MLServeError):
    """Container image build failed."""


class DeploymentError(MLServeError):
    """Deployment operation failed."""
