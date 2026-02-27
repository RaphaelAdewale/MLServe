"""Tests for Pydantic API schemas."""

import pytest
from pydantic import ValidationError

from mlserve.api.schemas import DeployRequest, DeployResponse, HealthResponse


class TestDeployRequest:
    """Test DeployRequest validation."""

    def test_valid_name(self):
        req = DeployRequest(name="fraud-detector")
        assert req.name == "fraud-detector"
        assert req.framework is None
        assert req.replicas == 1

    def test_valid_with_all_fields(self):
        req = DeployRequest(name="my-model", framework="sklearn", replicas=3)
        assert req.framework == "sklearn"
        assert req.replicas == 3

    def test_name_no_uppercase(self):
        with pytest.raises(ValidationError):
            DeployRequest(name="FraudDetector")

    def test_name_no_spaces(self):
        with pytest.raises(ValidationError):
            DeployRequest(name="fraud detector")

    def test_name_no_underscores(self):
        with pytest.raises(ValidationError):
            DeployRequest(name="fraud_detector")

    def test_replicas_min(self):
        with pytest.raises(ValidationError):
            DeployRequest(name="my-model", replicas=0)

    def test_replicas_max(self):
        with pytest.raises(ValidationError):
            DeployRequest(name="my-model", replicas=11)


class TestDeployResponse:
    """Test DeployResponse serialization."""

    def test_from_dict(self):
        data = {
            "name": "test-model",
            "version": 1,
            "framework": "sklearn",
            "status": "running",
            "endpoint_url": "http://localhost/models/test-model/predict",
            "container_image": "localhost:5000/mlserve/test-model:v1",
            "port": 9001,
        }
        resp = DeployResponse(**data)
        assert resp.name == "test-model"
        assert resp.version == 1

    def test_optional_fields(self):
        resp = DeployResponse(name="test", version=1, framework="onnx", status="building")
        assert resp.endpoint_url is None
        assert resp.port is None


class TestHealthResponse:
    """Test HealthResponse."""

    def test_health(self):
        resp = HealthResponse(status="ok", version="0.1.0")
        assert resp.status == "ok"
