"""Tests for database models and operations."""

import pytest
from sqlalchemy import select

from mlserve.core.models import Deployment, DeploymentEvent, Model, ModelVersion


class TestModelCRUD:
    """Test basic CRUD operations on ORM models."""

    async def test_create_model(self, db_session):
        model = Model(name="test-model", framework="sklearn")
        db_session.add(model)
        await db_session.flush()

        result = await db_session.execute(select(Model).where(Model.name == "test-model"))
        fetched = result.scalar_one()

        assert fetched.name == "test-model"
        assert fetched.framework == "sklearn"
        assert fetched.id is not None
        assert fetched.created_at is not None

    async def test_create_model_version(self, db_session):
        model = Model(name="versioned-model", framework="onnx")
        db_session.add(model)
        await db_session.flush()

        version = ModelVersion(
            model_id=model.id,
            version=1,
            artifact_path="/artifacts/versioned-model/v1/model.onnx",
            checksum_sha256="abc123",
            file_size_bytes=1024,
        )
        db_session.add(version)
        await db_session.flush()

        result = await db_session.execute(
            select(ModelVersion).where(ModelVersion.model_id == model.id)
        )
        fetched = result.scalar_one()

        assert fetched.version == 1
        assert fetched.checksum_sha256 == "abc123"

    async def test_create_deployment(self, db_session):
        model = Model(name="deploy-test", framework="sklearn")
        db_session.add(model)
        await db_session.flush()

        version = ModelVersion(
            model_id=model.id,
            version=1,
            artifact_path="/path/to/model",
            checksum_sha256="def456",
        )
        db_session.add(version)
        await db_session.flush()

        deployment = Deployment(
            name="deploy-test",
            model_version_id=version.id,
            status="running",
            endpoint_url="http://localhost/models/deploy-test/predict",
            port=9001,
        )
        db_session.add(deployment)
        await db_session.flush()

        result = await db_session.execute(
            select(Deployment).where(Deployment.name == "deploy-test")
        )
        fetched = result.scalar_one()

        assert fetched.status == "running"
        assert fetched.port == 9001

    async def test_deployment_events(self, db_session):
        model = Model(name="event-test", framework="sklearn")
        db_session.add(model)
        await db_session.flush()

        version = ModelVersion(
            model_id=model.id,
            version=1,
            artifact_path="/path",
            checksum_sha256="ghi789",
        )
        db_session.add(version)
        await db_session.flush()

        deployment = Deployment(
            name="event-test",
            model_version_id=version.id,
        )
        db_session.add(deployment)
        await db_session.flush()

        event = DeploymentEvent(
            deployment_id=deployment.id,
            event_type="building",
            message="Building container image",
        )
        db_session.add(event)
        await db_session.flush()

        result = await db_session.execute(
            select(DeploymentEvent).where(DeploymentEvent.deployment_id == deployment.id)
        )
        events = result.scalars().all()

        assert len(events) == 1
        assert events[0].event_type == "building"

    async def test_unique_model_name(self, db_session):
        model1 = Model(name="unique-test", framework="sklearn")
        db_session.add(model1)
        await db_session.flush()

        model2 = Model(name="unique-test", framework="onnx")
        db_session.add(model2)

        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            await db_session.flush()
