"""Tests for framework auto-detection."""

from pathlib import Path

import pytest

from mlserve.core.exceptions import FrameworkDetectionError
from mlserve.services.model_service import ModelService


class TestFrameworkDetection:
    """Test ModelService.detect_framework()."""

    def test_detect_sklearn_pkl(self, tmp_path: Path):
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"fake model data")
        assert ModelService.detect_framework(model_file) == "sklearn"

    def test_detect_sklearn_joblib(self, tmp_path: Path):
        model_file = tmp_path / "model.joblib"
        model_file.write_bytes(b"fake model data")
        assert ModelService.detect_framework(model_file) == "sklearn"

    def test_detect_onnx(self, tmp_path: Path):
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake model data")
        assert ModelService.detect_framework(model_file) == "onnx"

    def test_unknown_extension_raises(self, tmp_path: Path):
        model_file = tmp_path / "model.xyz"
        model_file.write_bytes(b"fake model data")
        with pytest.raises(FrameworkDetectionError, match="Cannot auto-detect"):
            ModelService.detect_framework(model_file)

    def test_no_extension_raises(self, tmp_path: Path):
        model_file = tmp_path / "model"
        model_file.write_bytes(b"fake model data")
        with pytest.raises(FrameworkDetectionError):
            ModelService.detect_framework(model_file)
