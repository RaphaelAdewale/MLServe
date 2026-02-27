"""
Framework-specific model loaders.

Each loader returns an object with a .predict(instances) method
that accepts a numpy array and returns predictions.

NOT imported by the host application — this is copied into Docker images.
"""

import logging

import numpy as np

logger = logging.getLogger("mlserve.runtime.loader")


class SklearnPredictor:
    """Wrapper for scikit-learn models loaded via joblib."""

    def __init__(self, model):
        self._model = model

    def predict(self, instances: np.ndarray) -> np.ndarray:
        return self._model.predict(instances)


class ONNXPredictor:
    """Wrapper for ONNX models using onnxruntime."""

    def __init__(self, session, input_name: str):
        self._session = session
        self._input_name = input_name

    def predict(self, instances: np.ndarray) -> np.ndarray:
        input_data = instances.astype(np.float32)
        results = self._session.run(None, {self._input_name: input_data})
        return np.array(results[0])


def _load_sklearn(path: str):
    """Load a scikit-learn model saved with joblib or pickle."""
    import joblib

    logger.info(f"Loading sklearn model from {path}")
    model = joblib.load(path)
    return SklearnPredictor(model)


def _load_onnx(path: str):
    """Load an ONNX model using onnxruntime."""
    import onnxruntime as ort

    logger.info(f"Loading ONNX model from {path}")
    session = ort.InferenceSession(path)
    input_name = session.get_inputs()[0].name
    return ONNXPredictor(session, input_name)


LOADERS = {
    "sklearn": _load_sklearn,
    "onnx": _load_onnx,
}


def load_model(path: str, framework: str):
    """
    Load a model based on framework type.

    Args:
        path: Path to the model file.
        framework: One of "sklearn", "onnx".

    Returns:
        An object with a .predict(np.ndarray) method.

    Raises:
        ValueError: If the framework is not supported.
        FileNotFoundError: If the model file does not exist.
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    loader = LOADERS.get(framework)
    if loader is None:
        supported = ", ".join(sorted(LOADERS.keys()))
        raise ValueError(f"Unsupported framework: '{framework}'. Supported: {supported}")

    return loader(path)
