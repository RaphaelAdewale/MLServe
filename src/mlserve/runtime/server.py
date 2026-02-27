"""
MLServe Serving Runtime — runs inside every model container.

This FastAPI server loads a single ML model and serves predictions via HTTP.
All configuration comes from environment variables.

NOT imported by the host application — this is copied into Docker images.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from model_loader import load_model
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

# --- Configuration from environment ---
MODEL_NAME = os.environ.get("MODEL_NAME", "unknown")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/model.pkl")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1")
FRAMEWORK = os.environ.get("FRAMEWORK", "sklearn")

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mlserve.runtime")

# --- Prometheus Metrics ---
PREDICTION_COUNT = Counter(
    "mlserve_predictions_total",
    "Total predictions served",
    ["model_name", "status"],
)
PREDICTION_LATENCY = Histogram(
    "mlserve_prediction_duration_seconds",
    "Prediction latency in seconds",
    ["model_name"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
MODEL_LOAD_TIME = None

# --- Global model reference ---
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, MODEL_LOAD_TIME

    logger.info(f"Loading model: {MODEL_NAME} v{MODEL_VERSION} (framework={FRAMEWORK})")
    start = time.perf_counter()

    try:
        model = load_model(MODEL_PATH, FRAMEWORK)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    MODEL_LOAD_TIME = time.perf_counter() - start
    logger.info(f"Model loaded in {MODEL_LOAD_TIME:.3f}s")

    yield

    logger.info("Shutting down runtime")


app = FastAPI(
    title=f"MLServe Runtime — {MODEL_NAME}",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# --- Request / Response schemas ---
class PredictRequest(BaseModel):
    instances: list
    """List of input instances. Each instance is a list of features."""


class PredictResponse(BaseModel):
    predictions: list
    model_name: str
    model_version: str


# --- Endpoints ---
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Run inference on the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        input_data = np.array(request.instances)
        predictions = model.predict(input_data)

        # Convert numpy types to Python native types for JSON serialization
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()

        PREDICTION_COUNT.labels(model_name=MODEL_NAME, status="success").inc()

        return PredictResponse(
            predictions=predictions,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
        )
    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_COUNT.labels(model_name=MODEL_NAME, status="error").inc()
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        duration = time.perf_counter() - start
        PREDICTION_LATENCY.labels(model_name=MODEL_NAME).observe(duration)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "not_ready",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "framework": FRAMEWORK,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/info")
async def info():
    """Model metadata."""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "framework": FRAMEWORK,
        "model_load_time_seconds": MODEL_LOAD_TIME,
    }
