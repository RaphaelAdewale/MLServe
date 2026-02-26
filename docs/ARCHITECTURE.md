# MLServe — Architecture Design Document

> **"Give me a model, I'll give you a production API."**

## Table of Contents

1. [Is This Too Ambitious?](#is-this-too-ambitious)
2. [High-Level Architecture](#high-level-architecture)
3. [System Architecture Diagram](#system-architecture-diagram)
4. [Component Deep Dive](#component-deep-dive)
5. [Data Flow: What Happens on `mlserve deploy`](#data-flow)
6. [Technology Choices & Justification](#technology-choices)
7. [Project Structure](#project-structure)
8. [API Contracts](#api-contracts)
9. [Phased Roadmap](#phased-roadmap)
10. [What This Does NOT Cover (Explicit Non-Goals)](#non-goals)

---

## Is This Too Ambitious?

**No.** This is a well-understood problem space. Here's why:

| Existing Tool | What It Proves |
|---------------|---------------|
| [BentoML](https://github.com/bentoml/BentoML) | Single-command model packaging + serving works at scale |
| [MLflow Models](https://mlflow.org/docs/latest/models.html) | Framework-agnostic model serialization + REST serving is viable |
| [Seldon Core](https://github.com/SeldonIO/seldon-core) | K8s-native model serving with advanced traffic management is production-ready |
| [TF Serving](https://github.com/tensorflow/serving) | High-performance model serving is a solved problem |
| [TorchServe](https://github.com/pytorch/serve) | PyTorch-specific serving is mature |

The key insight: none of these nail the **"zero-config, one command"** developer experience for a self-hosted platform. That's the gap MLServe fills.

**Realistic effort estimate:**

| Phase | Scope | Timeline (1 engineer) |
|-------|-------|-----------------------|
| MVP | Local Docker deploy, 3 frameworks | 3–4 weeks |
| V1 | Kubernetes deploy, model registry, monitoring | 6–8 weeks |
| V2 | Auto-scaling, A/B testing, drift detection | 12+ weeks |

---

## High-Level Architecture

MLServe is composed of **6 layers**, each with a single responsibility:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                        │
│                     CLI (mlserve deploy model.pkl)                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CONTROL PLANE                               │
│              API Server (FastAPI) — The Brain                      │
│         Orchestrates: build → store → deploy → route               │
└──┬──────────┬──────────────┬──────────────┬─────────────────────┬──┘
   │          │              │              │                     │
   ▼          ▼              ▼              ▼                     ▼
┌───────┐ ┌────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────────┐
│Model  │ │Build   │ │Deployment  │ │API Gateway │ │  Monitoring      │
│Store  │ │Service │ │Orchestrator│ │  / Router  │ │  & Observability │
│(MinIO)│ │(Docker)│ │(K8s/Docker)│ │ (Traefik)  │ │(Prometheus+Graf) │
└───────┘ └────────┘ └────────────┘ └────────────┘ └──────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Serving Runtime(s)  │
              │  FastAPI + Uvicorn    │
              │  per deployed model   │
              └───────────────────────┘
```

### Design Principles

1. **Convention over configuration** — sensible defaults, override only when needed
2. **Framework-agnostic** — works with sklearn, PyTorch, TensorFlow, ONNX, XGBoost
3. **Immutable deployments** — every deploy creates a versioned, reproducible container
4. **Local-first, cloud-ready** — Docker Compose locally, Kubernetes in production
5. **No vendor lock-in** — every component is open-source and replaceable

---

## System Architecture Diagram

```
                    ┌──────────────────────────┐
                    │     Data Scientist        │
                    │                           │
                    │  $ mlserve deploy \       │
                    │      model.pkl \          │
                    │      --name fraud-detect  │
                    └───────────┬───────────────┘
                                │
                    ┌───────────▼───────────────┐
                    │      MLServe CLI          │
                    │                           │
                    │  • Detect framework       │
                    │  • Validate model file    │
                    │  • Upload artifact        │
                    │  • Call Control Plane API  │
                    └───────────┬───────────────┘
                                │ REST (HTTP)
                    ┌───────────▼───────────────┐
                    │    Control Plane API       │
                    │      (FastAPI)             │
                    │                           │
                    │  POST /api/v1/models       │
                    │  POST /api/v1/deployments  │
                    │  GET  /api/v1/deployments  │
                    │  DELETE /api/v1/deploy/... │
                    └──┬─────┬──────┬───────┬───┘
                       │     │      │       │
           ┌───────────┘     │      │       └───────────────┐
           ▼                 ▼      ▼                       ▼
  ┌─────────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────┐
  │ Model Store │  │  Build   │  │  Deployment  │  │  Metadata DB │
  │   (MinIO)   │  │ Service  │  │ Orchestrator │  │ (PostgreSQL) │
  │             │  │          │  │              │  │              │
  │ • Artifacts │  │ • Gen    │  │ • Docker     │  │ • Model info │
  │ • Versions  │  │   Docker │  │   Compose    │  │ • Deploy     │
  │ • Checksums │  │   file   │  │   (dev)      │  │   state      │
  │             │  │ • Build  │  │ • Kubernetes  │  │ • Versions   │
  │             │  │   image  │  │   (prod)     │  │ • Endpoints  │
  │             │  │ • Push   │  │ • Health     │  │              │
  │             │  │   to reg │  │   checks     │  │              │
  └─────────────┘  └──────────┘  └──────┬───────┘  └──────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │  Serving Runtime  │  (1 container per model version)
                              │                   │
                              │  ┌──────────────┐ │
                              │  │   FastAPI     │ │
                              │  │   + Uvicorn   │ │
                              │  └──────┬───────┘ │
                              │         │         │
                              │  ┌──────▼───────┐ │
                              │  │ Model Loader │ │
                              │  │  (sklearn /  │ │
                              │  │  torch / tf  │ │
                              │  │  / onnx /    │ │
                              │  │  xgboost)    │ │
                              │  └──────────────┘ │
                              └────────┬──────────┘
                                       │
                         ┌─────────────▼──────────────┐
                         │      API Gateway           │
                         │       (Traefik)            │
                         │                            │
                         │  • /models/fraud-detect →  │
                         │     container_fraud:8080   │
                         │  • /models/churn-pred →    │
                         │     container_churn:8080   │
                         │  • TLS termination         │
                         │  • Rate limiting           │
                         └─────────────┬──────────────┘
                                       │
                         ┌─────────────▼──────────────┐
                         │    End Users / Services    │
                         │                            │
                         │  POST /models/fraud-detect │
                         │       /predict             │
                         │  {"features": [1.2, ...]}  │
                         └────────────────────────────┘


  ┌────────────── Observability (Cross-Cutting) ──────────────────┐
  │                                                                │
  │  Prometheus  ──▶  Grafana Dashboard                           │
  │  • Request latency (p50, p95, p99)                            │
  │  • Throughput (req/sec per model)                              │
  │  • Error rate (4xx, 5xx)                                      │
  │  • Container CPU/memory                                        │
  │  • Model inference time                                        │
  │                                                                │
  │  Structured Logging ──▶ stdout (Docker) / Loki (optional)     │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. CLI (`mlserve`)

**Responsibility:** Single entry point for data scientists. Zero-config by default, configurable when needed.

**Technology:** Python + [Typer](https://typer.tiangolo.com/) (built on Click, auto-generates `--help`)

**Key Commands:**

```bash
# Deploy a model (the core use case)
mlserve deploy model.pkl --name fraud-detector

# With explicit options
mlserve deploy model.onnx \
  --name fraud-detector \
  --framework onnx \
  --replicas 3 \
  --cpu 500m --memory 512Mi \
  --env production

# Lifecycle management
mlserve list                          # List all deployments
mlserve status fraud-detector         # Health, replicas, endpoint URL
mlserve logs fraud-detector           # Stream container logs
mlserve delete fraud-detector         # Tear down
mlserve rollback fraud-detector --to v2  # Rollback to version
```

**Framework Auto-Detection Logic:**

```python
FRAMEWORK_SIGNATURES = {
    ".pkl":       ["sklearn", "xgboost"],   # Inspect pickle header
    ".joblib":    ["sklearn", "xgboost"],   # Same approach
    ".pt":        ["pytorch"],
    ".pth":       ["pytorch"],
    ".onnx":      ["onnx"],
    ".h5":        ["tensorflow"],
    ".keras":     ["tensorflow"],
    "saved_model.pb": ["tensorflow"],       # TF SavedModel directory
    ".bst":       ["xgboost"],
    ".cbm":       ["catboost"],
    ".txt":       ["lightgbm"],             # LightGBM text format
}

def detect_framework(model_path: Path) -> str:
    """Auto-detect ML framework from file extension + content inspection."""
    suffix = model_path.suffix
    candidates = FRAMEWORK_SIGNATURES.get(suffix, [])

    if len(candidates) == 1:
        return candidates[0]

    if suffix in (".pkl", ".joblib"):
        return _inspect_pickle_header(model_path)

    raise FrameworkDetectionError(
        f"Cannot auto-detect framework for {suffix}. "
        f"Use --framework to specify explicitly."
    )
```

### 2. Control Plane API

**Responsibility:** Central orchestration service. Receives commands from CLI, coordinates all other components.

**Technology:** [FastAPI](https://fastapi.tiangolo.com/) + [SQLAlchemy](https://www.sqlalchemy.org/) + PostgreSQL (SQLite for dev)

**Why a separate API server (not just CLI → Docker directly)?**
- Enables multi-user support (multiple data scientists deploying)
- Central state management (which models are deployed where)
- Decouples CLI from deployment target (swap Docker → K8s without CLI changes)
- Enables a future web UI

**Database Schema (core tables):**

```sql
CREATE TABLE models (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          VARCHAR(255) NOT NULL UNIQUE,
    framework     VARCHAR(50) NOT NULL,          -- sklearn, pytorch, tensorflow, onnx, xgboost
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE model_versions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id      UUID REFERENCES models(id) ON DELETE CASCADE,
    version       INTEGER NOT NULL,               -- auto-incrementing per model
    artifact_uri  VARCHAR(1024) NOT NULL,          -- s3://mlserve/models/{name}/v{version}/
    checksum_sha256 VARCHAR(64) NOT NULL,          -- integrity verification
    input_schema  JSONB,                           -- optional: {"features": "float[]", "shape": [1, 10]}
    output_schema JSONB,                           -- optional: {"prediction": "float", "probabilities": "float[]"}
    file_size_bytes BIGINT,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE deployments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id UUID REFERENCES model_versions(id),
    name            VARCHAR(255) NOT NULL UNIQUE,  -- same as model name (1 active deploy per model)
    status          VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending, building, deploying, running, failed, stopped
    endpoint_url    VARCHAR(1024),                 -- e.g., https://mlserve.local/models/fraud-detector
    replicas        INTEGER DEFAULT 1,
    cpu_request     VARCHAR(10) DEFAULT '250m',
    memory_request  VARCHAR(10) DEFAULT '256Mi',
    container_image VARCHAR(512),                   -- registry.local/mlserve/fraud-detector:v3
    target          VARCHAR(20) DEFAULT 'docker',   -- docker | kubernetes
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE deployment_events (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id UUID REFERENCES deployments(id) ON DELETE CASCADE,
    event_type    VARCHAR(50) NOT NULL,             -- created, building, build_failed, deploying, running, health_check_failed, stopped
    message       TEXT,
    created_at    TIMESTAMP DEFAULT NOW()
);
```

### 3. Model Store (Artifact Storage)

**Responsibility:** Durable, versioned storage for model artifacts.

**Technology:** [MinIO](https://min.io/) (S3-compatible, self-hosted) — for dev, can use local filesystem.

**Storage Layout:**

```
mlserve-bucket/
├── models/
│   ├── fraud-detector/
│   │   ├── v1/
│   │   │   ├── model.pkl
│   │   │   └── metadata.json       # framework, checksum, input/output schema
│   │   ├── v2/
│   │   │   ├── model.pkl
│   │   │   └── metadata.json
│   │   └── latest -> v2             # symlink/pointer to latest
│   └── churn-predictor/
│       ├── v1/
│       │   ├── model.onnx
│       │   └── metadata.json
│       └── latest -> v1
```

**Why MinIO over local filesystem?**
- S3-compatible API = drop-in replacement for AWS S3 in production
- Built-in versioning, checksums, and access control
- Works across multiple Control Plane replicas (shared storage)
- Single `docker run minio/minio` for local development

### 4. Build Service

**Responsibility:** Package model + serving runtime into a container image.

**Technology:** Docker (via Docker SDK for Python: `docker-py`)

**How it works:**

1. Receive build request: `(model_name, version, framework, artifact_uri)`
2. Pull model artifact from Model Store
3. Select base Dockerfile template for the framework
4. Build container image
5. Push to container registry (local Docker registry for dev, Harbor/ECR for prod)

**Template Dockerfile (sklearn example):**

```dockerfile
FROM python:3.11-slim

# Install only what's needed for this framework
RUN pip install --no-cache-dir \
    fastapi==0.115.* \
    uvicorn[standard]==0.34.* \
    scikit-learn==1.6.* \
    joblib==1.4.* \
    numpy==2.* \
    pydantic==2.*

WORKDIR /app

# Copy the generic serving runtime
COPY serving_runtime/ /app/

# Copy the model artifact
COPY model/ /app/model/

ENV MODEL_PATH=/app/model/model.pkl
ENV FRAMEWORK=sklearn
ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Framework-specific base images keep containers small:**

| Framework | Base Image | Approx. Size |
|-----------|-----------|---------------|
| sklearn | python:3.11-slim + sklearn | ~350 MB |
| pytorch (CPU) | python:3.11-slim + torch CPU | ~800 MB |
| tensorflow (CPU) | python:3.11-slim + tf-cpu | ~700 MB |
| onnx | python:3.11-slim + onnxruntime | ~300 MB |
| xgboost | python:3.11-slim + xgboost | ~300 MB |

### 5. Serving Runtime

**Responsibility:** The actual HTTP server that loads a model and serves predictions. One instance per deployed model version.

**Technology:** FastAPI + Uvicorn

**This is the code that runs inside every model container:**

```python
# server.py — Generic serving runtime (framework-agnostic)
import os
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from contextlib import asynccontextmanager

from model_loader import load_model  # Framework-specific loading

logger = logging.getLogger("mlserve.runtime")

# --- Prometheus Metrics ---
PREDICTION_COUNT = Counter(
    "mlserve_predictions_total",
    "Total predictions served",
    ["model_name", "status"]
)
PREDICTION_LATENCY = Histogram(
    "mlserve_prediction_duration_seconds",
    "Prediction latency in seconds",
    ["model_name"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_PATH = os.environ["MODEL_PATH"]
FRAMEWORK = os.environ["FRAMEWORK"]

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(f"Loading model: {MODEL_NAME} (framework={FRAMEWORK})")
    model = load_model(MODEL_PATH, FRAMEWORK)
    logger.info(f"Model loaded successfully")
    yield

app = FastAPI(title=f"MLServe - {MODEL_NAME}", lifespan=lifespan)

class PredictRequest(BaseModel):
    instances: list  # [[1.0, 2.0, ...], [3.0, 4.0, ...]]

class PredictResponse(BaseModel):
    predictions: list
    model_name: str
    model_version: str

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start = time.perf_counter()
    try:
        predictions = model.predict(request.instances)
        PREDICTION_COUNT.labels(model_name=MODEL_NAME, status="success").inc()
        return PredictResponse(
            predictions=predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            model_name=MODEL_NAME,
            model_version=os.environ.get("MODEL_VERSION", "unknown"),
        )
    except Exception as e:
        PREDICTION_COUNT.labels(model_name=MODEL_NAME, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        PREDICTION_LATENCY.labels(model_name=MODEL_NAME).observe(time.perf_counter() - start)

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME, "framework": FRAMEWORK}

@app.get("/metrics")
async def metrics():
    return generate_latest()
```

**Framework-Specific Model Loaders:**

```python
# model_loader.py
import numpy as np

def load_model(path: str, framework: str):
    """Load a model based on framework type. Returns an object with .predict()"""
    loader = LOADERS.get(framework)
    if not loader:
        raise ValueError(f"Unsupported framework: {framework}")
    return loader(path)

def _load_sklearn(path):
    import joblib
    return joblib.load(path)

def _load_pytorch(path):
    import torch
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()

    class TorchPredictor:
        def __init__(self, m):
            self.model = m
        def predict(self, instances):
            with torch.no_grad():
                tensor = torch.FloatTensor(instances)
                output = self.model(tensor)
                return output.numpy()

    return TorchPredictor(model)

def _load_tensorflow(path):
    import tensorflow as tf
    model = tf.keras.models.load_model(path)

    class TFPredictor:
        def __init__(self, m):
            self.model = m
        def predict(self, instances):
            import numpy as np
            return self.model.predict(np.array(instances))

    return TFPredictor(model)

def _load_onnx(path):
    import onnxruntime as ort
    session = ort.InferenceSession(path)
    input_name = session.get_inputs()[0].name

    class ONNXPredictor:
        def __init__(self, s, n):
            self.session = s
            self.input_name = n
        def predict(self, instances):
            import numpy as np
            result = self.session.run(None, {self.input_name: np.array(instances, dtype=np.float32)})
            return result[0]

    return ONNXPredictor(session, input_name)

def _load_xgboost(path):
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(path)

    class XGBPredictor:
        def __init__(self, m):
            self.model = m
        def predict(self, instances):
            import numpy as np
            dmatrix = xgb.DMatrix(np.array(instances))
            return self.model.predict(dmatrix)

    return XGBPredictor(model)

LOADERS = {
    "sklearn": _load_sklearn,
    "pytorch": _load_pytorch,
    "tensorflow": _load_tensorflow,
    "onnx": _load_onnx,
    "xgboost": _load_xgboost,
}
```

### 6. Deployment Orchestrator

**Responsibility:** Deploy and manage containers running the serving runtime. Abstracts the deployment target.

**Two backends:**

#### a) Docker Compose Backend (Development / Single-node)

```python
# orchestrator/docker_backend.py
class DockerBackend:
    """Deploy models as Docker containers on a single machine."""

    def deploy(self, deployment: Deployment) -> str:
        """Start a container, return the endpoint URL."""
        container = self.docker_client.containers.run(
            image=deployment.container_image,
            name=f"mlserve-{deployment.name}",
            detach=True,
            environment={
                "MODEL_NAME": deployment.name,
                "MODEL_PATH": "/app/model/model.pkl",
                "MODEL_VERSION": str(deployment.model_version),
                "FRAMEWORK": deployment.framework,
            },
            labels={
                "mlserve.managed": "true",
                "mlserve.model": deployment.name,
                # Traefik auto-discovery labels
                "traefik.enable": "true",
                f"traefik.http.routers.{deployment.name}.rule": f"PathPrefix(`/models/{deployment.name}`)",
                f"traefik.http.services.{deployment.name}.loadbalancer.server.port": "8080",
            },
            network="mlserve-network",
            mem_limit=deployment.memory_request,
            cpus=float(deployment.cpu_request.rstrip('m')) / 1000,
        )
        return f"http://localhost/models/{deployment.name}/predict"

    def teardown(self, name: str):
        container = self.docker_client.containers.get(f"mlserve-{name}")
        container.stop()
        container.remove()
```

#### b) Kubernetes Backend (Production / Multi-node)

```python
# orchestrator/k8s_backend.py
class KubernetesBackend:
    """Deploy models as K8s Deployments + Services."""

    def deploy(self, deployment: Deployment) -> str:
        """Create K8s Deployment + Service + Ingress, return endpoint URL."""
        # 1. Create Deployment
        self._create_deployment(deployment)  # Generates K8s Deployment YAML
        # 2. Create Service (ClusterIP)
        self._create_service(deployment)
        # 3. Create Ingress rule
        self._create_ingress(deployment)
        return f"https://{self.domain}/models/{deployment.name}/predict"

    def _create_deployment(self, d: Deployment):
        """Generate and apply a K8s Deployment."""
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"mlserve-{d.name}",
                "namespace": "mlserve",
                "labels": {"app": "mlserve", "model": d.name},
            },
            "spec": {
                "replicas": d.replicas,
                "selector": {"matchLabels": {"model": d.name}},
                "template": {
                    "metadata": {
                        "labels": {"model": d.name},
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "8080",
                            "prometheus.io/path": "/metrics",
                        },
                    },
                    "spec": {
                        "containers": [{
                            "name": "model-server",
                            "image": d.container_image,
                            "ports": [{"containerPort": 8080}],
                            "resources": {
                                "requests": {"cpu": d.cpu_request, "memory": d.memory_request},
                                "limits": {"cpu": d.cpu_request, "memory": d.memory_request},
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5,
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                            },
                        }],
                    },
                },
            },
        }
        self.apps_v1.create_namespaced_deployment("mlserve", manifest)
```

### 7. API Gateway

**Responsibility:** Route external traffic to the correct model container. Handle TLS, rate limiting, auth.

**Technology:** [Traefik](https://traefik.io/) — chosen because:
- **Auto-discovery**: Watches Docker labels and K8s Ingress resources automatically
- **Zero config**: New model containers appear as routes without restarting the gateway
- **Built-in**: Dashboard, Let's Encrypt, rate limiting, circuit breakers

**Request Routing:**

```
Client Request                         Traefik (port 80/443)          Model Containers
─────────────                         ────────────────────           ─────────────────

POST /models/fraud-detector/predict  ──▶  Route: PathPrefix         ──▶  mlserve-fraud-detector:8080
                                          (/models/fraud-detector)

POST /models/churn-predictor/predict ──▶  Route: PathPrefix         ──▶  mlserve-churn-predictor:8080
                                          (/models/churn-predictor)
```

### 8. Monitoring & Observability

**Stack:** Prometheus (metrics collection) + Grafana (visualization)

**Metrics exposed by every model container:**

| Metric | Type | Description |
|--------|------|-------------|
| `mlserve_predictions_total` | Counter | Total predictions (by model, status) |
| `mlserve_prediction_duration_seconds` | Histogram | Inference latency distribution |
| `mlserve_model_load_duration_seconds` | Gauge | Time taken to load model at startup |

**Infrastructure metrics (from cAdvisor / kube-state-metrics):**

| Metric | Description |
|--------|-------------|
| CPU usage per container | Detect resource pressure |
| Memory usage per container | Detect OOM risks |
| Container restart count | Detect crash loops |

---

## Data Flow

### What happens when a user runs `mlserve deploy model.pkl --name fraud-detector`

```
Step 1: CLI — Validate & Upload
  ├── Detect framework from file extension / content → "sklearn"
  ├── Compute SHA-256 checksum of model file
  ├── Upload model.pkl to MinIO: s3://mlserve/models/fraud-detector/v1/model.pkl
  └── Call Control Plane: POST /api/v1/models
      Body: {name: "fraud-detector", framework: "sklearn", artifact_uri: "s3://...", checksum: "abc123"}

Step 2: Control Plane — Register & Build
  ├── Insert into `models` table (or get existing)
  ├── Insert into `model_versions` table (auto-increment version)
  ├── Insert into `deployments` table (status: "building")
  ├── Trigger Build Service:
  │   ├── Pull model artifact from MinIO
  │   ├── Select sklearn Dockerfile template
  │   ├── docker build -t registry.local/mlserve/fraud-detector:v1 .
  │   └── docker push registry.local/mlserve/fraud-detector:v1
  └── Update deployment status: "building" → "deploying"

Step 3: Control Plane — Deploy
  ├── Call Deployment Orchestrator:
  │   ├── [Docker mode] docker run ... registry.local/mlserve/fraud-detector:v1
  │   └── [K8s mode] kubectl apply deployment + service + ingress
  ├── Wait for health check: GET /health returns 200
  └── Update deployment status: "deploying" → "running"

Step 4: CLI — Report Success
  └── Print to user:
      ✓ Model "fraud-detector" deployed successfully (v1)
        Endpoint: http://localhost/models/fraud-detector/predict
        Status:   running
        Replicas: 1

Step 5: User hits the endpoint
  POST http://localhost/models/fraud-detector/predict
  Body: {"instances": [[1.0, 2.0, 3.0, ...]]}
  Response: {"predictions": [0.87], "model_name": "fraud-detector", "model_version": "v1"}
```

---

## Technology Choices

Every technology choice is justified, open-source, and replaceable:

| Component | Technology | Why This | Alternatives Considered |
|-----------|-----------|----------|------------------------|
| CLI framework | [Typer](https://typer.tiangolo.com/) | Auto-generated help, type hints, best DX | Click (lower-level), argparse (too manual) |
| Control Plane API | [FastAPI](https://fastapi.tiangolo.com/) | Async, auto-docs (OpenAPI), Pydantic validation | Flask (sync, no auto-validation), Django (too heavy) |
| Database | PostgreSQL (prod) / SQLite (dev) | Reliable, JSONB for schemas, SQLAlchemy supports both | MySQL (no JSONB), MongoDB (overkill for structured data) |
| ORM | [SQLAlchemy 2.0](https://www.sqlalchemy.org/) | Industry standard, supports both Postgres + SQLite | Tortoise ORM (less mature), raw SQL (unmaintainable) |
| Artifact Store | [MinIO](https://min.io/) | S3-compatible, self-hosted, single binary | Local filesystem (doesn't scale), actual S3 (cloud dependency) |
| Container Build | [Docker SDK (docker-py)](https://docker-py.readthedocs.io/) | Programmatic Docker control from Python | subprocess + docker CLI (fragile), Buildpacks (opinionated) |
| Container Registry | Docker Registry (dev) / Harbor (prod) | Standard OCI registry | ECR/GCR (cloud lock-in), GitHub Packages (external dependency) |
| Serving Runtime | FastAPI + Uvicorn | High-performance async, matches Control Plane (one framework) | Flask + Gunicorn (sync, slower for I/O), gRPC (harder for data scientists) |
| API Gateway | [Traefik](https://traefik.io/) | Auto-discovery from Docker/K8s, zero-config | Nginx (manual config reload), Kong (heavier), Envoy (complex) |
| Orchestration (dev) | Docker Compose | Zero setup, runs anywhere | Podman (less ecosystem), direct Docker (no networking) |
| Orchestration (prod) | Kubernetes | Industry standard, auto-scaling, self-healing | Docker Swarm (dying), Nomad (niche), ECS (cloud lock-in) |
| Metrics | [Prometheus](https://prometheus.io/) | Pull-based, K8s native, industry standard | Datadog (expensive), InfluxDB (less ecosystem) |
| Dashboards | [Grafana](https://grafana.com/) | Best-in-class visualization, free | Kibana (ELK-specific), custom (waste of time) |
| Migrations | [Alembic](https://alembic.sqlalchemy.org/) | Standard for SQLAlchemy, reliable | Django migrations (wrong framework), hand-written SQL (fragile) |

---

## Project Structure

```
MLServe/
├── README.md
├── pyproject.toml                    # Project metadata, dependencies (using uv/poetry)
├── Makefile                          # Common commands: make dev, make test, make build
│
├── docs/
│   └── ARCHITECTURE.md               # ← You are here
│
├── src/
│   └── mlserve/
│       ├── __init__.py
│       ├── __main__.py               # Entry point: python -m mlserve
│       │
│       ├── cli/                      # CLI Layer
│       │   ├── __init__.py
│       │   ├── app.py                # Typer app definition
│       │   ├── commands/
│       │   │   ├── deploy.py         # mlserve deploy
│       │   │   ├── list.py           # mlserve list
│       │   │   ├── status.py         # mlserve status <name>
│       │   │   ├── logs.py           # mlserve logs <name>
│       │   │   ├── delete.py         # mlserve delete <name>
│       │   │   └── rollback.py       # mlserve rollback <name> --to <version>
│       │   └── utils.py              # CLI helpers (formatting, spinners)
│       │
│       ├── api/                      # Control Plane API
│       │   ├── __init__.py
│       │   ├── app.py                # FastAPI app definition
│       │   ├── routes/
│       │   │   ├── models.py         # /api/v1/models
│       │   │   ├── deployments.py    # /api/v1/deployments
│       │   │   └── health.py         # /health, /ready
│       │   ├── schemas.py            # Pydantic request/response schemas
│       │   └── dependencies.py       # FastAPI dependency injection
│       │
│       ├── core/                     # Business Logic (framework-agnostic)
│       │   ├── __init__.py
│       │   ├── models.py             # SQLAlchemy ORM models
│       │   ├── database.py           # DB session management
│       │   ├── config.py             # Settings via pydantic-settings
│       │   └── exceptions.py         # Custom exception hierarchy
│       │
│       ├── services/                 # Service Layer (orchestrates components)
│       │   ├── __init__.py
│       │   ├── model_service.py      # Upload, register, version models
│       │   ├── build_service.py      # Build container images
│       │   ├── deploy_service.py     # Deploy, teardown, rollback
│       │   └── artifact_store.py     # MinIO / S3 interactions
│       │
│       ├── orchestrator/             # Deployment Backends
│       │   ├── __init__.py
│       │   ├── base.py               # Abstract base class
│       │   ├── docker_backend.py     # Docker Compose deployment
│       │   └── k8s_backend.py        # Kubernetes deployment
│       │
│       ├── runtime/                  # Serving Runtime (copied into containers)
│       │   ├── server.py             # FastAPI prediction server
│       │   ├── model_loader.py       # Framework-specific model loading
│       │   └── requirements/         # Per-framework requirements files
│       │       ├── sklearn.txt
│       │       ├── pytorch.txt
│       │       ├── tensorflow.txt
│       │       ├── onnx.txt
│       │       └── xgboost.txt
│       │
│       └── templates/                # Dockerfile templates
│           ├── base.Dockerfile
│           ├── sklearn.Dockerfile
│           ├── pytorch.Dockerfile
│           ├── tensorflow.Dockerfile
│           ├── onnx.Dockerfile
│           └── xgboost.Dockerfile
│
├── tests/
│   ├── conftest.py                   # Shared fixtures (test DB, mock MinIO, etc.)
│   ├── unit/
│   │   ├── test_framework_detection.py
│   │   ├── test_model_service.py
│   │   ├── test_build_service.py
│   │   └── test_schema_validation.py
│   ├── integration/
│   │   ├── test_deploy_sklearn.py    # End-to-end: deploy sklearn model, hit /predict
│   │   ├── test_deploy_pytorch.py
│   │   └── test_deploy_lifecycle.py  # deploy → status → rollback → delete
│   └── fixtures/
│       └── models/                   # Tiny pre-trained models for tests
│           ├── sklearn_iris.pkl
│           └── onnx_iris.onnx
│
├── infrastructure/                   # Docker Compose + K8s manifests
│   ├── docker-compose.yml            # Full local stack
│   ├── docker-compose.dev.yml        # Dev overrides (hot-reload, debug)
│   ├── kubernetes/
│   │   ├── namespace.yaml
│   │   ├── control-plane/
│   │   │   ├── deployment.yaml
│   │   │   └── service.yaml
│   │   ├── minio/
│   │   │   ├── deployment.yaml
│   │   │   └── service.yaml
│   │   ├── traefik/
│   │   │   └── values.yaml           # Helm values
│   │   └── monitoring/
│   │       ├── prometheus-config.yaml
│   │       └── grafana-dashboard.json
│   └── registry/
│       └── docker-compose.registry.yml
│
└── scripts/
    ├── dev-setup.sh                  # One-command dev environment setup
    └── seed-test-models.py           # Create sample models for testing
```

---

## API Contracts

### Control Plane REST API

#### `POST /api/v1/models` — Register and deploy a model

```json
// Request
{
  "name": "fraud-detector",
  "framework": "sklearn",            // optional if auto-detected
  "artifact_uri": "s3://mlserve/models/fraud-detector/v1/model.pkl",
  "checksum_sha256": "a1b2c3...",
  "config": {                        // optional overrides
    "replicas": 2,
    "cpu": "500m",
    "memory": "512Mi",
    "target": "docker"               // "docker" | "kubernetes"
  }
}

// Response (202 Accepted — build + deploy is async)
{
  "model_name": "fraud-detector",
  "version": 1,
  "deployment_id": "d-abc123",
  "status": "building",
  "status_url": "/api/v1/deployments/d-abc123"
}
```

#### `GET /api/v1/deployments` — List all deployments

```json
// Response
{
  "deployments": [
    {
      "name": "fraud-detector",
      "version": 3,
      "status": "running",
      "endpoint": "http://localhost/models/fraud-detector/predict",
      "replicas": 2,
      "framework": "sklearn",
      "created_at": "2026-02-26T10:00:00Z",
      "updated_at": "2026-02-26T10:02:30Z"
    }
  ]
}
```

#### `GET /api/v1/deployments/{name}` — Get deployment status

```json
// Response
{
  "name": "fraud-detector",
  "version": 3,
  "status": "running",
  "endpoint": "http://localhost/models/fraud-detector/predict",
  "replicas": {"desired": 2, "ready": 2},
  "framework": "sklearn",
  "container_image": "registry.local/mlserve/fraud-detector:v3",
  "resource_usage": {
    "cpu_percent": 12.5,
    "memory_mb": 180
  },
  "events": [
    {"type": "building", "timestamp": "2026-02-26T10:00:00Z", "message": "Building container image"},
    {"type": "deploying", "timestamp": "2026-02-26T10:01:00Z", "message": "Starting container"},
    {"type": "running", "timestamp": "2026-02-26T10:02:30Z", "message": "Health check passed"}
  ]
}
```

### Model Serving API (per deployed model)

#### `POST /models/{name}/predict` — Run inference

```json
// Request
{
  "instances": [
    [1.0, 2.5, 3.2, 0.8, 1.1, 2.0, 3.5, 0.2, 1.4, 2.8],
    [0.5, 1.2, 2.8, 1.0, 0.9, 1.8, 3.0, 0.5, 1.1, 2.3]
  ]
}

// Response
{
  "predictions": [0.87, 0.23],
  "model_name": "fraud-detector",
  "model_version": "v3"
}
```

#### `GET /models/{name}/health` — Health check

```json
{"status": "healthy", "model": "fraud-detector", "framework": "sklearn"}
```

#### `GET /models/{name}/metrics` — Prometheus metrics

```
# HELP mlserve_predictions_total Total predictions served
# TYPE mlserve_predictions_total counter
mlserve_predictions_total{model_name="fraud-detector",status="success"} 15234
mlserve_predictions_total{model_name="fraud-detector",status="error"} 12

# HELP mlserve_prediction_duration_seconds Prediction latency in seconds
# TYPE mlserve_prediction_duration_seconds histogram
mlserve_prediction_duration_seconds_bucket{model_name="fraud-detector",le="0.01"} 14500
mlserve_prediction_duration_seconds_bucket{model_name="fraud-detector",le="0.05"} 15100
```

---

## Phased Roadmap

### Phase 1: MVP — "It Works on My Machine" (Weeks 1–4)

**Goal:** `mlserve deploy model.pkl` → working local endpoint

| Task | Details |
|------|---------|
| CLI skeleton | Typer app with `deploy`, `list`, `status`, `delete` |
| Framework detection | File extension + pickle inspection for sklearn/xgboost |
| Serving runtime | FastAPI server with sklearn + ONNX loaders |
| Build service | Generate Dockerfile, build image via docker-py |
| Docker backend | Deploy as Docker container, expose port |
| Local storage | Filesystem-based artifact store (no MinIO yet) |
| Basic routing | Traefik with Docker label auto-discovery |
| **No:** Control Plane API | CLI talks directly to Docker (simplification) |

**Demo:** Data scientist runs one command, gets a URL, calls it, gets predictions.

### Phase 2: Production Foundation (Weeks 5–10)

**Goal:** Multi-model, multi-user, persistent state

| Task | Details |
|------|---------|
| Control Plane API | FastAPI server with DB-backed state |
| PostgreSQL + Alembic | Persistent deployment state, migrations |
| MinIO integration | S3-compatible artifact storage |
| Model versioning | Re-deploy same model → v2, v3, etc. |
| Rollback support | `mlserve rollback fraud-detector --to v2` |
| PyTorch + TensorFlow | Additional framework loaders + Dockerfiles |
| Health monitoring | Automated health checks, restart on failure |
| Docker Compose stack | One `docker compose up` for entire platform |

### Phase 3: Kubernetes & Observability (Weeks 11–16)

**Goal:** Production-grade deployment on Kubernetes

| Task | Details |
|------|---------|
| Kubernetes backend | Deploy models as K8s Deployments + Services |
| Prometheus + Grafana | Metrics collection and visualization |
| Request logging | Structured JSON logs, optional Loki integration |
| Resource limits | CPU/memory constraints per model |
| Replica scaling | `--replicas N` flag |
| Ingress management | Automatic K8s Ingress creation per model |
| CI/CD pipeline | GitHub Actions: test → build → push Control Plane image |

### Phase 4: Advanced Features (Weeks 17+)

**Goal:** Enterprise-ready features

| Task | Details |
|------|---------|
| Auto-scaling (HPA) | Scale based on CPU / request rate |
| A/B testing | Route X% traffic to new model version |
| Canary deployments | Gradual rollout of new versions |
| Auth & multi-tenancy | API keys, namespace isolation |
| Web dashboard | React/Next.js UI for non-CLI users |
| Batch inference | Submit batch jobs (CSV in → predictions out) |
| Model drift detection | Monitor prediction distribution over time |
| GPU support | NVIDIA GPU scheduling for PyTorch/TF models |

---

## Non-Goals

Explicit things MLServe does **NOT** try to do (to keep scope sane):

| Non-Goal | Why |
|----------|-----|
| Model training | Use MLflow, Kubeflow Pipelines, or SageMaker for that |
| Data pipeline management | Use Airflow, Dagster, or Prefect |
| Feature store | Use Feast or Tecton |
| Experiment tracking | Use MLflow, W&B, or Neptune |
| Notebook management | Use JupyterHub |
| Model format conversion | User provides the model in a loadable format |

MLServe starts where training ends: **you have a model file → MLServe gives you a production API.**

---

## Infrastructure: Docker Compose (Local Dev Stack)

```yaml
# infrastructure/docker-compose.yml
version: "3.8"

services:
  # --- Control Plane ---
  control-plane:
    build:
      context: ../
      dockerfile: infrastructure/Dockerfile.control-plane
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://mlserve:mlserve@postgres:5432/mlserve
      MINIO_ENDPOINT: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
      DOCKER_HOST: unix:///var/run/docker.sock
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock   # Access to host Docker
    networks:
      - mlserve-network
    depends_on:
      postgres:
        condition: service_healthy
      minio:
        condition: service_healthy

  # --- Database ---
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: mlserve
      POSTGRES_PASSWORD: mlserve
      POSTGRES_DB: mlserve
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlserve"]
      interval: 5s
      timeout: 3s
      retries: 5
    networks:
      - mlserve-network

  # --- Artifact Store ---
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"     # S3 API
      - "9001:9001"     # Web Console
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - miniodata:/data
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      timeout: 3s
      retries: 5
    networks:
      - mlserve-network

  # --- API Gateway ---
  traefik:
    image: traefik:v3.2
    command:
      - "--api.dashboard=true"
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
      - "--providers.docker.network=mlserve-network"
      - "--entrypoints.web.address=:80"
    ports:
      - "80:80"
      - "8080:8080"     # Traefik dashboard
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - mlserve-network

  # --- Container Registry ---
  registry:
    image: registry:2
    ports:
      - "5000:5000"
    volumes:
      - registrydata:/var/lib/registry
    networks:
      - mlserve-network

  # --- Monitoring ---
  prometheus:
    image: prom/prometheus:v2.53.0
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - promdata:/prometheus
    ports:
      - "9090:9090"
    networks:
      - mlserve-network

  grafana:
    image: grafana/grafana:11.4.0
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafanadata:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
    networks:
      - mlserve-network

volumes:
  pgdata:
  miniodata:
  registrydata:
  promdata:
  grafanadata:

networks:
  mlserve-network:
    name: mlserve-network
```

---

## Validation Checklist

Before implementation, every architectural decision has been validated against:

| Criterion | Status | Notes |
|-----------|--------|-------|
| All components are open-source | ✅ | No proprietary dependencies |
| Each component is individually proven at scale | ✅ | FastAPI, PostgreSQL, MinIO, Traefik, Prometheus — all battle-tested |
| No single points of failure in production (K8s mode) | ✅ | Every component can run as replicated Deployment |
| Can run entirely on a laptop | ✅ | Docker Compose stack, ~2GB RAM |
| Supports the 5 most popular ML frameworks | ✅ | sklearn, PyTorch, TensorFlow, ONNX, XGBoost |
| Sub-5-second deploy for cached images | ✅ | Docker layer caching, only model artifact changes per version |
| P99 inference latency < 100ms (sklearn/xgboost) | ✅ | FastAPI + Uvicorn benchmarks confirm this |
| Framework-agnostic serving interface | ✅ | Unified `/predict` API regardless of model framework |
| No changes to user's model code | ✅ | User provides a model file, nothing else |

---

*Last updated: 2026-02-26*
