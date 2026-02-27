# MLServe

> **Give me a model, I'll give you a production API.**

MLServe takes a trained ML model file, packages it into a Docker container with a FastAPI prediction server, deploys it, and routes traffic through Traefik — all in one command.

Supports **scikit-learn** (`.pkl`, `.joblib`) and **ONNX** (`.onnx`).

---

## Quick Start

**Prerequisites:** Python 3.12+, Docker (running), [Poetry](https://python-poetry.org/docs/#installation)

```bash
# Setup
git clone https://github.com/RaphaelAdewale/MLServe.git
cd MLServe
poetry install

# Start infrastructure (Traefik + Docker Registry)
make infra-up

# Start the MLServe API server
poetry run uvicorn mlserve.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open a **second terminal** and deploy a model:

```bash
# Create a test model (needs scikit-learn: pip install scikit-learn joblib)
python -c "
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib
X, y = load_iris(return_X_y=True)
joblib.dump(LogisticRegression(max_iter=200).fit(X, y), 'iris_model.pkl')
print('Created iris_model.pkl')
"

# Deploy it
curl -s -X POST http://localhost:8000/api/v1/models/deploy \
  -F "file=@iris_model.pkl" \
  -F "name=iris-classifier" | python3 -m json.tool

# Make a prediction
curl -s -X POST http://localhost/models/iris-classifier/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2]]}' | python3 -m json.tool
```

```json
{ "predictions": [0], "model_name": "iris-classifier", "model_version": "1" }
```

```bash
# Clean up
curl -s -X DELETE http://localhost:8000/api/v1/deployments/iris-classifier
make infra-down
```

---

## API Endpoints

**Control Plane** — `localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/models/deploy` | Deploy a model (multipart upload) |
| `GET` | `/api/v1/deployments` | List all deployments |
| `GET` | `/api/v1/deployments/{name}` | Deployment details |
| `DELETE` | `/api/v1/deployments/{name}` | Delete a deployment |
| `GET` | `/docs` | Interactive Swagger UI |

**Model Containers** — via Traefik on `localhost`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/models/{name}/predict` | Run inference |
| `GET` | `/models/{name}/health` | Health check |
| `GET` | `/models/{name}/metrics` | Prometheus metrics |

---

## Project Structure

```
src/mlserve/
├── api/            # FastAPI control plane (routes, schemas, dependencies)
├── cli/            # Typer CLI with Rich output
├── core/           # Config, database, ORM models, exceptions
├── services/       # Business logic (build, deploy, orchestrate)
├── runtime/        # Prediction server (runs inside containers, not on host)
└── templates/      # Jinja2 Dockerfile template
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design document.

---

## Tech Stack

FastAPI · Typer · SQLAlchemy 2.0 (async) · Docker SDK · Traefik · Jinja2 · Prometheus · Rich

---

## Development

```bash
make test       # Run tests (20 unit tests)
make lint       # Ruff linter
make check      # Lint + tests
make format     # Auto-format
make clean      # Remove caches and DB
```

---

## License

MIT
