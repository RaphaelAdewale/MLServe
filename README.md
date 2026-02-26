# MLServe

> **"Give me a model, I'll give you a production API."**

A self-service platform that lets data scientists deploy ML models with a single CLI command.

```bash
$ mlserve deploy model.pkl --name fraud-detector

✓ Framework detected: sklearn
✓ Container built: registry.local/mlserve/fraud-detector:v1
✓ Deployed successfully

  Endpoint: http://localhost/models/fraud-detector/predict
  Status:   running
  Replicas: 1
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design document, including:

- System architecture diagrams
- Component deep dives (CLI, Control Plane, Build Service, Serving Runtime, Orchestrator, Gateway, Monitoring)
- Complete data flow for `mlserve deploy`
- Technology choices with justification
- API contracts
- Phased roadmap (MVP → Production → Kubernetes → Advanced)
- Project structure

## Supported Frameworks

| Framework | File Types | Status |
|-----------|-----------|--------|
| scikit-learn | `.pkl`, `.joblib` | MVP |
| ONNX | `.onnx` | MVP |
| PyTorch | `.pt`, `.pth` | V1 |
| TensorFlow/Keras | `.h5`, `.keras`, `saved_model/` | V1 |
| XGBoost | `.bst`, `.pkl`, `.joblib` | V1 |

## Quick Start

```bash
# Start the platform (Docker Compose)
docker compose -f infrastructure/docker-compose.yml up -d

# Deploy a model
pip install mlserve
mlserve deploy my_model.pkl --name my-model

# Hit the endpoint
curl -X POST http://localhost/models/my-model/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1.0, 2.0, 3.0]]}'
```

## License

MIT
