#!/usr/bin/env python3
"""Generate small test model fixtures for unit/integration tests.

Run: python tests/fixtures/generate_models.py

Requires: scikit-learn, numpy  (install in a venv if not present)
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "models"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_sklearn_pickle():
    """Create a minimal .pkl file that joblib can load.

    Rather than depend on scikit-learn at test time we just create
    a plain-bytes file with .pkl extension.  The unit tests only
    care about *detecting* the framework from the extension – they
    don't load the model.
    """
    path = FIXTURES_DIR / "tiny_model.pkl"
    # A minimal pickle-protocol-2 blob that unpickles to the dict {"v": 1}
    # pickle.dumps({"v": 1}, protocol=2) → b'\x80\x02}q\x00U\x01vq\x01K\x01s.'
    path.write_bytes(b"\x80\x02}q\x00U\x01vq\x01K\x01s.")
    print(f"  ✓ {path}")


def generate_joblib():
    """Create a tiny .joblib file (same content, different extension)."""
    path = FIXTURES_DIR / "tiny_model.joblib"
    path.write_bytes(b"\x80\x02}q\x00U\x01vq\x01K\x01s.")
    print(f"  ✓ {path}")


def generate_onnx():
    """Create a minimal valid-ish .onnx stub.

    Real ONNX files are protobuf, but for framework-detection tests
    we only need the right extension. Write the ONNX magic bytes.
    """
    path = FIXTURES_DIR / "tiny_model.onnx"
    # Minimal protobuf-like header (not a valid model, but enough for detection tests)
    path.write_bytes(b"\x08\x07\x12\x04test")
    print(f"  ✓ {path}")


def generate_unknown():
    """Create a file with an unsupported extension."""
    path = FIXTURES_DIR / "unknown_model.txt"
    path.write_text("not a real model")
    print(f"  ✓ {path}")


def generate_no_extension():
    """Create a file with no extension."""
    path = FIXTURES_DIR / "modelfile"
    path.write_bytes(b"\x00")
    print(f"  ✓ {path}")


if __name__ == "__main__":
    print("Generating test model fixtures…")
    generate_sklearn_pickle()
    generate_joblib()
    generate_onnx()
    generate_unknown()
    generate_no_extension()
    print("Done.")
