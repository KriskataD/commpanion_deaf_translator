"""ONNX Runtime helpers for QNN execution."""
from __future__ import annotations

from pathlib import Path

import onnxruntime as ort


def get_providers() -> list[str]:
    """Return the available ONNX Runtime providers."""
    return ort.get_available_providers()


def make_session(
    onnx_path: str | Path,
    providers: list[str] | None = None,
) -> ort.InferenceSession:
    """Create an ONNX Runtime session with QNN only by default."""
    onnx_path = str(onnx_path)
    available = get_providers()
    providers = providers or ["QNNExecutionProvider"]
    if "QNNExecutionProvider" in providers:
        if "QNNExecutionProvider" not in available:
            raise RuntimeError(
                "QNNExecutionProvider is required but not available. "
                "Ensure the Qualcomm QNN runtime is installed and ONNX Runtime "
                "is built with QNN support."
            )
        print("✅ QNNExecutionProvider is available.")

    sess_options = ort.SessionOptions()
    return ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
