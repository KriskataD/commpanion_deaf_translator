"""ONNX Runtime helpers for QNN execution."""
from __future__ import annotations

from pathlib import Path

import onnxruntime as ort


def get_providers() -> list[str]:
    """Return the available ONNX Runtime providers."""
    return ort.get_available_providers()


def make_session(onnx_path: str | Path) -> ort.InferenceSession:
    """Create an ONNX Runtime session with QNN preferred, CPU fallback."""
    onnx_path = str(onnx_path)
    available = get_providers()
    if "QNNExecutionProvider" in available:
        print("✅ QNNExecutionProvider is available.")
    else:
        print("⚠️ QNNExecutionProvider not available; falling back to CPUExecutionProvider.")

    sess_options = ort.SessionOptions()
    providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
