"""ONNX Runtime helpers for QNN execution."""
from __future__ import annotations

import os
from pathlib import Path

import onnxruntime as ort


def get_providers() -> list[str]:
    """Return the available ONNX Runtime providers."""
    return ort.get_available_providers()


def make_session(
    onnx_path: str | Path,
    providers: list[str] | None = None,
    sess_options: ort.SessionOptions | None = None,
) -> ort.InferenceSession:
    """Create an ONNX Runtime session with QNN only by default."""
    onnx_path = str(onnx_path)
    available = get_providers()
    providers = providers or ["QNNExecutionProvider"]
    qnn_backend_path = os.getenv("QNN_BACKEND_PATH")
    if "QNNExecutionProvider" in providers:
        if "QNNExecutionProvider" not in available:
            raise RuntimeError(
                "QNNExecutionProvider is required but not available. "
                "Ensure the Qualcomm QNN runtime is installed and ONNX Runtime "
                "is built with QNN support."
            )
        print("✅ QNNExecutionProvider is available.")

    sess_options = sess_options or ort.SessionOptions()
    ort_providers: list[str] | list[tuple[str, dict[str, str]]] = providers
    if qnn_backend_path and "QNNExecutionProvider" in providers:
        ort_providers = [
            ("QNNExecutionProvider", {"backend_path": qnn_backend_path}),
            *[provider for provider in providers if provider != "QNNExecutionProvider"],
        ]
        print(f"✅ Using QNN backend path: {qnn_backend_path}")
    return ort.InferenceSession(onnx_path, sess_options=sess_options, providers=ort_providers)
