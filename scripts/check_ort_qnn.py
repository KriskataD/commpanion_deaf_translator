"""Utility to check ONNX Runtime QNN provider availability."""
from __future__ import annotations

import onnxruntime as ort


def main() -> None:
    providers = ort.get_available_providers()
    has_qnn = "QNNExecutionProvider" in providers
    print(f"onnxruntime version: {ort.__version__}")
    print(f"available providers: {providers}")
    print(f"has_qnn_provider: {has_qnn}")


if __name__ == "__main__":
    main()
