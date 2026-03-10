import time
from pathlib import Path

import cv2
import numpy as np

from src.ocr.easyocr_qnn import EasyOcrQnn


def _open_camera(cam_id: int = 0):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    raise RuntimeError(f"Could not open camera {cam_id} with CAP_DSHOW")


def _sharpness_score(frame_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def capture_best_frame(cap, warmup=5, burst=10, sleep_s=0.03):
    for _ in range(warmup):
        cap.read()
        time.sleep(sleep_s)

    best = None
    best_s = -1.0
    for _ in range(burst):
        ok, f = cap.read()
        if not ok:
            continue
        s = _sharpness_score(f)
        if s > best_s:
            best_s = s
            best = f.copy()
        time.sleep(sleep_s)

    if best is None:
        raise RuntimeError("Failed to capture a frame")
    return best, best_s


class OcrScanner:
    def __init__(
        self,
        detector_onnx: str,
        recognizer_onnx: str,
        charset_path: str = "src/ocr/charset_en.txt",
        camera_id: int = 0,
    ):
        self.ocr = EasyOcrQnn(detector_onnx, recognizer_onnx, qnn_only=True, chars=None)
        # Ensure EasyOcrQnn loads charset_en.txt internally as you patched.
        self.camera_id = camera_id

    def scan_once(self, save_debug: bool = True) -> str:
        cap = _open_camera(self.camera_id)
        # Prefer HD for far text
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        try:
            frame, sharp = capture_best_frame(cap)
        finally:
            cap.release()

        if save_debug:
            Path("captured").mkdir(exist_ok=True)
            ts = int(time.time() * 1000)
            cv2.imwrite(f"captured/detect_{ts}.jpg", frame)

        results = self.ocr.readtext(frame)
        text = "\n".join(r.text for r in results if r.text.strip()).strip()
        return text