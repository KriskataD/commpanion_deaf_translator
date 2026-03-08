from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from easyocr.craft_utils import getDetBoxes
from easyocr.utils import group_text_box, four_point_transform

from src.npu.ort_qnn import make_session

# These match Qualcomm EasyOCR defaults (good starting point) :contentReference[oaicite:4]{index=4}
DETECTOR_ARGS = dict(
    text_threshold=0.7,
    link_threshold=0.4,
    low_text=0.4,
    poly=False,
    slope_ths=0.1,
    ycenter_ths=0.5,
    height_ths=0.5,
    width_ths=0.5,
    add_margin=0.1,
    min_size=20,
)

# A robust fallback charset (ASCII-ish). If your model uses a different charset,
# this still gives a usable demo; you can improve it later.
_FALLBACK_CHARS = (
    '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
)

@dataclass
class OcrResult:
    box: object  # either (xmin,xmax,ymin,ymax) or 4-corners
    text: str
    confidence: float

def load_charset(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def _resize_pad(img: np.ndarray, target_hw: Tuple[int, int], *, pad_value: int = 0, h_align: str = "center"):
    """Aspect-preserving resize + padding to (H,W). Returns padded_img, scale, (pad_x, pad_y)."""
    th, tw = target_hw
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image")

    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    if img.ndim == 2:
        canvas = np.full((th, tw), pad_value, dtype=img.dtype)
    else:
        canvas = np.full((th, tw, img.shape[2]), pad_value, dtype=img.dtype)

    pad_y = (th - nh) // 2
    if h_align == "left":
        pad_x = 0
    elif h_align == "right":
        pad_x = (tw - nw)
    else:
        pad_x = (tw - nw) // 2

    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, (pad_x, pad_y)

def _ctc_greedy_decode(logits: np.ndarray, chars: str, blank_idx: int = 0) -> Tuple[str, float]:
    probs = _softmax(logits, axis=1)
    idx = np.argmax(probs, axis=1)
    conf_t = np.max(probs, axis=1)

    # If chars length == 96: indices 1..96 map to chars[0..95]
    # If chars length == 97 and includes blank at 0: indices 0..96 map to chars[0..96] (skip blank)
    if len(chars) == 97:
        # assume chars[0] is blank-like; map i -> chars[i]
        def map_idx(i: int) -> str | None:
            if i == blank_idx:
                return None
            return chars[i]
    else:
        def map_idx(i: int) -> str | None:
            if i == blank_idx:
                return None
            j = i - 1
            if 0 <= j < len(chars):
                return chars[j]
            return None

    out, out_confs = [], []
    prev = -1
    for i, p in zip(idx, conf_t):
        i = int(i)
        if i == blank_idx:
            prev = i
            continue
        if i == prev:
            continue
        ch = map_idx(i)
        if ch is not None:
            out.append(ch)
            out_confs.append(float(p))
        prev = i

    text = "".join(out).strip()
    confidence = float(np.mean(out_confs)) if out_confs else 0.0
    return text, confidence

class EasyOcrQnn:
    def __init__(
        self,
        detector_onnx: str | Path,
        recognizer_onnx: str | Path,
        detector_hw: Tuple[int, int] = (608, 800),
        recognizer_hw: Tuple[int, int] = (64, 800),
        chars: Optional[str] = None,
        qnn_only: bool = True,
    ) -> None:
        self.detector_hw = detector_hw
        self.recognizer_hw = recognizer_hw
        self.chars = chars or load_charset("src/ocr/charset_en.txt")

        providers = ["QNNExecutionProvider"] if qnn_only else ["QNNExecutionProvider", "CPUExecutionProvider"]
        self.det_sess = make_session(detector_onnx, providers=providers)
        self.rec_sess = make_session(recognizer_onnx, providers=providers)

        self.det_in = self.det_sess.get_inputs()[0].name
        self.det_out = self.det_sess.get_outputs()[0].name
        self.rec_in = self.rec_sess.get_inputs()[0].name
        self.rec_out = self.rec_sess.get_outputs()[0].name

    def _detect_boxes(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]], List[Tuple[Tuple[int,int],Tuple[int,int],Tuple[int,int],Tuple[int,int]]]]:
        """
        Returns: gray_original, horizontal_boxes, free_boxes
        """
        h0, w0 = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Detector expects RGB float in [0,1], NCHW with target (608,800) :contentReference[oaicite:5]{index=5}
        pad_val = int(rgb[0,0,0])
        rgb_pad, scale, (pad_x, pad_y) = _resize_pad(rgb, self.detector_hw, pad_value=pad_val, h_align="center")
        det_in = (rgb_pad.astype(np.float32) / 255.0).transpose(2,0,1)[None, ...]

        det_map = self.det_sess.run([self.det_out], {self.det_in: det_in})[0]  # (1,304,400,2) for float model
        out = det_map[0]
        score_text = out[:, :, 0]
        score_link = out[:, :, 1]

        # CRAFT postprocess → boxes in output-map coordinates :contentReference[oaicite:6]{index=6}
        horizontal_boxes, free_boxes, _ = getDetBoxes(
            score_text, score_link,
            text_threshold=DETECTOR_ARGS["text_threshold"],
            link_threshold=DETECTOR_ARGS["link_threshold"],
            low_text=DETECTOR_ARGS["low_text"],
            poly=DETECTOR_ARGS["poly"],
        )

        # Convert boxes -> original image coords:
        # output-map coords are ~1/2 of network input coords → multiply by 2 first
        detections = []
        for i in range(len(horizontal_boxes)):
            box = free_boxes[i] if free_boxes[i] is not None else horizontal_boxes[i]
            if box is None:
                continue
            pts = np.array(box, dtype=np.float32).reshape(-1,2)
            pts *= 2.0
            pts[:,0] -= float(pad_x)
            pts[:,1] -= float(pad_y)
            pts /= float(scale)
            detections.append(pts.reshape(-1).astype(np.int32))

        # Grouping/cleanup (same idea as Qualcomm app) :contentReference[oaicite:7]{index=7}
        horizontal_list_raw, free_list_raw = group_text_box(
            detections,
            slope_ths=DETECTOR_ARGS["slope_ths"],
            ycenter_ths=DETECTOR_ARGS["ycenter_ths"],
            height_ths=DETECTOR_ARGS["height_ths"],
            width_ths=DETECTOR_ARGS["width_ths"],
            add_margin=DETECTOR_ARGS["add_margin"],
        )

        horizontal_list = [tuple(map(int, x)) for x in horizontal_list_raw]  # (xmin,xmax,ymin,ymax)
        free_list = [
            (tuple(map(int, a)), tuple(map(int, b)), tuple(map(int, c)), tuple(map(int, d)))
            for (a,b,c,d) in free_list_raw
        ]

        # Min-size filtering
        min_size = int(DETECTOR_ARGS["min_size"])
        if min_size > 0:
            horizontal_list = [b for b in horizontal_list if max(b[1]-b[0], b[3]-b[2]) > min_size]
            free_list = [
                b for b in free_list
                if max(max(p[0] for p in b)-min(p[0] for p in b),
                       max(p[1] for p in b)-min(p[1] for p in b)) > min_size
            ]

        # Clip horizontals to original image bounds
        horiz_clipped = []
        for (xmin, xmax, ymin, ymax) in horizontal_list:
            xmin = max(0, min(xmin, w0-1))
            xmax = max(0, min(xmax, w0))
            ymin = max(0, min(ymin, h0-1))
            ymax = max(0, min(ymax, h0))
            if xmax > xmin and ymax > ymin:
                horiz_clipped.append((xmin, xmax, ymin, ymax))

        return gray, horiz_clipped, free_list

    def _recognize_cutout(self, cutout_gray: np.ndarray) -> OcrResult:
        # Recognizer expects 1x1x64x800 float :contentReference[oaicite:8]{index=8}
        pad_val = int(cutout_gray[0,0]) if cutout_gray.size else 0
        pad_img, _, _ = _resize_pad(cutout_gray, self.recognizer_hw, pad_value=pad_val, h_align="left")
        rec_in = (pad_img.astype(np.float32) / 255.0)[None, None, :, :]

        out = self.rec_sess.run([self.rec_out], {self.rec_in: rec_in})[0]  # (1,199,97)
        logits = out[0]  # (T,C)
        text, conf = _ctc_greedy_decode(logits, self.chars, blank_idx=0)
        return OcrResult(box=None, text=text, confidence=conf)

    def readtext(self, frame_bgr: np.ndarray) -> List[OcrResult]:
        gray, horizontal_boxes, free_boxes = self._detect_boxes(frame_bgr)

        cutouts: List[Tuple[np.ndarray, object, int]] = []

        # free boxes
        for fb in free_boxes:
            rect = np.array(fb, dtype="float32")
            cut = four_point_transform(gray, rect)
            if 0 in cut.shape:
                continue
            y_min = int(min(p[1] for p in fb))
            cutouts.append((cut, fb, y_min))

        # horizontal boxes
        for (xmin, xmax, ymin, ymax) in horizontal_boxes:
            cut = gray[ymin:ymax, xmin:xmax]
            if 0 in cut.shape:
                continue
            cutouts.append((cut, (xmin, xmax, ymin, ymax), ymin))

        # Sort top-to-bottom
        cutouts.sort(key=lambda x: x[2])

        results: List[OcrResult] = []
        for cut, box, _ in cutouts:
            r = self._recognize_cutout(cut)
            if r.text:
                r.box = box
                results.append(r)

        return results