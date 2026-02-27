from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class SessionIoInfo:
    inputs: list[ort.NodeArg]
    outputs: list[ort.NodeArg]


@dataclass(frozen=True)
class SttProfile:
    name: str
    hf_id: str
    n_mels: int
    encoder_input_dtype: np.dtype
    attention_mask_dtype: np.dtype
    mask_style: str
    logits_dtype: np.dtype
    pid_style: str


SMALL_QUANT_PROFILE = SttProfile(
    name="small-quantized",
    hf_id="openai/whisper-small",
    n_mels=80,
    encoder_input_dtype=np.uint16,
    attention_mask_dtype=np.uint16,
    mask_style="small-quantized",
    logits_dtype=np.uint16,
    pid_style="reverse",
)

TURBO_PROFILE = SttProfile(
    name="large-v3-turbo",
    hf_id="openai/whisper-large-v3-turbo",
    n_mels=128,
    encoder_input_dtype=np.float16,
    attention_mask_dtype=np.float16,
    mask_style="additive-f16",
    logits_dtype=np.float16,
    pid_style="reverse",
)


def _ort_type_to_np_dtype(ort_type: str) -> np.dtype | None:
    t = (ort_type or "").lower()
    if "uint8" in t:
        return np.uint8
    if "uint16" in t:
        return np.uint16
    if "int32" in t:
        return np.int32
    if "int64" in t:
        return np.int64
    if "float16" in t:
        return np.float16
    if "float" in t:
        return np.float32
    return None


def detect_profile(
    encoder_session: ort.InferenceSession,
    decoder_session: ort.InferenceSession,
    encoder_io: SessionIoInfo,
    decoder_io: SessionIoInfo,
) -> SttProfile:
    _ = encoder_session, decoder_session
    enc_input = next((n for n in encoder_io.inputs if "input_features" in n.name.lower()), None)
    if enc_input is None:
        raise RuntimeError("Unable to detect STT profile: encoder input_features not found.")

    enc_dtype = _ort_type_to_np_dtype(enc_input.type or "")
    enc_shape = tuple(enc_input.shape or [])
    mel_dim = int(enc_shape[1]) if len(enc_shape) > 1 and isinstance(enc_shape[1], int) else None

    dec_logits = next((n for n in decoder_io.outputs if "logits" in n.name.lower()), None)
    logits_dtype = dec_logits.type if dec_logits is not None else "unknown"

    if enc_dtype == np.float16 and mel_dim == 128:
        return TURBO_PROFILE
    if enc_dtype == np.uint16 and mel_dim == 80:
        return SMALL_QUANT_PROFILE

    raise RuntimeError(
        "Unable to detect STT profile from model IO. "
        f"encoder input_features dtype={enc_input.type}, shape={enc_shape}; "
        f"decoder logits dtype={logits_dtype}."
    )
