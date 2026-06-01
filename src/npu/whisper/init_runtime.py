from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperTokenizer

from ..ort_qnn import make_session
from .profiles import (
    SMALL_QUANT_PROFILE,
    TURBO_PROFILE,
    SessionIoInfo,
    detect_profile,
)


def dump_model_io(encoder_dir: str | Path, decoder_dir: str | Path) -> None:
    """Print encoder/decoder IO metadata without constructing the full STT class."""
    encoder_dir = Path(encoder_dir)
    decoder_dir = Path(decoder_dir)
    encoder_onnx = encoder_dir / "model.onnx"
    decoder_onnx = decoder_dir / "model.onnx"
    if not encoder_onnx.exists():
        raise FileNotFoundError(f"Missing ONNX model: {encoder_onnx}")
    if not decoder_onnx.exists():
        raise FileNotFoundError(f"Missing ONNX model: {decoder_onnx}")

    encoder_session = make_session(encoder_onnx)
    decoder_session = make_session(decoder_onnx)

    print("\nEncoder inputs:")
    for node in encoder_session.get_inputs():
        print(f"  - {node.name}: shape={node.shape}, type={node.type}")
    print("Encoder outputs:")
    for node in encoder_session.get_outputs():
        print(f"  - {node.name}: shape={node.shape}, type={node.type}")

    print("\nDecoder inputs:")
    for node in decoder_session.get_inputs():
        print(f"  - {node.name}: shape={node.shape}, type={node.type}")
    print("Decoder outputs:")
    for node in decoder_session.get_outputs():
        print(f"  - {node.name}: shape={node.shape}, type={node.type}")


class WhisperInitRuntimeMixin:
    def __init__(
        self,
        encoder_dir: str | Path,
        decoder_dir: str | Path,
        stt_model: str = "auto",
        prefer_qnn: bool = True,
        debug: bool = False,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.encoder_dir = Path(encoder_dir)
        self.decoder_dir = Path(decoder_dir)
        self.stt_model = stt_model
        self.prefer_qnn = prefer_qnn
        self.debug = debug
        self.debug_kv = debug

        self._resolve_model_paths_and_validate()
        self._create_sessions()
        self._init_io_info()
        self._select_profile()
        self._init_feature_extractor_and_tokenizer()
        self._discover_io_names_and_cache_metadata()

        if self.debug:
            self.logger.info("Providers encoder: %s", self.encoder_session.get_providers())
            self.logger.info("Providers decoder: %s", self.decoder_session.get_providers())
            self.logger.info(
                "attn_max_len=%d self_cache_len=%d has_kv_cache=%s",
                self.attn_max_len, self.self_cache_len, self.has_kv_cache
            )

    def _select_profile(self) -> None:
        if self.stt_model == "auto":
            self.profile = detect_profile(
                self.encoder_session,
                self.decoder_session,
                self.encoder_io,
                self.decoder_io,
            )
        elif self.stt_model == "small-quantized":
            self.profile = SMALL_QUANT_PROFILE
        elif self.stt_model == "large-v3-turbo":
            self.profile = TURBO_PROFILE
        else:
            raise ValueError(f"Unsupported stt_model={self.stt_model!r}")

        encoder_node = self.encoder_io.inputs[0] if self.encoder_io.inputs else None
        attention_node = next((n for n in self.decoder_io.inputs if "attention_mask" in n.name.lower()), None)
        logits_node = next((n for n in self.decoder_io.outputs if "logits" in n.name.lower()), None)
        self.logger.info(
            "Selected STT profile: name=%s hf_id=%s n_mels=%d encoder_dtype=%s mask_dtype=%s logits_dtype=%s",
            self.profile.name,
            self.profile.hf_id,
            self.profile.n_mels,
            encoder_node.type if encoder_node is not None else "unknown",
            attention_node.type if attention_node is not None else "unknown",
            logits_node.type if logits_node is not None else "unknown",
        )

    def _resolve_model_paths_and_validate(self) -> None:
        self.encoder_onnx = self.encoder_dir / "model.onnx"
        self.decoder_onnx = self.decoder_dir / "model.onnx"

        self.logger.info("QNN encoder model path: %s", self.encoder_onnx)
        self.logger.info("QNN decoder model path: %s", self.decoder_onnx)

        self._validate_model_files(self.encoder_onnx)
        self._validate_model_files(self.decoder_onnx)

    def _create_sessions(self) -> None:
        try:
            self.encoder_session = make_session(self.encoder_onnx, providers=self._get_encoder_providers())
            self.decoder_session = make_session(self.decoder_onnx, providers=self._get_decoder_providers())
        except Exception:
            self.logger.exception("Failed to create ONNX Runtime sessions.")
            raise

    def _init_io_info(self) -> None:
        self.encoder_io = SessionIoInfo(
            inputs=self.encoder_session.get_inputs(),
            outputs=self.encoder_session.get_outputs(),
        )
        self.decoder_io = SessionIoInfo(
            inputs=self.decoder_session.get_inputs(),
            outputs=self.decoder_session.get_outputs(),
        )

        if self.debug:
            self.logger.info("=== Decoder outputs declared by ONNX (get_outputs) ===")
            for node in self.decoder_io.outputs:
                self.logger.info("OUT %-35s shape=%s type=%s", node.name, node.shape, node.type)

    def _init_feature_extractor_and_tokenizer(self) -> None:
        # Use the model's actual preprocessor_config.json for BOTH profiles.
        # (large-v3-turbo has feature_size=128 etc in its HF config)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.profile.hf_id)

        self.tokenizer = WhisperTokenizer.from_pretrained(self.profile.hf_id)
        self.config = WhisperConfig.from_pretrained(self.profile.hf_id)

        # Your updated token_selection / decoder_runtime should rely on these
        self.suppress_blank = True

        # ---- suppress_tokens: match Whisper's behavior more closely ----
        st = [int(x) for x in (self.config.suppress_tokens or [])]

        # OpenAI Whisper uses "-1" as a special meaning for suppression defaults (commonly non-speech tokens).
        # HF sometimes expands this already, but keep compatibility.
        # (Note: OpenAI repo discussions mention the default "-1" behavior explicitly.)
        if -1 in st:
            st = [t for t in st if t >= 0]
            if hasattr(self.tokenizer, "non_speech_tokens"):
                st.extend(int(t) for t in self.tokenizer.non_speech_tokens)

        self.suppress_tokens = set(st)

        # ---- begin_suppress_tokens: suppress-at-start list (Whisper-style) ----
        bst = getattr(self.config, "begin_suppress_tokens", None) or []
        self.begin_suppress_tokens = set(int(x) for x in bst)

    def _discover_io_names_and_cache_metadata(self) -> None:
        # Names from IO
        self.encoder_input_name = self._must_find(self.encoder_io.inputs, ["input_features"])
        self.encoder_cross_cache_names = self._find_cross_cache_names(self.encoder_io.outputs)

        self.decoder_input_ids_name = self._must_find(self.decoder_io.inputs, ["input_ids"])
        self.decoder_attention_mask_name = self._must_find(self.decoder_io.inputs, ["attention_mask"])
        self.decoder_position_ids_name = self._must_find(self.decoder_io.inputs, ["position_ids"])
        self.decoder_logits_name = self._must_find(self.decoder_io.outputs, ["logits"])

        self.decoder_cross_cache_names = self._find_cross_cache_names(self.decoder_io.inputs)
        self.decoder_uses_cross_cache = bool(self.decoder_cross_cache_names)

        # KV cache (self) in/out names
        self.kv_self_in_names = [
            n.name for n in self.decoder_io.inputs
            if "cache_self" in n.name.lower() and n.name.lower().endswith("_in")
        ]
        self.kv_self_out_names = [
            n.name for n in self.decoder_io.outputs
            if "cache_self" in n.name.lower() and n.name.lower().endswith("_out")
        ]
        self.has_kv_cache = bool(self.kv_self_in_names and self.kv_self_out_names)

        # Infer max positions + cache len from shapes
        self.attn_max_len = int(self._get_input_shape_lastdim(self.decoder_attention_mask_name))  # 200
        self.self_cache_len = int(self._get_any_self_cache_len())  # 199

    def _get_encoder_providers(self) -> list[str]:
        if os.getenv("QNN_ENCODER_CPU", "").lower() in {"1", "true", "yes"}:
            self.logger.warning("QNN encoder forced to CPUExecutionProvider for debugging.")
            return ["CPUExecutionProvider"]
        return ["QNNExecutionProvider", "CPUExecutionProvider"]

    def _get_decoder_providers(self) -> list[str]:
        if os.getenv("QNN_DECODER_CPU", "").lower() in {"1", "true", "yes"}:
            self.logger.warning("QNN decoder forced to CPUExecutionProvider for debugging.")
            return ["CPUExecutionProvider"]
        return ["QNNExecutionProvider", "CPUExecutionProvider"]

    def dump_io(self) -> None:
        print("\nEncoder inputs:")
        for node in self.encoder_io.inputs:
            print(f"  - {node.name}: shape={node.shape}, type={node.type}")
        print("Encoder outputs:")
        for node in self.encoder_io.outputs:
            print(f"  - {node.name}: shape={node.shape}, type={node.type}")

        print("\nDecoder inputs:")
        for node in self.decoder_io.inputs:
            print(f"  - {node.name}: shape={node.shape}, type={node.type}")
        print("Decoder outputs:")
        for node in self.decoder_io.outputs:
            print(f"  - {node.name}: shape={node.shape}, type={node.type}")

    def _must_find(self, nodes: Iterable[ort.NodeArg], candidates: list[str]) -> str:
        name = self._find_name(nodes, candidates)
        if name is None:
            raise RuntimeError(f"Unable to find one of {candidates} in {[n.name for n in nodes]}")
        return name

    def _find_name(self, nodes: Iterable[ort.NodeArg], candidates: list[str]) -> str | None:
        for cand in candidates:
            for n in nodes:
                if n.name.lower() == cand.lower():
                    return n.name
        for cand in candidates:
            for n in nodes:
                if cand.lower() in n.name.lower():
                    return n.name
        return None

    def _find_cross_cache_names(self, nodes: Iterable[ort.NodeArg]) -> list[str]:
        names: list[tuple[int, str]] = []
        for node in nodes:
            if "cache_cross" in node.name.lower():
                names.append((self._extract_cache_index(node.name), node.name))
        return [n for _, n in sorted(names, key=lambda x: x[0])]

    def _extract_cache_index(self, name: str) -> int:
        parts = name.split("_")
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return 0

    def _get_input_shape_lastdim(self, input_name: str) -> int:
        node = next(n for n in self.decoder_io.inputs if n.name == input_name)
        if not node.shape or not isinstance(node.shape[-1], int):
            raise RuntimeError(f"Input {input_name} has non-static last dim: {node.shape}")
        return int(node.shape[-1])

    def _get_any_self_cache_len(self) -> int:
        # Infer robustly from a self-cache tensor by matching attn_max_len-1 on any axis.
        expected = max(0, int(self.attn_max_len) - 1)
        for n in self.decoder_io.inputs:
            if "cache_self" in n.name.lower() and n.name.lower().endswith("_in"):
                if not n.shape:
                    continue
                for dim in n.shape:
                    if isinstance(dim, int) and int(dim) == expected:
                        return int(dim)
                for dim in n.shape:
                    if isinstance(dim, int) and int(dim) > 1:
                        return int(dim)
        raise RuntimeError("Unable to infer self cache length from decoder inputs.")

    def _dtype_for_input(self, name: str, fallback=np.int64):
        node = next((n for n in self.decoder_io.inputs if n.name == name), None)
        if node is None or not node.type:
            return fallback
        t = node.type.lower()
        if "uint16" in t:
            return np.uint16
        if "uint8" in t:
            return np.uint8
        if "int32" in t:
            return np.int32
        if "int64" in t:
            return np.int64
        if "float16" in t:
            return np.float16
        if "float" in t:
            return np.float32
        return fallback

    def _numpy_dtype_from_ort(self, ort_type: str) -> np.dtype:
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
        return np.float32

    def _validate_model_files(self, onnx_path: Path) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")
        weights_path = onnx_path.with_suffix(".bin")
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing external weights file: {weights_path}")
