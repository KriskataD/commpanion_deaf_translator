"""Whisper QNN STT using ONNX Runtime + QNN Execution Provider."""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import logging
import os
from pathlib import Path
from typing import Any, Iterable
import wave
import time
from collections import Counter

import numpy as np
import onnxruntime as ort
import torch
from transformers import WhisperTokenizer, WhisperConfig, WhisperFeatureExtractor

from .ort_qnn import make_session


# --------------------------
# Module-level helpers (dump_model_io, dataclasses)
# --------------------------

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


class WhisperQnnSTT:
    """Run Whisper QNN STT (small-quantized or large-v3-turbo) with ONNX Runtime QNN."""

    # --------------------------
    # Class init: validation, sessions, IO discovery, model config/tokenizer init
    # --------------------------
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
        if self.profile.name == "large-v3-turbo":
            self.feature_extractor = WhisperFeatureExtractor(
                feature_size=self.profile.n_mels,
                sampling_rate=16000,
            )
        else:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.profile.hf_id)

        self.tokenizer = WhisperTokenizer.from_pretrained(self.profile.hf_id)
        self.config = WhisperConfig.from_pretrained(self.profile.hf_id)
        self.suppress_tokens = set(self.config.suppress_tokens or [])

        # Anti-repetition guards for greedy decoding.
        self.no_repeat_ngram_size = 3
        self.loop_window = 48
        self.loop_tail = 16
        self.loop_diversity_threshold = 0.18
        self.loop_hits_to_stop = 2
        self.enable_repeat_guards = True
        self.enable_compression_guard = True
        self.compression_ratio_threshold = 2.2
        self.compression_check_interval = 10
        self.min_loop_check_tokens = 12
        self._loop_hit_count = 0

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

    # --------------------------
    # Providers
    # --------------------------
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

    # --------------------------
    # Public methods (dump_io, transcribe_wav)
    # --------------------------
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

    def transcribe_wav(self, wav_path: Path, language: str | None = None) -> str:
        self.logger.info("Starting QNN transcription: %s (language=%s)", wav_path, language or "auto")

        if self.debug:
            self._debug_logits_fallback_warned = False

        audio = self._load_and_log_audio(Path(wav_path))
        features = self._extract_and_pack_features(audio)
        enc_cross_cache = self._run_encoder(features)

        prompt_ids = self._build_prompt_ids(language)
        if self.debug:
            self.logger.info("Prompt ids: %s", prompt_ids)
            self.logger.info("Prompt tokens: %s", self.tokenizer.decode(prompt_ids, skip_special_tokens=False))

        if not prompt_ids:
            raise RuntimeError("Prompt ids empty.")

        logits, kv_cache, pos, input_ids = self._decoder_prefill(prompt_ids, enc_cross_cache)
        input_ids = self._decoder_generate(logits, kv_cache, pos, input_ids, enc_cross_cache)

        return self._final_decode_and_log(input_ids)

    # --------------------------
    # Transcribe pipeline helpers (audio load, mel, feature pack, encoder run, decoder prefill, decode loop)
    # --------------------------
    def _load_and_log_audio(self, wav_path: Path) -> np.ndarray:
        audio = self._load_wav_mono_16k(wav_path)
        self.logger.info("WAV loaded. Samples=%d", audio.shape[0])

        # After loading audio as float32 in [-1, 1]
        rms = float(np.sqrt(np.mean(audio**2)) + 1e-12)
        peak = float(np.max(np.abs(audio)) + 1e-12)
        self.logger.info(
            "Audio stats: duration=%.2fs rms=%.6f peak=%.6f",
            len(audio) / 16000.0, rms, peak
        )

        return audio

    def _extract_and_pack_features(self, audio: np.ndarray) -> np.ndarray:
        mel = self._log_mel_spectrogram(audio)               # [1,80,3000] float32
        features = self._prepare_encoder_features(mel)       # uint16 expected
        if self.debug:
            self.logger.info("Encoder features min=%s max=%s", int(features.min()), int(features.max()))
        self.logger.info("Encoder features ready. Shape=%s, dtype=%s", features.shape, features.dtype)
        return features

    def _run_encoder(self, features: np.ndarray) -> dict[str, np.ndarray]:
        # ----- encoder -> cross-cache outputs -----
        enc_inputs = {self.encoder_input_name: features}
        t0 = time.perf_counter()
        enc_out = self.encoder_session.run(self.encoder_cross_cache_names, enc_inputs)
        self.logger.info("Encoder run returned in %.3fs", time.perf_counter() - t0)
        return {n: v for n, v in zip(self.encoder_cross_cache_names, enc_out)}

    def _decoder_prefill(
        self,
        prompt_ids: list[int],
        enc_cross_cache: dict[str, np.ndarray],
    ) -> tuple[np.ndarray | None, dict[str, np.ndarray], int, list[int]]:
        # KV cache init
        kv_cache = self._initialize_kv_cache() if self.has_kv_cache else {}

        # Prefill each prompt token sequentially (because input_ids is [1,1])
        # This warms the self-cache and produces logits for the next token.
        logits = None
        pos = 0
        prev_cache_out: dict[str, np.ndarray] | None = None
        for t, tok in enumerate(prompt_ids):
            if self.debug_kv and prev_cache_out is not None:
                wiring_ok = self._cache_dicts_equal(kv_cache, prev_cache_out)
                self.logger.info("PREFILL wiring: cache_in(t+1)==cache_out(t): %s", wiring_ok)

            cache_in = kv_cache
            logits, kv_cache = self._decoder_step(
                token_id=int(tok),
                pos=pos,
                kv_cache=kv_cache,
                enc_cross_cache=enc_cross_cache,
            )

            if self.debug_kv:
                layers_match, common_top_idx, global_max = self._cache_delta_summary(cache_in, kv_cache, pos)
                tok_str = self.tokenizer.decode([int(tok)], skip_special_tokens=False)
                self.logger.info(
                    "PREFILL t=%d token=%d/%r pos=%d cache_delta: layers_match_pos=%d/%d common_top_idx=%d global_max=%s",
                    t,
                    int(tok),
                    tok_str,
                    pos,
                    layers_match,
                    len(cache_in),
                    common_top_idx,
                    global_max,
                )

            prev_cache_out = kv_cache
            pos += 1
            if pos >= self.self_cache_len:
                break

        # Now generate new tokens
        input_ids: list[int] = prompt_ids.copy()
        return logits, kv_cache, pos, input_ids

    def _decoder_generate(
        self,
        logits: np.ndarray | None,
        kv_cache: dict[str, np.ndarray],
        pos: int,
        input_ids: list[int],
        enc_cross_cache: dict[str, np.ndarray],
    ) -> list[int]:
        eot_token = int(getattr(self.tokenizer, "eos_token_id", -1))
        if eot_token < 0:
            raise RuntimeError("Tokenizer eos_token_id missing.")
        self._loop_hit_count = 0
        prompt_len = len(input_ids)

        # pos currently == len(prompt_ids)  (next position index to be generated)
        remaining_positions = max(0, (self.self_cache_len - pos))
        max_new_tokens = min(200, remaining_positions)

        self.logger.info(
            "Decoder prefill done. pos=%d self_cache_len=%d attn_max_len=%d -> max_new_tokens=%d",
            pos, self.self_cache_len, self.attn_max_len, max_new_tokens
        )

        # self._block_eot_steps = 8   # Fix 4: block EOS/EOT for first N generation selections

        # We already have logits from the last prefill step (unless prompt was empty)
        for step in range(max_new_tokens):
            banned_count = 0
            if self.debug_kv and step % 10 == 0:
                self.logger.info("Decoder gen step %d/%d", step + 1, max_new_tokens)

            if logits is None:
                # Safety: compute logits for "next token" from last known token at previous position
                last_tok = input_ids[-1]
                logits, kv_cache = self._decoder_step(
                    token_id=int(last_tok),
                    pos=max(0, pos - 1),
                    kv_cache=kv_cache,
                    enc_cross_cache=enc_cross_cache,
                )

            scores = self._logits_to_scores(logits)

            # Suppress Whisper control/special tokens, but never suppress EOS/EOT.
            if getattr(self, "suppress_tokens", None):
                for tid in self.suppress_tokens:
                    if tid == eot_token:
                        continue
                    if 0 <= tid < scores.shape[0]:
                        scores[tid] = -1e9

            if self.enable_repeat_guards:
                generated_ids = input_ids[prompt_len:]
                banned_count = self._apply_no_repeat_ngrams(
                    scores=scores,
                    generated_ids=generated_ids,
                    n=self.no_repeat_ngram_size,
                    eos_id=eot_token,
                )

            next_token = int(np.argmax(scores))
            if self.debug_kv and step == 0:
                top5_ids, top5_scores = self._topk_from_logits(logits, k=5)
                top5_toks = [self.tokenizer.decode([int(tid)], skip_special_tokens=False) for tid in top5_ids]
                chosen_tok = self.tokenizer.decode([next_token], skip_special_tokens=False)
                self.logger.info(
                    "GEN-START from last-prompt logits: top5_ids=%s top5_toks=%s chosen_id=%d chosen_tok=%r",
                    top5_ids,
                    top5_toks,
                    next_token,
                    chosen_tok,
                )
                self.logger.info(
                    "GEN-STEP0 inputs: pos=%d input_id=%d/%r (from last prompt logits)",
                    pos,
                    next_token,
                    chosen_tok,
                )

            # If model ends immediately, stop.
            if next_token == eot_token:
                input_ids.append(next_token)
                break

            # IMPORTANT: next_token belongs to CURRENT pos
            input_ids.append(next_token)

            if self.enable_repeat_guards:
                generated_ids = input_ids[prompt_len:]
                loop_detected, diversity, tail_unique = self._looks_like_loop(
                    generated_ids=generated_ids,
                    window=self.loop_window,
                    tail=self.loop_tail,
                    diversity_thr=self.loop_diversity_threshold,
                )
                compression_suspect = False
                if self.enable_compression_guard and (loop_detected or (step + 1) % self.compression_check_interval == 0):
                    tail_text = self.tokenizer.decode(generated_ids[-30:], skip_special_tokens=True)[-200:]
                    compression_ratio = self._compression_ratio(tail_text)
                    compression_suspect = compression_ratio > self.compression_ratio_threshold
                else:
                    tail_text = self.tokenizer.decode(generated_ids[-30:], skip_special_tokens=True)[-200:]
                    compression_ratio = 0.0

                self.logger.info(
                    "RepeatGuard step=%d diversity=%.3f banned_count=%d tail='%s'",
                    step,
                    diversity,
                    banned_count,
                    tail_text.replace("\n", " ")[:120],
                )

                if loop_detected or compression_suspect:
                    self._loop_hit_count += 1
                else:
                    self._loop_hit_count = 0

                if self._loop_hit_count >= self.loop_hits_to_stop:
                    warning_msg = (
                        "⚠️ REPETITION LOOP DETECTED: stopping decode early "
                        f"step={step} diversity={diversity:.3f} tail_unique={tail_unique} "
                        f"banned_count={banned_count} compression_ratio={compression_ratio:.2f}"
                    )
                    self.logger.warning(warning_msg)
                    self.logger.warning("RepeatGuard tail snippet: %s", tail_text.replace("\n", " ")[:200])
                    print(f"⚠️ [RepeatGuard] Loop detected at step={step}, forcing EOT.")
                    print(f"⚠️ [RepeatGuard] Tail snippet: {tail_text.replace(chr(10), ' ')[:200]}")
                    if input_ids[-1] != eot_token:
                        input_ids.append(eot_token)
                    break

            logits, kv_cache = self._decoder_step(
                token_id=next_token,
                pos=pos,  # <-- correct position for this token
                kv_cache=kv_cache,
                enc_cross_cache=enc_cross_cache,
            )

            if self.debug_kv and step == 0:
                top5_ids, top5_scores = self._topk_from_logits(logits, k=5)
                top5_toks = [self.tokenizer.decode([int(tid)], skip_special_tokens=False) for tid in top5_ids]
                self.logger.info(
                    "GEN-STEP1 logits: top5_ids=%s top5_toks=%s top5_scores=%s",
                    top5_ids,
                    top5_toks,
                    [float(s) for s in top5_scores],
                )

            pos += 1
            if pos >= self.self_cache_len:
                break

        return input_ids

    def _apply_no_repeat_ngrams(
        self,
        scores: np.ndarray,
        generated_ids: list[int],
        n: int,
        eos_id: int,
    ) -> int:
        if n <= 1 or len(generated_ids) < n - 1:
            return 0

        prefix = tuple(generated_ids[-(n - 1):])
        banned: set[int] = set()
        for i in range(0, len(generated_ids) - n + 1):
            if tuple(generated_ids[i:i + n - 1]) == prefix:
                next_idx = i + n - 1
                if 0 <= next_idx < len(generated_ids):
                    banned.add(int(generated_ids[next_idx]))

        if eos_id in banned:
            banned.remove(eos_id)

        for token_id in banned:
            if 0 <= token_id < scores.shape[0]:
                scores[token_id] = -1e9

        return len(banned)

    def _looks_like_loop(
        self,
        generated_ids: list[int],
        window: int,
        tail: int,
        diversity_thr: float,
    ) -> tuple[bool, float, int]:
        min_tokens = max(self.min_loop_check_tokens, max(1, tail) * 2)
        if len(generated_ids) < min_tokens:
            return False, 1.0, len(set(generated_ids[-tail:])) if generated_ids else 0

        recent = generated_ids[-min(window, len(generated_ids)):]
        tail_ids = recent[-tail:] if tail > 0 else recent
        diversity = len(set(recent)) / max(1, len(recent))
        tail_unique = len(set(tail_ids))

        cond_tail_stuck = tail_unique <= 2
        cond_low_diversity = diversity < diversity_thr
        bigrams = list(zip(recent, recent[1:]))
        cond_cycle = len(set(bigrams)) < max(6, len(bigrams) // 6)
        cond_tail_repeat = self._has_repeating_tail_cycle(tail_ids)

        return (cond_tail_stuck or cond_low_diversity or cond_cycle or cond_tail_repeat), diversity, tail_unique

    def _has_repeating_tail_cycle(self, tail_ids: list[int]) -> bool:
        if len(tail_ids) < 6:
            return False

        max_cycle = min(8, len(tail_ids) // 3)
        for cycle_len in range(1, max_cycle + 1):
            pattern = tail_ids[-cycle_len:]
            repeats = 1
            idx = len(tail_ids) - cycle_len
            while idx - cycle_len >= 0 and tail_ids[idx - cycle_len:idx] == pattern:
                repeats += 1
                idx -= cycle_len
                if repeats >= 3:
                    return True
        return False

    def _compression_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        raw = text.encode("utf-8", errors="ignore")
        compressed = gzip.compress(raw)
        return len(raw) / max(1, len(compressed))

    def _final_decode_and_log(self, input_ids: list[int]) -> str:
        decoded_raw = self.tokenizer.decode(input_ids, skip_special_tokens=False).strip()
        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True).strip()

        self.logger.info("DECODE (no-skip)='%s'", decoded_raw[:200])
        self.logger.info("DECODE (skip)='%s'", decoded[:200])
        self.logger.info("Completed QNN transcription (chars=%d).", len(decoded))
        return decoded

    def _validate_model_files(self, onnx_path: Path) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")
        weights_path = onnx_path.with_suffix(".bin")
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing external weights file: {weights_path}")

    def _load_wav_mono_16k(self, wav_path: Path) -> np.ndarray:
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width != 2:
            raise RuntimeError("Only 16-bit PCM WAV files are supported.")

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        if sample_rate != 16000:
            audio = self._resample_audio(audio, sample_rate, 16000)

        return audio

    def _resample_audio(self, audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
        if sample_rate == target_rate:
            return audio
        duration = audio.shape[0] / sample_rate
        target_length = int(duration * target_rate)
        source_indices = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        target_indices = np.linspace(0.0, duration, num=target_length, endpoint=False)
        return np.interp(target_indices, source_indices, audio).astype(np.float32)

    def _log_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        feats = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="np"
        ).input_features
        return feats.astype(np.float32)

    def _prepare_encoder_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preserve small-quantized packing exactly; turbo uses float16 features directly.
        """
        node = self.encoder_io.inputs[0]
        t = (node.type or "").lower()

        if self.profile.name == "large-v3-turbo":
            return features.astype(np.float16)

        if "uint16" in t:
            # From your ONNX inspection:
            scale = np.float32(4.677007018472068e-05)
            zp    = np.uint16(32072)

            x = features.astype(np.float32)
            q = np.rint(x / scale + float(zp)).astype(np.int64)
            q = np.clip(q, 0, 65535).astype(np.uint16)
            return q

        # fallback
        return features.astype(np.float32)

    def _mel_filterbank(self, n_mels: int, n_fft: int, sample_rate: int) -> torch.Tensor:
        def hz_to_mel(freq: float) -> float:
            return 2595.0 * np.log10(1.0 + freq / 700.0)

        def mel_to_hz(mel: float) -> float:
            return 700.0 * (10 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(sample_rate / 2)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_frequencies = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        filter_bank = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(1, n_mels + 1):
            start, center, end = bin_frequencies[i - 1 : i + 2]
            if center > start:
                filter_bank[i - 1, start:center] = (np.arange(start, center) - start) / (center - start)
            if end > center:
                filter_bank[i - 1, center:end] = (end - np.arange(center, end)) / (end - center)

        return torch.from_numpy(filter_bank).float()

    # --------------------------
    # Decoder step + token selection
    # --------------------------
    def _decoder_step(
        self,
        token_id: int,
        pos: int,
        kv_cache: dict[str, np.ndarray],
        enc_cross_cache: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run ONE decoder call. The model expects input_ids [1,1], attention_mask [1,1,1,200], position_ids [1]."""
        decoder_inputs: dict[str, Any] = {}

        input_ids_dtype = self._dtype_for_input(self.decoder_input_ids_name, fallback=np.int32)
        decoder_inputs[self.decoder_input_ids_name] = np.array([[token_id]], dtype=input_ids_dtype)

        count = min(pos + 1, self.self_cache_len)

        if self.profile.name == "large-v3-turbo":
            attn = np.full((1, 1, 1, self.attn_max_len), np.float16(-65504.0), dtype=np.float16)
            attn[0, 0, 0, :count] = np.float16(0.0)
            decoder_inputs[self.decoder_attention_mask_name] = attn
            if self.debug:
                self.logger.info("attn_mask float16 min=%.1f max=%.1f", float(attn.min()), float(attn.max()))
        else:
            # attention_mask: [1,1,1,200] uint16 (plain values, not packed-fp16)
            attn = np.zeros((1, 1, 1, self.attn_max_len), dtype=np.uint16)
            attn[0, 0, 0, -count:] = np.uint16(65535)
            decoder_inputs[self.decoder_attention_mask_name] = attn
            self.logger.info("attn_mask uint16 min=%d max=%d", int(attn.min()), int(attn.max()))

        # position_ids: [1] int32 (NOT [1,1])
        pid_dtype = self._dtype_for_input(self.decoder_position_ids_name, fallback=np.int32)
        if self.profile.name == "large-v3-turbo":
            # IMPORTANT: turbo expects FORWARD positions
            pid = count - 1   # == pos unless clipped
        else:
            # small-quantized expects REVERSE positions
            pid = self.self_cache_len - count
        decoder_inputs[self.decoder_position_ids_name] = np.array([pid], dtype=pid_dtype)


        # KV cache in (self)
        if self.has_kv_cache:
            decoder_inputs.update(kv_cache)

        # Cross cache in
        if self.decoder_uses_cross_cache:
            decoder_inputs.update(enc_cross_cache)

        self.logger.info("pos=%d count=%d pid=%d", pos, count, pid)

        t0 = time.perf_counter()
        outputs = self.decoder_session.run(None, decoder_inputs)

        dt = time.perf_counter() - t0
        if self.debug_kv:
            self.logger.info("Decoder session.run() returned in %.3fs", dt)

        output_map = {node.name: value for node, value in zip(self.decoder_io.outputs, outputs)}
        logits = output_map[self.decoder_logits_name]

        # Update KV cache from *_out
        if self.has_kv_cache:
            new_cache = {name.replace("_out", "_in"): output_map[name] for name in self.kv_self_out_names}
        else:
            new_cache = {}

        return logits, new_cache

    def _select_next_token_from_logits(self, logits: np.ndarray) -> int:
        scores = self._logits_to_scores(logits)
        self.logger.info("scores stats: min=%.4f max=%.4f", float(scores.min()), float(scores.max()))
        
        #eot = int(getattr(self.tokenizer, "eos_token_id", -1))
        #if getattr(self, "_block_eot_steps", 0) > 0 and 0 <= eot < scores.shape[0]:
        #    scores[eot] = -1e9
        #    self._block_eot_steps -= 1

        # Suppress Whisper control/special tokens
        if getattr(self, "suppress_tokens", None):
            for tid in self.suppress_tokens:
                if 0 <= tid < scores.shape[0]:
                    scores[tid] = -1e9

        return int(np.argmax(scores))

    def _topk_from_logits(self, logits: np.ndarray, k: int = 5) -> tuple[list[int], list[float]]:
        scores = self._logits_to_scores(logits)
        top = np.argsort(scores)[-k:][::-1]
        return [int(i) for i in top], [float(scores[int(i)]) for i in top]

    def _infer_seq_axis(self, tensor: np.ndarray, self_cache_len: int) -> int:
        matches = [i for i, d in enumerate(tensor.shape) if int(d) == int(self_cache_len)]
        if not matches:
            raise RuntimeError(f"Unable to infer seq axis for cache tensor shape={tensor.shape}")
        return matches[-1]

    def _cache_delta_summary(
        self,
        cache_in: dict[str, np.ndarray],
        cache_out: dict[str, np.ndarray],
        position: int,
    ) -> tuple[int, int, float]:
        top_idxs: list[int] = []
        global_max = 0.0

        for name, in_tensor in cache_in.items():
            out_tensor = cache_out.get(name)
            if not isinstance(in_tensor, np.ndarray) or not isinstance(out_tensor, np.ndarray):
                continue

            seq_axis = self._infer_seq_axis(in_tensor, self.self_cache_len)
            if in_tensor.dtype.kind in {"u", "i"} or out_tensor.dtype.kind in {"u", "i"}:
                delta = np.abs(out_tensor.astype(np.int32) - in_tensor.astype(np.int32))
            else:
                delta = np.abs(out_tensor.astype(np.float32) - in_tensor.astype(np.float32))

            reduce_axes = tuple(ax for ax in range(delta.ndim) if ax != seq_axis)
            delta_per_pos = delta if not reduce_axes else np.max(delta, axis=reduce_axes)
            delta_per_pos = np.asarray(delta_per_pos).reshape(-1)
            top_idx = int(np.argmax(delta_per_pos))
            top_idxs.append(top_idx)
            global_max = max(global_max, float(np.max(delta_per_pos)))

        if not top_idxs:
            return 0, -1, global_max

        common_top_idx = Counter(top_idxs).most_common(1)[0][0]
        layers_match_pos = int(sum(1 for idx in top_idxs if idx == int(position)))
        return layers_match_pos, int(common_top_idx), global_max

    def _cache_dicts_equal(self, lhs: dict[str, np.ndarray], rhs: dict[str, np.ndarray]) -> bool:
        if set(lhs.keys()) != set(rhs.keys()):
            return False
        for k in lhs:
            if not np.array_equal(lhs[k], rhs[k]):
                return False
        return True

    def _logits_to_scores(self, logits: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(np.squeeze(logits)).reshape(-1)

        if x.dtype == np.uint16:
            # From your decoder graph inspection:
            scale = np.float32(0.0012925398768857121)
            zp    = np.float32(17867.0)

            scores = (x.astype(np.float32) - zp) * scale
            return scores

        return x.astype(np.float32)

    # --------------------------
    # IO discovery helpers
    # --------------------------
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

    # --------------------------
    # Prompt + KV cache init
    # --------------------------
    def _build_prompt_ids(self, language: str | None) -> list[int]:
        lang = (language or "en").lower()

        # Pull SOT directly from WhisperTokenizer rather than relying on a numeric id.
        start = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if start is None or int(start) < 0:
            raise RuntimeError("Cannot resolve WhisperTokenizer SOT token id.")

        # Get language/task prompt ids from tokenizer (e.g. <|en|><|transcribe|><|notimestamps|>)
        rest: list[int] = []
        if hasattr(self.tokenizer, "get_decoder_prompt_ids"):
            items = self.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
            rest = [int(tid) for _, tid in items]

        return [int(start)] + rest

    def _initialize_kv_cache(self) -> dict[str, np.ndarray]:
        cache: dict[str, np.ndarray] = {}
        for node in self.decoder_io.inputs:
            if node.name in self.kv_self_in_names:
                shape = tuple(int(d) for d in node.shape)  # fully static
                #cache[node.name] = np.zeros(shape, dtype=self._numpy_dtype_from_ort(node.type))
                dtype = self._numpy_dtype_from_ort(node.type)
                if dtype == np.uint8:
                    cache[node.name] = np.full(shape, 128, dtype=np.uint8)  # ✅ common zero-point
                else:
                    cache[node.name] = np.zeros(shape, dtype=dtype)
        return cache

    # --------------------------
    # Dtype + numpy dtype helpers
    # --------------------------
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


class WhisperSmallQuantizedQNNSTT(WhisperQnnSTT):
    """Backward-compatible wrapper for the small quantized Whisper profile."""

    def __init__(
        self,
        encoder_dir: str | Path,
        decoder_dir: str | Path,
        prefer_qnn: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(
            encoder_dir=encoder_dir,
            decoder_dir=decoder_dir,
            stt_model="small-quantized",
            prefer_qnn=prefer_qnn,
            debug=debug,
        )


class WhisperLargeV3TurboQNNSTT(WhisperQnnSTT):
    """Convenience wrapper for the large-v3-turbo Whisper profile."""

    def __init__(
        self,
        encoder_dir: str | Path,
        decoder_dir: str | Path,
        prefer_qnn: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(
            encoder_dir=encoder_dir,
            decoder_dir=decoder_dir,
            stt_model="large-v3-turbo",
            prefer_qnn=prefer_qnn,
            debug=debug,
        )
