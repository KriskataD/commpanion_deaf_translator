"""Whisper Small Quantized STT using ONNX Runtime + QNN Execution Provider."""
from __future__ import annotations

from dataclasses import dataclass
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


class WhisperSmallQuantizedQNNSTT:
    """Run Whisper Small Quantized (encoder+decoder) with ONNX Runtime QNN."""

    # --------------------------
    # Class init: validation, sessions, IO discovery, model config/tokenizer init
    # --------------------------
    def __init__(
        self,
        encoder_dir: str | Path,
        decoder_dir: str | Path,
        prefer_qnn: bool = True,
        debug: bool = False,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.encoder_dir = Path(encoder_dir)
        self.decoder_dir = Path(decoder_dir)
        self.prefer_qnn = prefer_qnn
        self.debug = debug
        self.debug_kv = debug

        self._resolve_model_paths_and_validate()
        self._create_sessions()
        self._init_io_info()
        self._init_feature_extractor_and_tokenizer()
        self._discover_io_names_and_cache_metadata()

        if self.debug:
            self.logger.info("Providers encoder: %s", self.encoder_session.get_providers())
            self.logger.info("Providers decoder: %s", self.decoder_session.get_providers())
            self.logger.info(
                "attn_max_len=%d self_cache_len=%d has_kv_cache=%s",
                self.attn_max_len, self.self_cache_len, self.has_kv_cache
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
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
        self.config = WhisperConfig.from_pretrained("openai/whisper-small")
        self.suppress_tokens = set(self.config.suppress_tokens or [])

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
    # Public methods (dump_io, transcribe wrappers)
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

    def predict(self, *args: Any, **kwargs: Any) -> str:
        return self.transcribe(*args, **kwargs)

    def transcribe(
        self,
        audio: np.ndarray | str | Path,
        audio_sample_rate: int | None = None,
        language: str | None = None,
    ) -> str:
        if self.debug:
            self._debug_logits_fallback_warned = False
        tokens = self.transcribe_tokens(audio, audio_sample_rate=audio_sample_rate, language=language)
        return self._final_decode_and_log(tokens)

    def transcribe_wav(self, wav_path: Path, language: str | None = None) -> str:
        return self.transcribe(Path(wav_path), language=language)

    def transcribe_tokens(
        self,
        audio: np.ndarray | str | Path,
        audio_sample_rate: int | None = None,
        language: str | None = None,
    ) -> list[int]:
        if isinstance(audio, (str, Path)):
            wav_path = Path(audio)
            self.logger.info("Starting QNN transcription: %s (language=%s)", wav_path, language or "auto")
            source_audio = self._load_and_log_audio(wav_path)
            source_rate = 16000
        else:
            if audio_sample_rate is None:
                raise ValueError("audio_sample_rate must be provided when audio is a numpy array.")
            source_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            source_rate = int(audio_sample_rate)
            self.logger.info(
                "Starting QNN transcription from numpy audio: samples=%d sample_rate=%d (language=%s)",
                source_audio.shape[0],
                source_rate,
                language or "auto",
            )

        chunks = self._chunk_and_resample_audio(source_audio, source_rate)
        all_tokens: list[int] = []
        for idx, chunk in enumerate(chunks):
            self.logger.info("Transcribing chunk %d/%d (samples=%d)", idx + 1, len(chunks), chunk.shape[0])
            chunk_tokens = self._transcribe_single_chunk(chunk, language=language)
            all_tokens.extend(chunk_tokens)
        return all_tokens

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

    def _transcribe_single_chunk(self, audio: np.ndarray, language: str | None = None) -> list[int]:
        features = self._extract_and_pack_features(audio)
        enc_cross_cache = self._run_encoder(features)

        prompt_ids = self._build_prompt_ids(language)
        if self.debug:
            self.logger.info("Prompt ids: %s", prompt_ids)
            self.logger.info("Prompt tokens: %s", self.tokenizer.decode(prompt_ids, skip_special_tokens=False))

        if not prompt_ids:
            raise RuntimeError("Prompt ids empty.")

        return self._decoder_decode_single_loop(prompt_ids, enc_cross_cache)

    def _chunk_and_resample_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model_sample_rate: int = 16000,
        model_chunk_seconds: int = 30,
    ) -> list[np.ndarray]:
        if sample_rate != model_sample_rate:
            audio = self._resample_audio(audio, sample_rate, model_sample_rate)

        chunk_samples = max(1, int(model_sample_rate * model_chunk_seconds))
        if audio.shape[0] <= chunk_samples:
            return [audio]

        chunks: list[np.ndarray] = []
        for start in range(0, audio.shape[0], chunk_samples):
            chunks.append(audio[start:start + chunk_samples])
        return chunks

    def _run_encoder(self, features: np.ndarray) -> dict[str, np.ndarray]:
        # ----- encoder -> cross-cache outputs -----
        enc_inputs = {self.encoder_input_name: features}
        t0 = time.perf_counter()
        enc_out = self.encoder_session.run(self.encoder_cross_cache_names, enc_inputs)
        self.logger.info("Encoder run returned in %.3fs", time.perf_counter() - t0)
        return {n: v for n, v in zip(self.encoder_cross_cache_names, enc_out)}

    def _decoder_decode_single_loop(
        self,
        prompt_ids: list[int],
        enc_cross_cache: dict[str, np.ndarray],
    ) -> list[int]:
        kv_cache = self._initialize_kv_cache() if self.has_kv_cache else {}
        logits: np.ndarray | None = None
        pos = 0
        prev_cache_out: dict[str, np.ndarray] | None = None
        input_ids: list[int] = prompt_ids.copy()

        eot_token = int(getattr(self.tokenizer, "eos_token_id", -1))
        if eot_token < 0:
            raise RuntimeError("Tokenizer eos_token_id missing.")

        prompt_len = len(prompt_ids)
        generation_started = False
        max_new_tokens = 0
        gen_step = 0

        while pos < self.attn_max_len:
            if pos < prompt_len:
                tok = int(prompt_ids[pos])

                if self.debug_kv and prev_cache_out is not None:
                    wiring_ok = self._cache_dicts_equal(kv_cache, prev_cache_out)
                    self.logger.info("PREFILL wiring: cache_in(t+1)==cache_out(t): %s", wiring_ok)

                cache_in = kv_cache
                logits, kv_cache = self._decoder_step(
                    token_id=tok,
                    pos=pos,
                    kv_cache=kv_cache,
                    enc_cross_cache=enc_cross_cache,
                )

                if self.debug_kv:
                    layers_match, common_top_idx, global_max = self._cache_delta_summary(cache_in, kv_cache, pos)
                    tok_str = self.tokenizer.decode([tok], skip_special_tokens=False)
                    self.logger.info(
                        "PREFILL t=%d token=%d/%r pos=%d cache_delta: layers_match_pos=%d/%d common_top_idx=%d global_max=%s",
                        pos,
                        tok,
                        tok_str,
                        pos,
                        layers_match,
                        len(cache_in),
                        common_top_idx,
                        global_max,
                    )

                prev_cache_out = kv_cache
                pos += 1
                continue

            if not generation_started:
                remaining_positions = max(0, (self.attn_max_len - pos))
                max_new_tokens = min(200, remaining_positions)
                self.logger.info(
                    "Decoder prefill done. pos=%d attn_max_len=%d -> max_new_tokens=%d",
                    pos, self.attn_max_len, max_new_tokens
                )
                generation_started = True

            if gen_step >= max_new_tokens:
                break

            if self.debug_kv and gen_step % 10 == 0:
                self.logger.info("Decoder gen step %d/%d", gen_step + 1, max_new_tokens)

            if logits is None:
                # Safety: compute logits for "next token" from last known token at previous position
                last_tok = input_ids[-1]
                logits, kv_cache = self._decoder_step(
                    token_id=int(last_tok),
                    pos=max(0, pos - 1),
                    kv_cache=kv_cache,
                    enc_cross_cache=enc_cross_cache,
                )

            next_token = int(self._select_next_token_from_logits(logits))
            if self.debug_kv and gen_step == 0:
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

            logits, kv_cache = self._decoder_step(
                token_id=next_token,
                pos=pos,  # <-- correct position for this token
                kv_cache=kv_cache,
                enc_cross_cache=enc_cross_cache,
            )

            if self.debug_kv and gen_step == 0:
                top5_ids, top5_scores = self._topk_from_logits(logits, k=5)
                top5_toks = [self.tokenizer.decode([int(tid)], skip_special_tokens=False) for tid in top5_ids]
                self.logger.info(
                    "GEN-STEP1 logits: top5_ids=%s top5_toks=%s top5_scores=%s",
                    top5_ids,
                    top5_toks,
                    [float(s) for s in top5_scores],
                )

            pos += 1
            gen_step += 1

        return input_ids

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
        ).input_features  # float32, shape (1, 80, 3000)
        return feats.astype(np.float32)

    def _prepare_encoder_features(self, features: np.ndarray) -> np.ndarray:
        # Encoder expects uint16, and this is the only input we pack from float16 bits.
        node = self.encoder_io.inputs[0]
        t = (node.type or "").lower()

        if "uint16" in t:
            # ✅ pack float16 bits into uint16 (same idea as your logits handling)
            packed = features.astype(np.float16).view(np.uint16)
            return packed

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

        # attention_mask: [1,1,1,200] uint16 (plain values, not packed-fp16)
        #attn = np.zeros((1, 1, 1, self.attn_max_len), dtype=np.uint16)
        count = min(pos + 1, self.attn_max_len)

        attn_f16 = np.zeros((1, 1, 1, self.attn_max_len), dtype=np.float16)

        # Right-align active tokens so position_ids/kv cache line up with attention.
        attn_f16[0, 0, 0, -count:] = np.float16(1.0)

        decoder_inputs[self.decoder_attention_mask_name] = attn_f16.view(np.uint16)

        m = decoder_inputs[self.decoder_attention_mask_name]
        self.logger.info("attn_mask uint16 min=%d max=%d", int(m.min()), int(m.max()))


        # position_ids: [1] int32 (NOT [1,1])
        pid_dtype = self._dtype_for_input(self.decoder_position_ids_name, fallback=np.int32)
        pid = self.self_cache_len - count   # 199,198,197,... for pos=0,1,2...
        if pid < 0:
            pid = 0
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

    def _scores_with_suppression(self, logits: np.ndarray) -> np.ndarray:
        scores = self._logits_to_scores(logits)

        #eot = int(getattr(self.tokenizer, "eos_token_id", -1))
        #if getattr(self, "_block_eot_steps", 0) > 0 and 0 <= eot < scores.shape[0]:
        #    scores[eot] = -1e9
        #    self._block_eot_steps -= 1

        # Suppress Whisper control/special tokens
        if getattr(self, "suppress_tokens", None):
            for tid in self.suppress_tokens:
                if 0 <= tid < scores.shape[0]:
                    scores[tid] = -1e9

        return scores

    def _select_next_token_from_logits(self, logits: np.ndarray) -> int:
        scores = self._scores_with_suppression(logits)
        return int(np.argmax(scores))

    def _topk_from_logits(self, logits: np.ndarray, k: int = 5) -> tuple[list[int], list[float]]:
        scores = self._scores_with_suppression(logits)
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
            scores = x.view(np.float16).astype(np.float32)
            finite_ratio = float(np.isfinite(scores).mean()) if scores.size else 1.0

            if finite_ratio < 0.99:
                if self.debug and not getattr(self, "_debug_logits_fallback_warned", False):
                    self._debug_logits_fallback_warned = True
                    self.logger.warning(
                        "Decoded fp16 logits have low finite ratio (%.2f%%); falling back to uint16->float32 interpretation.",
                        finite_ratio * 100.0,
                    )
                return x.astype(np.float32)

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
        # use first self cache input to infer length (199)
        for n in self.decoder_io.inputs:
            if "cache_self" in n.name.lower() and n.name.lower().endswith("_in"):
                if n.shape and isinstance(n.shape[-1], int):
                    return int(n.shape[-1])
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
