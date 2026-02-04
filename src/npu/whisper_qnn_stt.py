"""Whisper Small Quantized STT using ONNX Runtime + QNN Execution Provider."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Iterable
import wave
import time

import numpy as np
import onnxruntime as ort
import torch
from transformers import WhisperTokenizer, WhisperConfig

from .ort_qnn import make_session


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

        self.encoder_onnx = self.encoder_dir / "model.onnx"
        self.decoder_onnx = self.decoder_dir / "model.onnx"

        self.logger.info("QNN encoder model path: %s", self.encoder_onnx)
        self.logger.info("QNN decoder model path: %s", self.decoder_onnx)

        self._validate_model_files(self.encoder_onnx)
        self._validate_model_files(self.decoder_onnx)

        try:
            self.encoder_session = make_session(self.encoder_onnx, providers=self._get_encoder_providers())
            self.decoder_session = make_session(self.decoder_onnx, providers=self._get_decoder_providers())
        except Exception:
            self.logger.exception("Failed to create ONNX Runtime sessions.")
            raise

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
        self.kv_self_in_names = [n.name for n in self.decoder_io.inputs if "cache_self" in n.name.lower() and n.name.lower().endswith("_in")]
        self.kv_self_out_names = [n.name for n in self.decoder_io.outputs if "cache_self" in n.name.lower() and n.name.lower().endswith("_out")]
        self.has_kv_cache = bool(self.kv_self_in_names and self.kv_self_out_names)

        # Infer max positions + cache len from shapes
        self.attn_max_len = int(self._get_input_shape_lastdim(self.decoder_attention_mask_name))  # 200
        self.self_cache_len = int(self._get_any_self_cache_len())  # 199

        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
        self.config = WhisperConfig.from_pretrained("openai/whisper-small")
        self.suppress_tokens = set(self.config.suppress_tokens or [])

        if self.debug:
            self.logger.info("Providers encoder: %s", self.encoder_session.get_providers())
            self.logger.info("Providers decoder: %s", self.decoder_session.get_providers())
            self.logger.info("attn_max_len=%d self_cache_len=%d has_kv_cache=%s",
                             self.attn_max_len, self.self_cache_len, self.has_kv_cache)

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
    # Public debug
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

    # --------------------------
    # Transcription
    # --------------------------
    def transcribe_wav(self, wav_path: Path, language: str | None = None) -> str:
        self.logger.info("Starting QNN transcription: %s (language=%s)", wav_path, language or "auto")

        if self.debug:
            self._debug_top5_printed = False

        audio = self._load_wav_mono_16k(Path(wav_path))
        self.logger.info("WAV loaded. Samples=%d", audio.shape[0])

        mel = self._log_mel_spectrogram(audio)               # [1,80,3000] float32
        features = self._prepare_encoder_features(mel)       # uint16 expected
        self.logger.info("Encoder features ready. Shape=%s, dtype=%s", features.shape, features.dtype)

        # ----- encoder -> cross-cache outputs -----
        enc_inputs = {self.encoder_input_name: features}
        t0 = time.perf_counter()
        enc_out = self.encoder_session.run(self.encoder_cross_cache_names, enc_inputs)
        self.logger.info("Encoder run returned in %.3fs", time.perf_counter() - t0)
        enc_cross_cache = {n: v for n, v in zip(self.encoder_cross_cache_names, enc_out)}

        # ----- decoder prompt prefill (IMPORTANT) -----
        prompt_ids = self._build_prompt_ids(language)
        if self.debug:
            self.logger.info("Prompt ids: %s", prompt_ids)
            self.logger.info("Prompt tokens: %s", self.tokenizer.decode(prompt_ids, skip_special_tokens=False))

        if not prompt_ids:
            raise RuntimeError("Prompt ids empty.")

        # KV cache init
        kv_cache = self._initialize_kv_cache() if self.has_kv_cache else {}

        # Prefill each prompt token sequentially (because input_ids is [1,1])
        # This warms the self-cache and produces logits for the next token.
        logits = None
        pos = 0
        for tok in prompt_ids:
            logits, kv_cache = self._decoder_step(
                token_id=int(tok),
                pos=pos,
                kv_cache=kv_cache,
                enc_cross_cache=enc_cross_cache,
            )
            pos += 1
            if pos >= self.attn_max_len:
                break

        # Now generate new tokens
        input_ids: list[int] = prompt_ids.copy()

        eot_token = int(getattr(self.tokenizer, "eos_token_id", -1))
        if eot_token < 0:
            raise RuntimeError("Tokenizer eos_token_id missing.")

        # pos currently == len(prompt_ids)  (next position index to be generated)
        remaining_positions = max(0, (self.attn_max_len - pos))
        max_new_tokens = min(200, remaining_positions)

        self.logger.info(
            "Decoder prefill done. pos=%d attn_max_len=%d -> max_new_tokens=%d",
            pos, self.attn_max_len, max_new_tokens
        )

        # We already have logits from the last prefill step (unless prompt was empty)
        for step in range(max_new_tokens):
            if step % 10 == 0:
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

            next_token = int(self._select_next_token_from_logits(logits))

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

            pos += 1
            if pos >= self.attn_max_len:
                break


        

        decoded_raw = self.tokenizer.decode(input_ids, skip_special_tokens=False).strip()
        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True).strip()

        self.logger.info("DECODE (no-skip)='%s'", decoded_raw[:200])
        self.logger.info("DECODE (skip)='%s'", decoded[:200])
        self.logger.info("Completed QNN transcription (chars=%d).", len(decoded))
        return decoded

    # --------------------------
    # Decoder step helpers
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

        decoder_inputs[self.decoder_input_ids_name] = np.array([[token_id]], dtype=np.int32)

        # attention_mask: [1,1,1,200] uint16
        # Set allowed positions up to (pos) inclusive to 1, rest 0.
        am_dtype = self._dtype_for_input(self.decoder_attention_mask_name, fallback=np.uint16)
        attn = np.zeros((1, 1, 1, self.attn_max_len), dtype=am_dtype)
        upto = min(pos + 1, self.attn_max_len)
        attn[0, 0, 0, :upto] = 1
        decoder_inputs[self.decoder_attention_mask_name] = attn

        # position_ids: [1] int32 (NOT [1,1])
        pid_dtype = self._dtype_for_input(self.decoder_position_ids_name, fallback=np.int32)
        decoder_inputs[self.decoder_position_ids_name] = np.array([pos], dtype=pid_dtype)

        # KV cache in (self)
        if self.has_kv_cache:
            decoder_inputs.update(kv_cache)

        # Cross cache in
        if self.decoder_uses_cross_cache:
            decoder_inputs.update(enc_cross_cache)

        if self.debug:
            self._log_decoder_inputs(decoder_inputs)

        t0 = time.perf_counter()
        outputs = self.decoder_session.run(None, decoder_inputs)

        # --- One-time: dump runtime outputs (names, shape, dtype, min/max) ---
        if self.debug and not getattr(self, "_debug_decoder_outputs_printed", False):
            self._debug_decoder_outputs_printed = True
            self.logger.info("=== Decoder runtime outputs (session.run) ===")
            self.logger.info("Returned %d tensors", len(outputs))

            for node, value in zip(self.decoder_io.outputs, outputs):
                if isinstance(value, np.ndarray):
                    # min/max are useful but can be expensive; do it once
                    try:
                        vmin = float(value.min()) if value.size else None
                        vmax = float(value.max()) if value.size else None
                    except Exception:
                        vmin = vmax = None
                    self.logger.info(
                        "OUT %-35s shape=%s dtype=%s min=%s max=%s",
                        node.name, value.shape, value.dtype, vmin, vmax
                    )
                else:
                    self.logger.info("OUT %-35s (non-ndarray) type=%s", node.name, type(value))

            # Also print which one you're currently treating as logits
            self.logger.info("Configured decoder_logits_name = %s", self.decoder_logits_name)


        dt = time.perf_counter() - t0
        self.logger.info("Decoder session.run() returned in %.3fs", dt)

        output_map = {node.name: value for node, value in zip(self.decoder_io.outputs, outputs)}
        logits = output_map[self.decoder_logits_name]

        if self.debug and isinstance(logits, np.ndarray):
            squeezed = np.squeeze(logits)
            self.logger.info(
                "Selected logits tensor shape=%s dtype=%s squeezed_shape=%s squeezed_size=%d",
                logits.shape, logits.dtype, squeezed.shape, squeezed.size
            )
            # ✅ For this exported model, vocab is NOT the last dim; use squeezed size check instead.
            if squeezed.size != 51865:
                self.logger.warning(
                    "⚠️ Logits size unexpected (expected 51865 vocab). Got squeezed_size=%d shape=%s",
                    squeezed.size, logits.shape
                )

        # Update KV cache from *_out
        if self.has_kv_cache:
            new_cache = {name.replace("_out", "_in"): output_map[name] for name in self.kv_self_out_names}
        else:
            new_cache = {}

        return logits, new_cache

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

    def _select_next_token_from_logits(self, logits: np.ndarray) -> int:
        x = np.squeeze(logits)

        # Make sure we end up with [vocab]
        if x.ndim != 1:
            x = x.reshape(-1)

        # QNN often returns fp16 packed as uint16 bits
        if x.dtype == np.uint16:
            # Interpret as float16 bit-patterns (NOT int16)
            x = x.view(np.float16).astype(np.float32)
            scores = x
        else:
            # Normal case
            scores = x.astype(np.float32)

        # Suppress Whisper control/special tokens
        if getattr(self, "suppress_tokens", None):
            for tid in self.suppress_tokens:
                if 0 <= tid < scores.shape[0]:
                    scores[tid] = -1e9

        # ✅ DEBUG: print top-5 once per transcription
        if not getattr(self, "_debug_top5_printed", False):
            self._debug_top5_printed = True
            s = scores.copy()
            top = np.argsort(s)[-5:][::-1]
            self.logger.info("Top-5 token ids: %s", top.tolist())
            self.logger.info(
                "Top-5 tokens: %s",
                [self.tokenizer.decode([int(t)], skip_special_tokens=False) for t in top],
            )
            self.logger.info("Top-5 scores: %s", [float(s[int(t)]) for t in top])

        return int(np.argmax(scores))


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
    # Prompt + KV init
    # --------------------------
    def _build_prompt_ids(self, language: str | None) -> list[int]:
        lang = (language or "en").lower()

        # ✅ Use Whisper's intended decoder start token (usually 50258 = <|startoftranscript|>)
        start = getattr(self.config, "decoder_start_token_id", None)
        if start is None:
            start = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if start is None:
            raise RuntimeError("Cannot resolve Whisper decoder start token id.")

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
                cache[node.name] = np.zeros(shape, dtype=self._numpy_dtype_from_ort(node.type))
        return cache

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

    # --------------------------
    # Logging helpers
    # --------------------------
    def _brief_tensor(self, x: Any) -> str:
        if not isinstance(x, np.ndarray):
            return f"{type(x)}"
        if x.size == 0:
            return f"shape={x.shape} dtype={x.dtype} empty"
        try:
            return f"shape={x.shape} dtype={x.dtype} min={x.min()} max={x.max()}"
        except Exception:
            return f"shape={x.shape} dtype={x.dtype}"

    def _log_decoder_inputs(self, decoder_inputs: dict[str, Any]) -> None:
        for k, v in decoder_inputs.items():
            self.logger.info("DEC IN %-40s %s", k, self._brief_tensor(v))

    # --------------------------
    # Audio + mel
    # --------------------------
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
        n_fft = 400
        hop_length = 160
        win_length = 400
        n_mels = 80
        max_samples = 16000 * 30

        if audio.shape[0] < max_samples:
            audio = np.pad(audio, (0, max_samples - audio.shape[0]))
        else:
            audio = audio[:max_samples]

        audio_tensor = torch.from_numpy(audio)
        window = torch.hann_window(win_length)
        stft = torch.stft(
            audio_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        magnitudes = stft.abs().pow(2.0)

        mel_filters = self._mel_filterbank(n_mels=n_mels, n_fft=n_fft, sample_rate=16000)
        mel_spec = torch.matmul(mel_filters, magnitudes)
        mel_spec = torch.clamp(mel_spec, min=1e-10).log10()
        mel_spec = torch.maximum(mel_spec, mel_spec.max() - 8.0)
        mel_spec = (mel_spec + 4.0) / 4.0

        # Force frames to 3000 (matches encoder input)
        target_frames = 3000
        T = int(mel_spec.shape[-1])
        if T < target_frames:
            mel_spec = torch.nn.functional.pad(mel_spec, (0, target_frames - T))
        else:
            mel_spec = mel_spec[:, :target_frames]

        return mel_spec.unsqueeze(0).numpy().astype(np.float32)

    def _prepare_encoder_features(self, features: np.ndarray) -> np.ndarray:
        # encoder expects uint16
        node = self.encoder_io.inputs[0]
        t = (node.type or "").lower()
        if "uint16" in t:
            scaled = np.clip(features * 65535.0, 0, 65535)
            return scaled.astype(np.uint16)
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
