"""Whisper Small Quantized STT using ONNX Runtime + QNN Execution Provider."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import wave

import numpy as np
import onnxruntime as ort
import torch
from transformers import WhisperTokenizer

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
    """Structured IO metadata for an ONNX session."""

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
        self.encoder_dir = Path(encoder_dir)
        self.decoder_dir = Path(decoder_dir)
        self.prefer_qnn = prefer_qnn
        self.debug = debug

        self.encoder_onnx = self.encoder_dir / "model.onnx"
        self.decoder_onnx = self.decoder_dir / "model.onnx"
        self._validate_model_files(self.encoder_onnx)
        self._validate_model_files(self.decoder_onnx)

        self.encoder_session = make_session(self.encoder_onnx)
        self.decoder_session = make_session(self.decoder_onnx)

        self.encoder_io = SessionIoInfo(
            inputs=self.encoder_session.get_inputs(),
            outputs=self.encoder_session.get_outputs(),
        )
        self.decoder_io = SessionIoInfo(
            inputs=self.decoder_session.get_inputs(),
            outputs=self.decoder_session.get_outputs(),
        )

        self.encoder_input_name = self._find_encoder_input_name()
        self.encoder_output_name = self._find_encoder_output_name()

        self.decoder_input_ids_name = self._find_decoder_input_ids_name()
        self.decoder_encoder_states_name = self._find_decoder_encoder_states_name()
        self.decoder_attention_mask_name = self._find_name(
            self.decoder_io.inputs, ["decoder_attention_mask", "attention_mask"]
        )
        self.decoder_encoder_attention_mask_name = self._find_name(
            self.decoder_io.inputs, ["encoder_attention_mask", "encoder_mask"]
        )
        self.decoder_logits_name = self._find_decoder_logits_name()

        self.past_input_names = [
            node.name for node in self.decoder_io.inputs if "past" in node.name.lower()
        ]
        self.present_output_names = [
            node.name
            for node in self.decoder_io.outputs
            if "present" in node.name or "past" in node.name
        ]
        self.has_kv_cache = bool(self.past_input_names and self.present_output_names)

        if self.debug:
            print("Decoder KV-cache enabled:", self.has_kv_cache)
            encoder_providers = self.encoder_session.get_providers()
            decoder_providers = self.decoder_session.get_providers()
            print("Selected providers (encoder):", encoder_providers)
            print("Selected providers (decoder):", decoder_providers)
            print("QNN selected (encoder):", "QNNExecutionProvider" in encoder_providers)
            print("QNN selected (decoder):", "QNNExecutionProvider" in decoder_providers)

        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")

    def dump_io(self) -> None:
        """Print encoder/decoder IO metadata for debugging and adaptation."""
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
        """Transcribe a WAV file to text."""
        audio = self._load_wav_mono_16k(Path(wav_path))
        features = self._log_mel_spectrogram(audio)
        encoder_outputs = self.encoder_session.run(
            [self.encoder_output_name],
            {self.encoder_input_name: features},
        )
        encoder_hidden_states = encoder_outputs[0]

        prompt_ids = self._build_prompt_ids(language)
        input_ids: list[int] = prompt_ids.copy()
        max_new_tokens = 448
        eot_token = getattr(self.tokenizer, "eos_token_id", None)
        if eot_token is None:
            raise RuntimeError("Tokenizer does not define eos_token_id.")

        past_cache = self._initialize_past_cache() if self.has_kv_cache else None
        cache_ready = False
        for _ in range(max_new_tokens):
            input_ids_to_feed = input_ids if not cache_ready else [input_ids[-1]]
            decoder_inputs: dict[str, Any] = {
                self.decoder_input_ids_name: np.array([input_ids_to_feed], dtype=np.int64),
                self.decoder_encoder_states_name: encoder_hidden_states,
            }

            if self.decoder_attention_mask_name:
                decoder_inputs[self.decoder_attention_mask_name] = np.ones(
                    (1, len(input_ids_to_feed)), dtype=np.int64
                )

            if self.decoder_encoder_attention_mask_name:
                decoder_inputs[self.decoder_encoder_attention_mask_name] = np.ones(
                    (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]),
                    dtype=np.int64,
                )

            if past_cache is not None:
                decoder_inputs.update(past_cache)

            outputs = self.decoder_session.run(None, decoder_inputs)
            output_map = {node.name: value for node, value in zip(self.decoder_io.outputs, outputs)}

            logits = output_map.get(self.decoder_logits_name)
            if logits is None:
                raise RuntimeError(
                    "Decoder outputs missing logits. "
                    f"Got outputs: {[node.name for node in self.decoder_io.outputs]}"
                )

            next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            input_ids.append(next_token)

            if past_cache is not None:
                past_cache = {
                    name: output_map[name]
                    for name in self.present_output_names
                    if name in output_map
                }
                cache_ready = True

            if next_token == eot_token:
                break

        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        return decoded.strip()

    def _validate_model_files(self, onnx_path: Path) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")
        weights_path = onnx_path.with_suffix(".bin")
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing external weights file: {weights_path}")

    def _find_name(self, nodes: Iterable[ort.NodeArg], candidates: list[str]) -> str | None:
        for candidate in candidates:
            for node in nodes:
                if node.name.lower() == candidate.lower():
                    return node.name
            for node in nodes:
                if candidate.lower() in node.name.lower():
                    return node.name
        return None

    def _find_encoder_input_name(self) -> str:
        name = self._find_name(self.encoder_io.inputs, ["input_features", "input"])
        if name is None:
            raise RuntimeError(self._format_io_error("encoder input", self.encoder_io))
        return name

    def _find_encoder_output_name(self) -> str:
        name = self._find_name(self.encoder_io.outputs, ["last_hidden_state", "hidden", "output"])
        if name is None:
            raise RuntimeError(self._format_io_error("encoder output", self.encoder_io))
        return name

    def _find_decoder_input_ids_name(self) -> str:
        name = self._find_name(self.decoder_io.inputs, ["input_ids", "decoder_input_ids"])
        if name is None:
            raise RuntimeError(self._format_io_error("decoder input_ids", self.decoder_io))
        return name

    def _find_decoder_encoder_states_name(self) -> str:
        name = self._find_name(self.decoder_io.inputs, ["encoder_hidden_states", "encoder_outputs"])
        if name is None:
            raise RuntimeError(self._format_io_error("decoder encoder states", self.decoder_io))
        return name

    def _find_decoder_logits_name(self) -> str:
        name = self._find_name(self.decoder_io.outputs, ["logits", "logit"])
        if name is None:
            raise RuntimeError(self._format_io_error("decoder logits", self.decoder_io))
        return name

    def _format_io_error(self, label: str, io_info: SessionIoInfo) -> str:
        inputs = [node.name for node in io_info.inputs]
        outputs = [node.name for node in io_info.outputs]
        return (
            f"Unable to find {label}. "
            f"Inputs: {inputs}. Outputs: {outputs}. Call dump_io() to inspect."
        )

    def _build_prompt_ids(self, language: str | None) -> list[int]:
        if hasattr(self.tokenizer, "get_decoder_prompt_ids"):
            prompt_items = self.tokenizer.get_decoder_prompt_ids(
                language=language or "en",
                task="transcribe",
            )
            return [token_id for _, token_id in prompt_items]

        bos = self.tokenizer.bos_token_id
        if bos is None:
            raise RuntimeError("Tokenizer does not define bos_token_id.")
        return [bos]

    def _initialize_past_cache(self) -> dict[str, np.ndarray]:
        cache: dict[str, np.ndarray] = {}
        for node in self.decoder_io.inputs:
            if node.name not in self.past_input_names:
                continue
            shape = self._resolve_past_shape(node)
            dtype = np.float16 if "float16" in node.type else np.float32
            cache[node.name] = np.zeros(shape, dtype=dtype)
        return cache

    def _resolve_past_shape(self, node: ort.NodeArg) -> tuple[int, ...]:
        if node.shape is None:
            raise RuntimeError(
                "Unable to resolve past_key_values shape; check decoder IO with dump_io()."
            )
        resolved_shape: list[int] = []
        for dim in node.shape:
            if isinstance(dim, int) and dim > 0:
                resolved_shape.append(dim)
                continue
            dim_label = str(dim).lower()
            if "batch" in dim_label:
                resolved_shape.append(1)
            elif "seq" in dim_label or "past" in dim_label:
                resolved_shape.append(0)
            else:
                resolved_shape.append(1)
        if not resolved_shape:
            raise RuntimeError(
                "Unable to resolve past_key_values shape; check decoder IO with dump_io()."
            )
        return tuple(resolved_shape)

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

        return mel_spec.unsqueeze(0).numpy().astype(np.float32)

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
                filter_bank[i - 1, start:center] = (
                    np.arange(start, center) - start
                ) / (center - start)
            if end > center:
                filter_bank[i - 1, center:end] = (
                    end - np.arange(center, end)
                ) / (end - center)

        return torch.from_numpy(filter_bank).float()
