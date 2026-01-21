# src/stt.py  (CPU-only Whisper multilingual via ONNXRuntime)
from __future__ import annotations

from pathlib import Path
from typing import Optional

import wave
import numpy as np

from transformers import WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq


class SpeechToTextApplication:
    """
    Transcribe speech from WAV files using multilingual Whisper-Base exported to ONNX.

    Expects an Optimum ONNX export folder (default):
      src/models/whisper_base_multilingual_onnx/

    This is CPU-only (ONNXRuntime CPUExecutionProvider).
    """

    def __init__(
        self,
        audio_records_path: Path | str | None = None,
        models_dir: Path | str | None = None,
        model_name: str = "whisper_base",
    ) -> None:
        # Keep API compatible with your translator.py (model_name accepted but not required here).
        if isinstance(audio_records_path, str):
            self.audio_records_path: Path | None = Path(audio_records_path)
        else:
            self.audio_records_path: Path | None = audio_records_path

        base_models_dir = Path(models_dir) if models_dir is not None else Path(__file__).parent / "models"
        self.export_dir = base_models_dir / "whisper_base_multilingual_onnx"

        if not self.export_dir.exists():
            raise FileNotFoundError(
                f"Missing ONNX export folder:\n  {self.export_dir}\n\n"
                f"Create it with:\n"
                f"  optimum-cli export onnx --model openai/whisper-base "
                f"--task automatic-speech-recognition --library transformers "
                f"{self.export_dir}\n"
            )

        # Multilingual Whisper processor (tokenizer + feature extractor)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")

        # Load ONNX model (CPU)
        self.model = ORTModelForSpeechSeq2Seq.from_pretrained(
            str(self.export_dir),
            provider="CPUExecutionProvider",
        )

        self.last_audio_file: Optional[Path] = None

    def _get_audio_file(self) -> Path:
        if self.audio_records_path is None:
            raise ValueError("Audio records path is not set.")
        audio_files = list(self.audio_records_path.glob("*.wav"))
        if not audio_files:
            raise FileNotFoundError("No audio files found.")
        # pick newest
        audio_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        self.last_audio_file = audio_files[0]
        return audio_files[0]

    def _delete_audio_file(self) -> None:
        if self.last_audio_file and self.last_audio_file.exists():
            self.last_audio_file.unlink()
            self.last_audio_file = None

    @staticmethod
    def _load_wav_mono_float32(wav_path: Path) -> tuple[np.ndarray, int]:
        """
        Load a PCM WAV file via the standard library and return (audio_float32, sample_rate).
        Assumes 16-bit PCM (your recorder writes pyaudio.paInt16).
        """
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.getnframes()
            raw = wf.readframes(frames)

        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got sampwidth={sampwidth} bytes.")

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio, sr

    @staticmethod
    def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
        """Simple linear resampler (good enough for a working baseline)."""
        if src_sr == dst_sr:
            return audio.astype(np.float32)

        old_n = audio.shape[0]
        new_n = int(round(old_n * (dst_sr / float(src_sr))))
        if old_n == 0 or new_n == 0:
            return np.zeros((0,), dtype=np.float32)

        x_old = np.linspace(0.0, 1.0, num=old_n, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_n, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    def transcribe(self) -> str:
        wav_path = self._get_audio_file()

        audio, sr = self._load_wav_mono_float32(wav_path)
        audio_16k = self._resample_linear(audio, sr, 16000)

        # Prepare features
        inputs = self.processor(audio_16k, sampling_rate=16000, return_tensors="pt")

        # Generate transcription
        generated_ids = self.model.generate(inputs["input_features"])

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f"Transcription result: {text}")

        self._delete_audio_file()
        return text


def is_whisper_base_available() -> bool:
    """Kept for compatibility with translator.py logic."""
    return True
