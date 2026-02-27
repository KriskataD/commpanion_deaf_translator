from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import torch


class WhisperAudioFeaturesMixin:
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

    def _extract_and_pack_features(self, audio: np.ndarray) -> np.ndarray:
        mel = self._log_mel_spectrogram(audio)               # [1,80,3000] float32
        features = self._prepare_encoder_features(mel)       # uint16 expected
        if self.debug:
            self.logger.info("Encoder features min=%s max=%s", int(features.min()), int(features.max()))
        self.logger.info("Encoder features ready. Shape=%s, dtype=%s", features.shape, features.dtype)
        return features

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
