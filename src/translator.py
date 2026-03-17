"""
Translator pipeline using QNN Whisper STT, M2M100 translation, and optional TTS playback.

Speak into the microphone, the audio is recorded until silence is detected,
transcribed, translated, and the translated text is printed and spoken.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import threading
import time
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Optional

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

import subprocess
import sys

from .npu.whisper_qnn_stt import WhisperQnnSTT
from .recorder import AudioRecorder
from .tts import _TTS
from .captions_client import CaptionsClient

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


class PerformanceTracker:
    """Collects speech-translation and OCR performance metrics."""

    def __init__(self) -> None:
        self.session_start_s = time.perf_counter()
        self.speech_cycles: list[dict[str, Any]] = []
        self.ocr_cycles: list[dict[str, Any]] = []
        self.ocr_command_record_samples: list[float] = []
        self.ocr_command_stt_samples: list[float] = []
        self._process = None
        if psutil is not None:
            try:
                self._process = psutil.Process(os.getpid())
                self._process.cpu_percent(interval=None)
            except Exception:
                self._process = None

    @staticmethod
    def _word_count(text: str) -> int:
        return len((text or "").split())

    @staticmethod
    def _numeric_stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"avg": None, "min": None, "max": None, "std": None}
        return {
            "avg": mean(values),
            "min": min(values),
            "max": max(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }

    def _snapshot_resources(self, cycle: dict[str, Any]) -> None:
        if self._process is None:
            return
        try:
            cycle["rss_memory_mb"] = self._process.memory_info().rss / (1024 * 1024)
        except Exception:
            cycle["rss_memory_mb"] = None
        try:
            cycle["cpu_percent"] = self._process.cpu_percent(interval=None)
        except Exception:
            cycle["cpu_percent"] = None

    def start_speech_cycle(self) -> dict[str, Any]:
        return {
            "mode": "speech_translation",
            "record_time_s": None,
            "stt_time_s": None,
            "translation_time_s": None,
            "tts_time_s": None,
            "cycle_total_time_s": None,
            "input_chars": 0,
            "output_chars": 0,
            "input_words": 0,
            "output_words": 0,
            "success": False,
            "error_stage": None,
            "cpu_percent": None,
            "rss_memory_mb": None,
            "_cycle_start_s": time.perf_counter(),
        }

    def complete_speech_cycle(
        self,
        cycle: dict[str, Any],
        *,
        success: bool,
        error_stage: str | None,
        transcription: str,
        translated: str,
    ) -> None:
        cycle["cycle_total_time_s"] = time.perf_counter() - cycle["_cycle_start_s"]
        cycle["success"] = success
        cycle["error_stage"] = error_stage
        cycle["input_chars"] = len((transcription or "").strip())
        cycle["output_chars"] = len((translated or "").strip())
        cycle["input_words"] = self._word_count(transcription)
        cycle["output_words"] = self._word_count(translated)
        cycle.pop("_cycle_start_s", None)
        self._snapshot_resources(cycle)
        self.speech_cycles.append(cycle)

    def start_ocr_cycle(self) -> dict[str, Any]:
        return {
            "mode": "ocr_detect",
            "ocr_scan_time_s": None,
            "ocr_total_time_s": None,
            "ocr_text_found": False,
            "ocr_chars": 0,
            "ocr_words": 0,
            "ocr_pages": 0,
            "ocr_translated": False,
            "ocr_translate_time_s": None,
            "ocr_translated_chars": 0,
            "ocr_translated_words": 0,
            "ocr_command_cycles": 0,
            "ocr_command_record_time_s": None,
            "ocr_command_stt_time_s": None,
            "ocr_next_count": 0,
            "ocr_previous_count": 0,
            "ocr_translate_count": 0,
            "ocr_stop_count": 0,
            "ocr_unknown_count": 0,
            "success": False,
            "error_stage": None,
            "cpu_percent": None,
            "rss_memory_mb": None,
            "_ocr_start_s": time.perf_counter(),
            "_ocr_command_record_samples": [],
            "_ocr_command_stt_samples": [],
        }

    def record_ocr_translation(self, cycle: dict[str, Any], translated_text: str, elapsed_s: float) -> None:
        cycle["ocr_translated"] = bool((translated_text or "").strip())
        cycle["ocr_translate_time_s"] = elapsed_s
        cycle["ocr_translated_chars"] = len((translated_text or "").strip())
        cycle["ocr_translated_words"] = self._word_count(translated_text)

    def record_ocr_command(
        self,
        cycle: dict[str, Any],
        *,
        record_time_s: float | None,
        stt_time_s: float | None,
        intent: str | None,
    ) -> None:
        cycle["ocr_command_cycles"] += 1
        if record_time_s is not None:
            self.ocr_command_record_samples.append(record_time_s)
            cycle["_ocr_command_record_samples"].append(record_time_s)
        if stt_time_s is not None:
            self.ocr_command_stt_samples.append(stt_time_s)
            cycle["_ocr_command_stt_samples"].append(stt_time_s)

        intent_key = {
            "next": "ocr_next_count",
            "previous": "ocr_previous_count",
            "translate": "ocr_translate_count",
            "stop": "ocr_stop_count",
        }.get(intent, "ocr_unknown_count")
        cycle[intent_key] += 1

    def complete_ocr_cycle(self, cycle: dict[str, Any], *, success: bool, error_stage: str | None) -> None:
        cycle["ocr_total_time_s"] = time.perf_counter() - cycle["_ocr_start_s"]
        cycle["success"] = success
        cycle["error_stage"] = error_stage

        record_samples = cycle.pop("_ocr_command_record_samples", [])
        stt_samples = cycle.pop("_ocr_command_stt_samples", [])
        cycle["ocr_command_record_time_s"] = mean(record_samples) if record_samples else None
        cycle["ocr_command_stt_time_s"] = mean(stt_samples) if stt_samples else None
        cycle.pop("_ocr_start_s", None)

        self._snapshot_resources(cycle)
        self.ocr_cycles.append(cycle)

    def summary(self) -> dict[str, Any]:
        session_wall_time_s = max(0.0, time.perf_counter() - self.session_start_s)

        speech_attempted = len(self.speech_cycles)
        speech_success = sum(1 for c in self.speech_cycles if c.get("success"))
        speech_successful = [c for c in self.speech_cycles if c.get("success")]

        def _speech_vals(key: str) -> list[float]:
            return [c[key] for c in speech_successful if c.get(key) is not None]

        speech_rss = [c["rss_memory_mb"] for c in self.speech_cycles if c.get("rss_memory_mb") is not None]
        speech_cpu = [c["cpu_percent"] for c in self.speech_cycles if c.get("cpu_percent") is not None]

        ocr_attempted = len(self.ocr_cycles)
        ocr_success = sum(1 for c in self.ocr_cycles if c.get("success"))
        ocr_with_text = [c for c in self.ocr_cycles if c.get("ocr_text_found")]

        def _ocr_vals(key: str) -> list[float]:
            return [c[key] for c in self.ocr_cycles if c.get(key) is not None]

        ocr_rss = [c["rss_memory_mb"] for c in self.ocr_cycles if c.get("rss_memory_mb") is not None]
        ocr_cpu = [c["cpu_percent"] for c in self.ocr_cycles if c.get("cpu_percent") is not None]

        return {
            "session_duration_s": session_wall_time_s,
            "speech_translation": {
                "attempted_cycles": speech_attempted,
                "successful_cycles": speech_success,
                "success_rate": (speech_success / speech_attempted) if speech_attempted else 0.0,
                "translations_per_minute": (speech_success / session_wall_time_s * 60.0) if session_wall_time_s > 0 else 0.0,
                "latency_s": {
                    "record_time_s": self._numeric_stats(_speech_vals("record_time_s")),
                    "stt_time_s": self._numeric_stats(_speech_vals("stt_time_s")),
                    "translation_time_s": self._numeric_stats(_speech_vals("translation_time_s")),
                    "tts_time_s": self._numeric_stats(_speech_vals("tts_time_s")),
                    "cycle_total_time_s": self._numeric_stats(_speech_vals("cycle_total_time_s")),
                },
                "text_volume": {
                    "avg_input_words": mean([c["input_words"] for c in speech_successful]) if speech_successful else 0.0,
                    "avg_output_words": mean([c["output_words"] for c in speech_successful]) if speech_successful else 0.0,
                    "avg_input_chars": mean([c["input_chars"] for c in speech_successful]) if speech_successful else 0.0,
                    "avg_output_chars": mean([c["output_chars"] for c in speech_successful]) if speech_successful else 0.0,
                },
                "resources": {
                    "psutil_available": self._process is not None,
                    "avg_rss_memory_mb": mean(speech_rss) if speech_rss else None,
                    "peak_rss_memory_mb": max(speech_rss) if speech_rss else None,
                    "avg_cpu_percent": mean(speech_cpu) if speech_cpu else None,
                },
            },
            "ocr_detect": {
                "scan_attempts": ocr_attempted,
                "successful_scans": ocr_success,
                "success_rate": (ocr_success / ocr_attempted) if ocr_attempted else 0.0,
                "latency_s": {
                    "ocr_scan_time_s": self._numeric_stats(_ocr_vals("ocr_scan_time_s")),
                    "ocr_total_time_s": self._numeric_stats(_ocr_vals("ocr_total_time_s")),
                    "ocr_translate_time_s": self._numeric_stats(_ocr_vals("ocr_translate_time_s")),
                    "ocr_command_record_time_s": self._numeric_stats(self.ocr_command_record_samples),
                    "ocr_command_stt_time_s": self._numeric_stats(self.ocr_command_stt_samples),
                },
                "text_volume": {
                    "avg_ocr_words": mean([c["ocr_words"] for c in ocr_with_text]) if ocr_with_text else 0.0,
                    "avg_ocr_chars": mean([c["ocr_chars"] for c in ocr_with_text]) if ocr_with_text else 0.0,
                    "avg_ocr_pages": mean([c["ocr_pages"] for c in ocr_with_text]) if ocr_with_text else 0.0,
                },
                "commands": {
                    "total_commands": sum(c.get("ocr_command_cycles", 0) for c in self.ocr_cycles),
                    "ocr_next_count": sum(c.get("ocr_next_count", 0) for c in self.ocr_cycles),
                    "ocr_previous_count": sum(c.get("ocr_previous_count", 0) for c in self.ocr_cycles),
                    "ocr_translate_count": sum(c.get("ocr_translate_count", 0) for c in self.ocr_cycles),
                    "ocr_stop_count": sum(c.get("ocr_stop_count", 0) for c in self.ocr_cycles),
                    "ocr_unknown_count": sum(c.get("ocr_unknown_count", 0) for c in self.ocr_cycles),
                },
                "resources": {
                    "psutil_available": self._process is not None,
                    "avg_rss_memory_mb": mean(ocr_rss) if ocr_rss else None,
                    "peak_rss_memory_mb": max(ocr_rss) if ocr_rss else None,
                    "avg_cpu_percent": mean(ocr_cpu) if ocr_cpu else None,
                },
            },
            "speech_cycles": self.speech_cycles,
            "ocr_cycles": self.ocr_cycles,
        }

class MultiLanguageTranslator:
    """Wrapper around the facebook/m2m100_418M translation model."""

    def __init__(self, model_name: str = "facebook/m2m100_418M") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def supported_languages(self) -> list[str]:
        """Return the language codes supported by the tokenizer/model."""

        return sorted(self.tokenizer.lang_code_to_id.keys())

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text from ``source_lang`` to ``target_lang`` using M2M100."""
        cleaned = text.strip()
        if not cleaned:
            return ""

        self.tokenizer.src_lang = source_lang
        encoded = self.tokenizer(cleaned, return_tensors="pt").to(self.device)
        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.get_lang_id(target_lang),
            max_new_tokens=256,
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


class TranslatorPipeline:
    """End-to-end pipeline: record -> transcribe -> translate -> speak/print."""

    def __init__(
        self,
        audio_dir: str | Path = "audio",
        source_lang: str = "en",
        target_lang: str = "fr",
        speak: bool = True,
        qnn_encoder_dir: str | Path = "models/whisper_small_quantized_encoder_optimized_onnx",
        qnn_decoder_dir: str | Path = "models/whisper_small_quantized_decoder_optimized_onnx",
        stt_model: str = "auto",
        stt_timeout: float | None = None,
        tts_timeout: float | None = None,
        launch_captions_overlay: bool = False,
        captions_monitor_index: int | None = None,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.speak = speak
        self.stt_model = stt_model
        self.stt_timeout = stt_timeout
        self.tts_timeout = tts_timeout

        self.logger = logging.getLogger(__name__)
        self.recorder = AudioRecorder()
        self.stt = self._build_stt_backend(qnn_encoder_dir, qnn_decoder_dir)
        self.translator = MultiLanguageTranslator()
        self.tts = _TTS(preferred_lang=self.target_lang) if self.speak else None

        self._captions_overlay_proc = None
        if launch_captions_overlay:
            self._kill_stale_captions_overlays()

            if self._captions_overlay_proc is None or self._captions_overlay_proc.poll() is not None:
                self._captions_overlay_proc = self._launch_captions_overlay(
                    port=37777,
                    monitor_index=captions_monitor_index,
                )

        # captions overlay client (optional)
        try:
            self.captions = CaptionsClient(port=37777)
        except Exception as e:
            self.logger.warning("Captions disabled: %s", e)
            self.captions = None
        self.last_audio_path: Path | None = None
        self.performance = PerformanceTracker()

        self._mic_lock = threading.Lock()
        default_mic = self.recorder.mic_selector.get_default_microphone()
        if default_mic:
            self.recorder.set_microphone(default_mic["index"])
            print(f"🎚️ Default microphone selected: {default_mic['name']}")
        else:
            print("❌ No microphones available. Recording will fail.")

    def record(self, filename: str = "last_rec.wav") -> Optional[Path]:
        """Record until silence and save the audio file inside audio_dir."""
        with self._mic_lock:
            print("🎙️ Start speaking (recording stops automatically on silence)...")
            self.recorder.start_recording()
            while self.recorder.is_recording:
                time.sleep(0.1)
            output_path = self.audio_dir / filename
            self.recorder.save_recording(str(output_path))
            self.recorder.cleanup()
        self.last_audio_path = output_path if output_path.exists() else None
        return self.last_audio_path

    def _run_with_timeout(self, fn: Callable[[], str], label: str) -> str:
        if self.stt_timeout is None:
            return fn()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn)
            try:
                return future.result(timeout=self.stt_timeout)
            except concurrent.futures.TimeoutError:
                self.logger.error("%s timed out after %.1fs.", label, self.stt_timeout)
                future.cancel()
                return ""
            except Exception:
                self.logger.exception("%s failed.", label)
                return ""

    def transcribe(self, language_override: str | None = None, delete: bool = True) -> str:
        """Transcribe the last recorded audio file using Whisper."""
        if not self.last_audio_path:
            raise FileNotFoundError("No recorded audio available for QNN STT.")
        try:
            return self._run_with_timeout(
                lambda: self.stt.transcribe_wav(self.last_audio_path, language=language_override),
                "QNN STT transcription",
            )
        finally:
            if delete:
                self.delete_last_audio_file()

    def _build_stt_backend(self, qnn_encoder_dir: str | Path, qnn_decoder_dir: str | Path):
        encoder_dir, decoder_dir = self._resolve_qnn_model_dirs(qnn_encoder_dir, qnn_decoder_dir)
        self.logger.info("QNN encoder directory: %s", encoder_dir)
        self.logger.info("QNN decoder directory: %s", decoder_dir)
        return WhisperQnnSTT(
            encoder_dir=encoder_dir,
            decoder_dir=decoder_dir,
            stt_model=self.stt_model,
            debug=True,
        )

    def _resolve_qnn_model_dirs(
        self,
        qnn_encoder_dir: str | Path,
        qnn_decoder_dir: str | Path,
    ) -> tuple[Path, Path]:
        env_encoder = os.getenv("QNN_ENCODER_DIR")
        env_decoder = os.getenv("QNN_DECODER_DIR")

        base_dir = Path(__file__).resolve().parent
        candidates = [
            (Path(qnn_encoder_dir), Path(qnn_decoder_dir)),
            (Path(qnn_encoder_dir).expanduser(), Path(qnn_decoder_dir).expanduser()),
            (base_dir / qnn_encoder_dir, base_dir / qnn_decoder_dir),
            (base_dir.parent / qnn_encoder_dir, base_dir.parent / qnn_decoder_dir),
        ]

        if env_encoder and env_decoder:
            candidates.insert(0, (Path(env_encoder), Path(env_decoder)))

        for encoder_dir, decoder_dir in candidates:
            if encoder_dir.exists() and decoder_dir.exists():
                return encoder_dir, decoder_dir

        attempted = " | ".join(f"{enc} / {dec}" for enc, dec in candidates)
        raise FileNotFoundError(
            "QNN Whisper model directories not found. "
            "Set --qnn-encoder-dir/--qnn-decoder-dir or QNN_ENCODER_DIR/QNN_DECODER_DIR. "
            f"Attempted: {attempted}"
        )

    def delete_last_audio_file(self) -> None:
        if self.last_audio_path and self.last_audio_path.exists():
            self.last_audio_path.unlink()
        self.last_audio_path = None

    def set_languages(self, source_lang: str, target_lang: str) -> None:
        """Update the language pair for subsequent translations."""
        self.source_lang = source_lang
        self.target_lang = target_lang
        if self.tts:
            self.tts.set_language(target_lang)

    def show_caption(
        self,
        text: str,
        ttl_ms: int | None = 9000,
        *,
        format_text: bool = True,
    ) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        try:
            if getattr(self, "captions", None):
                self.captions.send(cleaned, ttl_ms=ttl_ms, format_text=format_text)
        except Exception as e:
            self.logger.warning("Failed to send captions: %s", e)

    def speak_text(
        self,
        text: str,
        *,
        timeout_s: float | None = None,
        ttl_ms: float | None = 9000,
        show_caption: bool = True,
    ) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return

        if show_caption:
            self.show_caption(cleaned, ttl_ms=ttl_ms)

        if self.speak and self.tts:
            self.tts.start(cleaned, timeout_s=timeout_s)

    def translate_transcription(self, transcription: str, *, skip_tts: bool = False) -> str:
        translated = self.translator.translate(transcription, self.source_lang, self.target_lang)
        if translated:
            self.show_caption(translated, ttl_ms=9000)
            if not skip_tts and self.speak and self.tts:
                try:
                    self.tts.start(translated, timeout_s=self.tts_timeout)
                except Exception as e:
                    self.logger.warning("TTS failed: %s", e)
        print(f"➡️  Translated ({self.source_lang} → {self.target_lang}): {translated}")
        return translated

    def translate_text(self, text: str) -> str:
        translated = self.translator.translate(text, self.source_lang, self.target_lang)
        if translated:
            self.show_caption(translated, ttl_ms=9000)
            if self.speak and self.tts:
                try:
                    self.tts.start(translated, timeout_s=self.tts_timeout)
                except Exception as e:
                    self.logger.warning("TTS failed: %s", e)
        print(f"➡️  Translated text ({self.source_lang} → {self.target_lang}): {translated}")
        return translated

    def run_once(self) -> str:
        """Record audio once and return the translated text."""
        cycle = self.performance.start_speech_cycle()
        transcription = ""
        translated = ""
        error_stage: str | None = None

        record_start = time.perf_counter()
        audio_path = self.record()
        cycle["record_time_s"] = time.perf_counter() - record_start
        if not audio_path:
            print("❌ Failed to capture audio.")
            self.performance.complete_speech_cycle(
                cycle,
                success=False,
                error_stage="record",
                transcription=transcription,
                translated=translated,
            )
            return ""

        print("📝 Transcribing...")
        stt_start = time.perf_counter()
        try:
            transcription = self.transcribe(language_override=self.source_lang, delete=False)
            cycle["stt_time_s"] = time.perf_counter() - stt_start
        except Exception as exc:
            cycle["stt_time_s"] = time.perf_counter() - stt_start
            error_stage = "stt"
            self.logger.exception("Transcription failed.")
            print(f"❌ Transcription failed: {exc}")
            self.performance.complete_speech_cycle(
                cycle,
                success=False,
                error_stage=error_stage,
                transcription=transcription,
                translated=translated,
            )
            return ""
        print(f"Original text ({self.source_lang}): {transcription}")

        print("🌐 Translating...")
        translation_start = time.perf_counter()
        try:
            translated = self.translate_transcription(transcription, skip_tts=True)
            cycle["translation_time_s"] = time.perf_counter() - translation_start
        except Exception as exc:
            cycle["translation_time_s"] = time.perf_counter() - translation_start
            error_stage = "translate"
            self.logger.exception("Translation failed.")
            print(f"❌ Translation failed: {exc}")
            self.performance.complete_speech_cycle(
                cycle,
                success=False,
                error_stage=error_stage,
                transcription=transcription,
                translated=translated,
            )
            return ""

        tts_elapsed = 0.0 if not (self.speak and self.tts) else None
        if self.speak and self.tts:
            tts_start = time.perf_counter()
            try:
                self.tts.start(translated, timeout_s=self.tts_timeout)
                tts_elapsed = time.perf_counter() - tts_start
            except Exception as exc:
                tts_elapsed = time.perf_counter() - tts_start
                cycle["tts_time_s"] = tts_elapsed
                error_stage = "tts"
                self.logger.warning("TTS failed: %s", exc)
                self.performance.complete_speech_cycle(
                    cycle,
                    success=False,
                    error_stage=error_stage,
                    transcription=transcription,
                    translated=translated,
                )
                return translated
        cycle["tts_time_s"] = tts_elapsed
        self.performance.complete_speech_cycle(
            cycle,
            success=True,
            error_stage=None,
            transcription=transcription,
            translated=translated,
        )
        return translated

    @staticmethod
    def _fmt_stat(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2f}"

    def get_performance_summary(self) -> dict[str, Any]:
        return self.performance.summary()

    def print_performance_summary(self) -> None:
        summary = self.get_performance_summary()
        speech = summary["speech_translation"]
        ocr = summary["ocr_detect"]

        def _line(label: str, stats: dict[str, float | None]) -> str:
            return (
                f"  {label:<12} avg={self._fmt_stat(stats['avg'])}  "
                f"min={self._fmt_stat(stats['min'])}  "
                f"max={self._fmt_stat(stats['max'])}  "
                f"std={self._fmt_stat(stats['std'])}"
            )

        print("\n========== SESSION PERFORMANCE SUMMARY ==========")
        print(f"Session duration: {summary['session_duration_s']:.2f} s")
        print(f"Total cycles attempted: {speech['attempted_cycles']}")
        print(f"Successful cycles: {speech['successful_cycles']}")
        print(f"Success rate: {speech['success_rate'] * 100:.2f}%")
        print(f"Translations per minute: {speech['translations_per_minute']:.2f}")
        print("\nLatency metrics (seconds)")
        print(_line("Record", speech["latency_s"]["record_time_s"]))
        print(_line("STT", speech["latency_s"]["stt_time_s"]))
        print(_line("Translation", speech["latency_s"]["translation_time_s"]))
        print(_line("TTS", speech["latency_s"]["tts_time_s"]))
        print(_line("Total", speech["latency_s"]["cycle_total_time_s"]))
        print("\nText volume")
        print(f"  Input words   avg={speech['text_volume']['avg_input_words']:.2f}")
        print(f"  Output words  avg={speech['text_volume']['avg_output_words']:.2f}")
        print(f"  Input chars   avg={speech['text_volume']['avg_input_chars']:.2f}")
        print(f"  Output chars  avg={speech['text_volume']['avg_output_chars']:.2f}")
        print("\nResources")
        if speech["resources"]["psutil_available"]:
            print(
                f"  RSS memory MB avg={self._fmt_stat(speech['resources']['avg_rss_memory_mb'])}  "
                f"peak={self._fmt_stat(speech['resources']['peak_rss_memory_mb'])}"
            )
            print(f"  CPU %         avg={self._fmt_stat(speech['resources']['avg_cpu_percent'])}")
        else:
            print("  psutil unavailable; resource metrics disabled.")

        print("\n========== OCR / DETECT MODE SUMMARY ==========")
        print(f"OCR scan attempts: {ocr['scan_attempts']}")
        print(f"Successful OCR scans: {ocr['successful_scans']}")
        print(f"OCR success rate: {ocr['success_rate'] * 100:.2f}%")
        print("\nOCR latency metrics (seconds)")
        print(_line("Scan", ocr["latency_s"]["ocr_scan_time_s"]))
        print(_line("Detect total", ocr["latency_s"]["ocr_total_time_s"]))
        print(_line("OCR translate", ocr["latency_s"]["ocr_translate_time_s"]))
        print("\nOCR text volume")
        print(f"  OCR words     avg={ocr['text_volume']['avg_ocr_words']:.2f}")
        print(f"  OCR chars     avg={ocr['text_volume']['avg_ocr_chars']:.2f}")
        print(f"  OCR pages     avg={ocr['text_volume']['avg_ocr_pages']:.2f}")
        print("\nOCR voice commands")
        print(f"  Commands total: {ocr['commands']['total_commands']}")
        print(_line("Record", ocr["latency_s"]["ocr_command_record_time_s"]))
        print(_line("STT", ocr["latency_s"]["ocr_command_stt_time_s"]))
        print(
            "  "
            f"next={ocr['commands']['ocr_next_count']}  "
            f"previous={ocr['commands']['ocr_previous_count']}  "
            f"translate={ocr['commands']['ocr_translate_count']}  "
            f"stop={ocr['commands']['ocr_stop_count']}  "
            f"unknown={ocr['commands']['ocr_unknown_count']}"
        )
        if ocr["resources"]["psutil_available"]:
            print(
                f"  OCR RSS MB    avg={self._fmt_stat(ocr['resources']['avg_rss_memory_mb'])}  "
                f"peak={self._fmt_stat(ocr['resources']['peak_rss_memory_mb'])}"
            )
            print(f"  OCR CPU %     avg={self._fmt_stat(ocr['resources']['avg_cpu_percent'])}")
        print("===============================================")

    def save_performance_reports(
        self,
        logs_dir: str | Path = "logs",
    ) -> tuple[Path, Path, Path, Path, Path, Path]:
        summary = self.get_performance_summary()
        logs_path = Path(logs_dir)
        logs_path.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        summary_latest = logs_path / "perf_summary_latest.json"
        summary_timestamped = logs_path / f"perf_summary_{timestamp}.json"
        speech_latest = logs_path / "perf_cycles_latest.csv"
        speech_timestamped = logs_path / f"perf_cycles_{timestamp}.csv"
        ocr_latest = logs_path / "perf_ocr_cycles_latest.csv"
        ocr_timestamped = logs_path / f"perf_ocr_cycles_{timestamp}.csv"

        summary_payload = {
            "session_duration_s": summary["session_duration_s"],
            "speech_translation": summary["speech_translation"],
            "ocr_detect": summary["ocr_detect"],
        }
        for target in (summary_latest, summary_timestamped):
            target.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        speech_fields = [
            "mode",
            "record_time_s",
            "stt_time_s",
            "translation_time_s",
            "tts_time_s",
            "cycle_total_time_s",
            "input_chars",
            "output_chars",
            "input_words",
            "output_words",
            "success",
            "error_stage",
            "rss_memory_mb",
            "cpu_percent",
        ]
        for target in (speech_latest, speech_timestamped):
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=speech_fields)
                writer.writeheader()
                for cycle in self.performance.speech_cycles:
                    writer.writerow({field: cycle.get(field) for field in speech_fields})

        ocr_fields = [
            "mode",
            "ocr_scan_time_s",
            "ocr_total_time_s",
            "ocr_text_found",
            "ocr_chars",
            "ocr_words",
            "ocr_pages",
            "ocr_translated",
            "ocr_translate_time_s",
            "ocr_translated_chars",
            "ocr_translated_words",
            "ocr_command_cycles",
            "ocr_command_record_time_s",
            "ocr_command_stt_time_s",
            "ocr_next_count",
            "ocr_previous_count",
            "ocr_translate_count",
            "ocr_stop_count",
            "ocr_unknown_count",
            "success",
            "error_stage",
            "rss_memory_mb",
            "cpu_percent",
        ]
        for target in (ocr_latest, ocr_timestamped):
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=ocr_fields)
                writer.writeheader()
                for cycle in self.performance.ocr_cycles:
                    writer.writerow({field: cycle.get(field) for field in ocr_fields})

        return (
            summary_latest,
            speech_latest,
            ocr_latest,
            summary_timestamped,
            speech_timestamped,
            ocr_timestamped,
        )

    def run_loop(self) -> None:
        """Continuously record, translate, and optionally speak until interrupted."""

        print("Press Ctrl+C or type 'q' when prompted to stop translating.")
        try:
            while True:
                self.run_once()
                try:
                    user_input = input("Press Enter to translate again or type 'q' to quit: ").strip().lower()
                except EOFError:
                    break
                if user_input == "q":
                    break
                if user_input:
                    print("✏️  To change languages, restart with --source-lang/--target-lang arguments.")
        except KeyboardInterrupt:
            print("\n🛑 Translation loop interrupted by user.")

    def _launch_captions_overlay(self, port: int, monitor_index: int | None):
        project_root = Path(__file__).resolve().parent.parent

        exe = Path(sys.executable)
        pythonw = exe.with_name("pythonw.exe")
        launcher = str(pythonw if pythonw.exists() else exe)

        cmd = [
            launcher,
            "-m",
            "src.captions_overlay",
            "--port",
            str(port),
            "--font-size",
            "36",
            "--width",
            "1900",
            "--height",
            "140",
            "--prefer-non-primary",
            "--cover-taskbar",
            "--bottom-margin",
            "0",
        ]

        if monitor_index is not None:
            cmd.extend(["--monitor-index", str(monitor_index)])

        return subprocess.Popen(cmd, cwd=str(project_root))
    
    def shutdown_captions_overlay(self) -> None:
        self.clear_captions()

        proc = getattr(self, "_captions_overlay_proc", None)
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        pass
        except Exception as e:
            self.logger.warning("Failed to close tracked captions overlay: %s", e)
        finally:
            self._captions_overlay_proc = None

        self._kill_stale_captions_overlays()

    def clear_captions(self) -> None:
        try:
            if getattr(self, "captions", None):
                self.captions.clear()
        except Exception:
            pass

    def _kill_stale_captions_overlays(self) -> None:
        if os.name != "nt":
            return

        script = r"""
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and (
            $_.CommandLine -match 'src\.captions_overlay' -or
            $_.CommandLine -match 'captions_overlay\.py'
        )
    }
    foreach ($p in $procs) {
        try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {}
    }
    """
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self.logger.warning("Failed to sweep stale caption overlays: %s", e)


def _print_language_list(translator: MultiLanguageTranslator) -> None:
    print("Supported language codes (M2M100):")
    codes = translator.supported_languages()
    for i in range(0, len(codes), 8):
        print("  " + ", ".join(codes[i : i + 8]))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Speech translator using QNN Whisper STT and M2M100.")
    parser.add_argument("--source-lang", default="en", help="Source language code (e.g., en, fr, es).")
    parser.add_argument("--target-lang", default="fr", help="Target language code (e.g., fr, en, de).")
    parser.add_argument("--audio-dir", default="audio", help="Directory to save recordings.")
    parser.add_argument(
        "--qnn-encoder-dir",
        default="models/whisper_small_quantized_encoder_optimized_onnx",
        help="Directory containing the QNN Whisper encoder ONNX model (or set QNN_ENCODER_DIR).",
    )
    parser.add_argument(
        "--qnn-decoder-dir",
        default="models/whisper_small_quantized_decoder_optimized_onnx",
        help="Directory containing the QNN Whisper decoder ONNX model (or set QNN_DECODER_DIR).",
    )
    parser.add_argument(
        "--stt-model",
        default="auto",
        choices=["auto", "small-quantized", "large-v3-turbo"],
        help="Which Whisper STT backend profile to use.",
    )
    parser.add_argument(
        "--stt-timeout",
        type=float,
        help="Optional timeout (seconds) for STT transcription to avoid hangs.",
    )
    parser.add_argument("--no-speak", action="store_true", help="Disable TTS playback of translations.")
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="Print supported language codes and exit.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single record/transcribe/translate cycle instead of a loop.",
    )
    parser.add_argument("--captions-auto-start", action="store_true", help="Launch captions overlay automatically.")
    parser.add_argument("--captions-monitor-index", type=int, default=None, help="Windows monitor index for captions overlay.")


    args = parser.parse_args()

    default_small_encoder = "models/whisper_small_quantized_encoder_optimized_onnx"
    default_small_decoder = "models/whisper_small_quantized_decoder_optimized_onnx"
    if (
        args.stt_model == "large-v3-turbo"
        and args.qnn_encoder_dir == default_small_encoder
        and args.qnn_decoder_dir == default_small_decoder
    ):
        args.qnn_encoder_dir = "models/whisper_large_v3_turbo_encoder_optimized_onnx"
        args.qnn_decoder_dir = "models/whisper_large_v3_turbo_decoder_optimized_onnx"

    pipeline = TranslatorPipeline(
        audio_dir=args.audio_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        speak=not args.no_speak,
        qnn_encoder_dir=args.qnn_encoder_dir,
        qnn_decoder_dir=args.qnn_decoder_dir,
        stt_model=args.stt_model,
        stt_timeout=args.stt_timeout,
        launch_captions_overlay=args.captions_auto_start,
        captions_monitor_index=args.captions_monitor_index,
    )

    if args.list_languages:
        _print_language_list(pipeline.translator)
        return

    if args.once:
        pipeline.run_once()
    else:
        pipeline.run_loop()


if __name__ == "__main__":
    main()
