"""
Wake-word driven pipeline: listen for wake word, record a command,
run STT -> translate -> TTS. A placeholder branch is kept for future
sign language detection.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import threading
import time
from pathlib import Path
from typing import Callable

from .translator import TranslatorPipeline
from .wakeword_detector import WakeWordDetector

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


class WakeWordTranslationAssistant:
    """Coordinates wake-word detection with the translation pipeline."""

    def __init__(
        self,
        audio_dir: Path | str = "audio",
        source_lang: str = "en",
        target_lang: str = "fr",
        stt_model: str | None = None,
        wakeword_models: list[str] | None = None,
        wakeword_threshold: float = 0.25,
        wakeword_device_index: int | None = None,
        wakeword_debug: bool = False,
        wakeword_debug_interval: float = 1.0,
        speak: bool = True,
        prompt_user: bool = True,
        stay_awake: bool = False,
        tts_timeout: float | None = None,
    ) -> None:
        WakeWordDetector.download_models()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.translation = TranslatorPipeline(
            audio_dir=audio_dir,
            source_lang=source_lang,
            target_lang=target_lang,
            speak=speak,
            stt_model=stt_model,
            tts_timeout=tts_timeout,
        )
        self.prompt_user = prompt_user
        self.stay_awake = stay_awake
        self.tts_timeout = tts_timeout
        default_mic = self.translation.recorder.mic_selector.get_default_microphone()
        default_wake_device = default_mic["index"] if default_mic else None
        if wakeword_device_index is None and default_wake_device is not None:
            self.logger.info(
                "Using default microphone index %s for wake word detection.",
                default_wake_device,
            )
        self.wakeword_models = wakeword_models or ["hey_jarvis"]
        self.detector = WakeWordDetector(
            wakeword_models=self.wakeword_models,
            threshold=wakeword_threshold,
            input_device_index=wakeword_device_index if wakeword_device_index is not None else default_wake_device,
            log_predictions=wakeword_debug,
            log_interval_s=wakeword_debug_interval,
        )

        # register callbacks for each wake word model name
        for ww in self.wakeword_models:
            self.detector.register_callback(ww, self._on_wake_word_detected)

        self._processing_lock = threading.Lock()
        self._is_processing = False
        self._should_exit = False

        self._session_start = time.perf_counter()
        self._attempted_cycles = 0
        self._successful_cycles = 0
        self._record_times: list[float] = []
        self._stt_times: list[float] = []
        self._translation_times: list[float] = []
        self._tts_times: list[float] = []
        self._cycle_times: list[float] = []
        self._input_words: list[int] = []
        self._output_words: list[int] = []
        self._input_chars: list[int] = []
        self._output_chars: list[int] = []
        self._rss_samples: list[float] = []
        self._cpu_samples: list[float] = []
        self._process = psutil.Process() if psutil else None
        if self._process:
            self._process.cpu_percent(interval=None)

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"avg": None, "min": None, "max": None, "std": None}
        avg = sum(values) / len(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        return {"avg": avg, "min": min(values), "max": max(values), "std": std}

    @staticmethod
    def _word_count(text: str) -> int:
        return len([token for token in text.strip().split() if token])

    def _sample_resources(self) -> None:
        if self._process is None:
            return
        rss_mb = self._process.memory_info().rss / (1024 * 1024)
        cpu_pct = self._process.cpu_percent(interval=None)
        self._rss_samples.append(rss_mb)
        self._cpu_samples.append(cpu_pct)

    def _build_metrics(self) -> dict[str, object]:
        session_duration = time.perf_counter() - self._session_start
        translations_per_minute = (
            self._successful_cycles / (session_duration / 60.0)
            if session_duration > 0
            else math.nan
        )
        return {
            "session_duration_s": session_duration,
            "speech_translation": {
                "mode": "wake_translation_assistant",
                "attempted_cycles": self._attempted_cycles,
                "successful_cycles": self._successful_cycles,
                "success_rate": (
                    self._successful_cycles / self._attempted_cycles
                    if self._attempted_cycles
                    else 0.0
                ),
                "translations_per_minute": translations_per_minute,
                "latency_s": {
                    "record_time_s": self._stats(self._record_times),
                    "stt_time_s": self._stats(self._stt_times),
                    "translation_time_s": self._stats(self._translation_times),
                    "tts_time_s": self._stats(self._tts_times),
                    "cycle_total_time_s": self._stats(self._cycle_times),
                },
                "text_volume": {
                    "avg_input_words": (
                        sum(self._input_words) / len(self._input_words)
                        if self._input_words
                        else 0
                    ),
                    "avg_output_words": (
                        sum(self._output_words) / len(self._output_words)
                        if self._output_words
                        else 0
                    ),
                    "avg_input_chars": (
                        sum(self._input_chars) / len(self._input_chars)
                        if self._input_chars
                        else 0
                    ),
                    "avg_output_chars": (
                        sum(self._output_chars) / len(self._output_chars)
                        if self._output_chars
                        else 0
                    ),
                },
                "resources": {
                    "psutil_available": psutil is not None,
                    "avg_rss_memory_mb": (
                        sum(self._rss_samples) / len(self._rss_samples)
                        if self._rss_samples
                        else None
                    ),
                    "peak_rss_memory_mb": max(self._rss_samples) if self._rss_samples else None,
                    "avg_cpu_percent": (
                        sum(self._cpu_samples) / len(self._cpu_samples)
                        if self._cpu_samples
                        else None
                    ),
                },
            },
        }

    @staticmethod
    def _get_stop_intent(transcription: str) -> str | None:
        normalized = " ".join(transcription.lower().split())
        if "stop jarvis" in normalized:
            return "stop_program"
        if "stop listening" in normalized:
            return "stop_listening"
        return None

    @staticmethod
    def _is_stop_command(transcription: str) -> bool:
        normalized = " ".join(transcription.lower().split())
        return "stop listening" in normalized or "stop jarvis" in normalized

    @staticmethod
    def _get_mode_intent(transcription: str) -> str | None:
        normalized = " ".join(transcription.lower().split())
        if "translate" in normalized:
            return "translate"
        if "detect" in normalized:
            return "detect"
        return None

    def _handle_translate_mode(self) -> str | None:
        cycle_start = time.perf_counter()
        self._attempted_cycles += 1

        if self.translation.tts:
            self.translation.tts.start(
                "Ready to translate. Please say what you want translated.",
                timeout_s=self.tts_timeout,
            )
        time.sleep(0.1)
        record_start = time.perf_counter()
        audio_path = self.translation.record(filename="last_rec.wav")
        self._record_times.append(time.perf_counter() - record_start)
        if audio_path:
            self.logger.info("Translation audio captured: %s", audio_path)
        if not audio_path:
            self._stt_times.append(0.0)
            self._translation_times.append(0.0)
            self._tts_times.append(0.0)
            self._cycle_times.append(time.perf_counter() - cycle_start)
            self._sample_resources()
            self.logger.warning("No audio captured for translation.")
            if self.translation.tts:
                self.translation.tts.start(
                    "I did not catch anything to translate. Please try again.",
                    timeout_s=self.tts_timeout,
                )
            return None

        self.logger.info("Transcribing translation audio...")
        stt_start = time.perf_counter()
        transcription = self.translation.transcribe(delete=self.translation.source_lang == "en")
        self._stt_times.append(time.perf_counter() - stt_start)
        if not transcription or not transcription.strip():
            self._translation_times.append(0.0)
            self._tts_times.append(0.0)
            self._cycle_times.append(time.perf_counter() - cycle_start)
            self._sample_resources()
            if self.translation.source_lang != "en":
                self.translation.stt.delete_last_audio_file()
            if self.translation.tts:
                self.translation.tts.start(
                    "I did not catch that. Please try again.",
                    timeout_s=self.tts_timeout,
                )
            return None

        stop_intent = self._get_stop_intent(transcription)
        if self.translation.source_lang != "en":
            english_transcription = self.translation.transcribe(language_override="en", delete=True)
            english_stop_intent = self._get_stop_intent(english_transcription)
            if english_stop_intent:
                stop_intent = english_stop_intent

        if stop_intent:
            self._translation_times.append(0.0)
            self._tts_times.append(0.0)
            self._cycle_times.append(time.perf_counter() - cycle_start)
            self._sample_resources()
            if self.translation.tts:
                if stop_intent == "stop_program":
                    self.translation.tts.start(
                        "Stopping. Goodbye.",
                        timeout_s=self.tts_timeout,
                    )
                else:
                    self.translation.tts.start(
                        "Stopping. Say the wake word when you need me again.",
                        timeout_s=self.tts_timeout,
                    )
            return stop_intent

        self.logger.info("Translating wake word transcription...")
        translation_start = time.perf_counter()
        translated = self.translation.translator.translate(
            transcription,
            self.translation.source_lang,
            self.translation.target_lang,
        )
        self._translation_times.append(time.perf_counter() - translation_start)

        tts_start = time.perf_counter()
        if self.translation.speak and translated and self.translation.tts:
            self.translation.tts.start(translated, timeout_s=self.tts_timeout)
        self._tts_times.append(time.perf_counter() - tts_start)

        self._successful_cycles += 1
        self._input_words.append(self._word_count(transcription))
        self._output_words.append(self._word_count(translated))
        self._input_chars.append(len(transcription))
        self._output_chars.append(len(translated))
        self._cycle_times.append(time.perf_counter() - cycle_start)
        self._sample_resources()

        print(
            f"➡️  Translated ({self.translation.source_lang} → {self.translation.target_lang}): {translated}"
        )
        return None

    def _with_processing_lock(self, fn: Callable[[], None]) -> None:
        """Avoid overlapping wake-word callbacks."""
        with self._processing_lock:
            if self._is_processing:
                self.logger.info("Already processing a request; ignoring new wake word.")
                return
            self._is_processing = True
        try:
            fn()
        except Exception:
            self.logger.exception("Unhandled error while processing wake word request.")
        finally:
            with self._processing_lock:
                self._is_processing = False

    def _on_wake_word_detected(self, wakeword: str, score: float) -> None:
        self.logger.info("Wake word '%s' detected (score: %.2f)", wakeword, score)
        self._with_processing_lock(self._handle_request)

    def _handle_request(self) -> None:
        """Capture user speech and route to translation (sign-language branch TBD)."""
        self.detector.stop()
        try:
            if self.prompt_user and self.translation.tts:
                self.logger.info("Prompting user before recording.")
                prompt_text = (
                    "What can I do for you? Say translate or detect to begin."
                    if not self.stay_awake
                    else "Ready. Say translate to translate or detect for sign language. Say stop listening to finish."
                )
                self.translation.tts.start(prompt_text, timeout_s=self.tts_timeout)
                self.logger.info("Prompt completed. Starting recording.")

            while True:
                time.sleep(0.1)  # let the prompt finish before capturing audio
                audio_path = self.translation.record(filename="last_rec.wav")
                if audio_path:
                    self.logger.info("Wake word audio captured: %s", audio_path)
                if not audio_path:
                    self.logger.warning("No audio captured after wake word.")
                    return

                self.logger.info("Transcribing wake word audio...")
                prompt = self.translation.transcribe(delete=self.translation.source_lang == "en")
                if not prompt or not prompt.strip():
                    if self.translation.source_lang != "en":
                        self.translation.stt.delete_last_audio_file()
                    if self.translation.tts:
                        self.translation.tts.start(
                            "I did not catch that. Please try again.",
                            timeout_s=self.tts_timeout,
                        )
                    return

                normalized = prompt.strip().lower()
                print(f"📝 Transcription result: {prompt}")
                self.logger.info("Command captured: %s", normalized)

                stop_intent = self._get_stop_intent(prompt)
                if stop_intent:
                    if self.translation.source_lang != "en":
                        self.translation.stt.delete_last_audio_file()
                    if self.translation.tts:
                        if stop_intent == "stop_program":
                            self.translation.tts.start(
                                "Stopping. Goodbye.",
                                timeout_s=self.tts_timeout,
                            )
                        else:
                            self.translation.tts.start(
                                "Stopping. Say the wake word when you need me again.",
                                timeout_s=self.tts_timeout,
                            )
                    if stop_intent == "stop_program":
                        self._should_exit = True
                    return
                command_for_mode = prompt
                if self.translation.source_lang != "en":
                    english_prompt = self.translation.transcribe(language_override="en", delete=True)
                    stop_intent = self._get_stop_intent(english_prompt)
                    if stop_intent:
                        if self.translation.tts:
                            if stop_intent == "stop_program":
                                self.translation.tts.start(
                                    "Stopping. Goodbye.",
                                    timeout_s=self.tts_timeout,
                                )
                            else:
                                self.translation.tts.start(
                                    "Stopping. Say the wake word when you need me again.",
                                    timeout_s=self.tts_timeout,
                                )
                        if stop_intent == "stop_program":
                            self._should_exit = True
                        return
                    if english_prompt:
                        command_for_mode = english_prompt

                mode_intent = self._get_mode_intent(command_for_mode)
                if not mode_intent:
                    if self.translation.tts:
                        self.translation.tts.start(
                            "Please say translate or detect.",
                            timeout_s=self.tts_timeout,
                        )
                    continue

                if mode_intent == "detect":
                    if self.translation.source_lang != "en":
                        self.translation.stt.delete_last_audio_file()
                    if self.translation.tts:
                        self.translation.tts.start(
                            "Sign language detection pipeline is not ready yet.",
                            timeout_s=self.tts_timeout,
                        )
                    return

                if self.translation.source_lang != "en":
                    self.translation.stt.delete_last_audio_file()
                translate_stop_intent = self._handle_translate_mode()
                if translate_stop_intent:
                    if translate_stop_intent == "stop_program":
                        self._should_exit = True
                    return
                if not self.stay_awake:
                    return
                if self.prompt_user and self.translation.tts:
                    self.translation.tts.start(
                        "Say translate or detect to continue, or say stop listening to finish.",
                        timeout_s=self.tts_timeout,
                    )
        finally:
            if not self._should_exit:
                self.detector.start()

    def run(self) -> None:
        """Start wake-word listening loop."""
        print(
            f"✅ Ready. Listening for wake words {self.wakeword_models} to start translation "
            f"({self.translation.source_lang} → {self.translation.target_lang})."
        )
        try:
            self.detector.start()
            while not self._should_exit:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            self.detector.stop()
            self.detector.cleanup()
            self.translation.recorder.cleanup()
            print(json.dumps(self._build_metrics(), indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Wake-word controlled speech translator (wake word -> STT -> translate -> TTS)."
    )
    parser.add_argument("--source-lang", default="en", help="Source language code for translation.")
    parser.add_argument("--target-lang", default="fr", help="Target language code for translation.")
    parser.add_argument(
        "--audio-dir",
        default="audio",
        help="Directory where microphone captures are stored.",
    )
    parser.add_argument(
        "--stt-model",
        help=(
            "Whisper STT model to use. Supported values: openai_whisper[:model] "
            "(e.g., openai_whisper:small). Defaults to openai_whisper:base."
        ),
    )
    parser.add_argument(
        "--wakeword",
        action="append",
        help="Wake word model(s) to load (defaults to openWakeWord's hey_jarvis).",
    )
    parser.add_argument(
        "--wake-threshold",
        type=float,
        default=0.25,
        help="Detection threshold for wake word activation.",
    )
    parser.add_argument(
        "--wake-mic-index",
        type=int,
        help="PyAudio input device index to use for wake word detection.",
    )
    parser.add_argument(
        "--wake-debug",
        action="store_true",
        help="Log wake word scores periodically for debugging.",
    )
    parser.add_argument(
        "--wake-debug-interval",
        type=float,
        default=1.0,
        help="Seconds between wake word score logs when --wake-debug is enabled.",
    )
    parser.add_argument(
        "--no-speak",
        action="store_true",
        help="Disable TTS playback of translations.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Skip the TTS prompt before recording after wake word detection.",
    )
    parser.add_argument(
        "--stay-awake",
        action="store_true",
        help="Keep listening for additional translations after a wake word until you say 'stop listening'.",
    )
    parser.add_argument(
        "--tts-timeout",
        type=float,
        help="Optional timeout (seconds) for TTS playback to avoid hangs.",
    )
    args = parser.parse_args()

    assistant = WakeWordTranslationAssistant(
        audio_dir=args.audio_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        stt_model=args.stt_model,
        wakeword_models=args.wakeword,
        wakeword_threshold=args.wake_threshold,
        wakeword_device_index=args.wake_mic_index,
        wakeword_debug=args.wake_debug,
        wakeword_debug_interval=args.wake_debug_interval,
        speak=not args.no_speak,
        prompt_user=not args.no_prompt,
        stay_awake=args.stay_awake,
        tts_timeout=args.tts_timeout,
    )
    assistant.run()


if __name__ == "__main__":
    main()
