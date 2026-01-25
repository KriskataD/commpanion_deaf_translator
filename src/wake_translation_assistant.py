"""
Wake-word driven pipeline: listen for wake word, record a command,
run STT -> translate -> TTS. A placeholder branch is kept for future
sign language detection.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Callable

from .translator import TranslatorPipeline
from .wakeword_detector import WakeWordDetector


class WakeWordTranslationAssistant:
    """Coordinates wake-word detection with the translation pipeline."""

    def __init__(
        self,
        audio_dir: Path | str = "audio",
        source_lang: str = "en",
        target_lang: str = "fr",
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
                    "What can I do for you? Say translate to begin."
                    if not self.stay_awake
                    else "Ready. Say what you want translated. Say stop listening to finish."
                )
                self.translation.tts.start(prompt_text, timeout_s=self.tts_timeout)
                self.logger.info("Prompt completed. Starting recording.")

            stop_phrases = {
                "stop listening",
                "stop",
                "exit",
                "quit",
                "cancel",
            }
            while True:
                time.sleep(0.1)  # let the prompt finish before capturing audio
                audio_path = self.translation.record(filename="last_rec.wav")
                if audio_path:
                    self.logger.info("Wake word audio captured: %s", audio_path)
                if not audio_path:
                    self.logger.warning("No audio captured after wake word.")
                    return

                self.logger.info("Transcribing wake word audio...")
                prompt = self.translation.transcribe()
                if not prompt or not prompt.strip():
                    if self.translation.tts:
                        self.translation.tts.start(
                            "I did not catch that. Please try again.",
                            timeout_s=self.tts_timeout,
                        )
                    return

                normalized = prompt.strip().lower()
                self.logger.info("Command captured: %s", normalized)

                if "sign language" in normalized or "signing" in normalized:
                    if self.translation.tts:
                        self.translation.tts.start(
                            "Sign language detection pipeline is not ready yet.",
                            timeout_s=self.tts_timeout,
                        )
                    return

                if normalized in stop_phrases or "stop listening" in normalized:
                    if self.translation.tts:
                        self.translation.tts.start(
                            "Stopping. Say the wake word when you need me again.",
                            timeout_s=self.tts_timeout,
                        )
                    return

                # Default path: translate from configured source->target languages.
                self.logger.info("Translating wake word transcription...")
                self.translation.translate_transcription(prompt)

                if not self.stay_awake:
                    return
                if self.prompt_user and self.translation.tts:
                    self.translation.tts.start(
                        "Say another phrase or say stop listening to finish.",
                        timeout_s=self.tts_timeout,
                    )
        finally:
            self.detector.start()

    def run(self) -> None:
        """Start wake-word listening loop."""
        print(
            f"✅ Ready. Listening for wake words {self.wakeword_models} to start translation "
            f"({self.translation.source_lang} → {self.translation.target_lang})."
        )
        try:
            self.detector.start()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            self.detector.stop()
            self.detector.cleanup()
            self.translation.recorder.cleanup()


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
