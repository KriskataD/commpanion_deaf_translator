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
from .ocr.scan_once import OcrScanner


class WakeWordTranslationAssistant:
    """Coordinates wake-word detection with the translation pipeline."""

    def __init__(
        self,
        audio_dir: Path | str = "audio",
        source_lang: str = "en",
        target_lang: str = "fr",
        qnn_encoder_dir: Path | str = "models/whisper_small_quantized_encoder_optimized_onnx",
        qnn_decoder_dir: Path | str = "models/whisper_small_quantized_decoder_optimized_onnx",
        stt_model: str = "auto",
        wakeword_models: list[str] | None = None,
        wakeword_threshold: float = 0.25,
        wakeword_device_index: int | None = None,
        wakeword_debug: bool = False,
        wakeword_debug_interval: float = 1.0,
        speak: bool = True,
        prompt_user: bool = True,
        stay_awake: bool = False,
        stt_timeout: float | None = None,
        tts_timeout: float | None = None,
        launch_captions_overlay: bool = False,
        captions_monitor_index: int | None = None,
    ) -> None:
        WakeWordDetector.download_models()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.ocr_scanner = OcrScanner(
            detector_onnx="src/models/easyocr_detector_float_optimized_onnx/model.onnx",
            recognizer_onnx="src/models/easyocr_recognizer_float_optimized_onnx/model.onnx",
            camera_id=1,
        )

        self.translation = TranslatorPipeline(
            audio_dir=audio_dir,
            source_lang=source_lang,
            target_lang=target_lang,
            speak=speak,
            qnn_encoder_dir=qnn_encoder_dir,
            qnn_decoder_dir=qnn_decoder_dir,
            stt_model=stt_model,
            stt_timeout=stt_timeout,
            tts_timeout=tts_timeout,
            launch_captions_overlay=launch_captions_overlay,
            captions_monitor_index=captions_monitor_index,
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
        if self.translation.tts:
            self.translation.tts.start(
                "Ready to translate. Please say what you want translated.",
                timeout_s=self.tts_timeout,
            )
        time.sleep(0.1)
        audio_path = self.translation.record(filename="last_rec.wav")
        if audio_path:
            self.logger.info("Translation audio captured: %s", audio_path)
        if not audio_path:
            self.logger.warning("No audio captured for translation.")
            if self.translation.tts:
                self.translation.tts.start(
                    "I did not catch anything to translate. Please try again.",
                    timeout_s=self.tts_timeout,
                )
            return None

        self.logger.info("Transcribing translation audio...")
        try:
            transcription = self.translation.transcribe(delete=self.translation.source_lang == "en")
        except Exception:
            self.logger.exception("Translation transcription failed.")
            if self.translation.tts:
                self.translation.tts.start(
                    "I had trouble understanding that. Please try again.",
                    timeout_s=self.tts_timeout,
                )
            return None
        if not transcription or not transcription.strip():
            if self.translation.source_lang != "en":
                self.translation.delete_last_audio_file()
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
        self.translation.translate_transcription(transcription)
        return None

    def _handle_detect_mode(self):
        if self.translation.tts:
            self.translation.tts.start("Detecting text.", timeout_s=self.tts_timeout)

        try:
            text = self.ocr_scanner.scan_once(save_debug=True)
        except Exception as e:
            if self.translation.tts:
                self.translation.tts.start("Camera or OCR failed.", timeout_s=self.tts_timeout)
            print("Detect error:", e)
            return

        if not text:
            if self.translation.tts:
                self.translation.tts.start("No readable text found.", timeout_s=self.tts_timeout)
            return

        print("OCR text:\n", text)

        # Temporarily force OCR source language for demo (English)
        old_src = self.translation.source_lang
        try:
            self.translation.source_lang = "en"
            translated = self.translation.translate_text(text)  # will also send captions
        finally:
            self.translation.source_lang = old_src

        print("Translated:\n", translated)

        # Speak or show overlay
        if self.translation.tts:
            # keep short so it doesn't ramble
            self.translation.tts.start(translated[:220], timeout_s=self.tts_timeout)

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
                try:
                    prompt = self.translation.transcribe(
                        delete=self.translation.source_lang == "en",
                    )
                except Exception:
                    self.logger.exception("Wake word transcription failed.")
                    if self.translation.tts:
                        self.translation.tts.start(
                            "I had trouble understanding that. Please try again.",
                            timeout_s=self.tts_timeout,
                        )
                    return
                if not prompt or not prompt.strip():
                    if self.translation.source_lang != "en":
                        self.translation.delete_last_audio_file()
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
                        self.translation.delete_last_audio_file()
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
                        self.translation.delete_last_audio_file()
                    self._handle_detect_mode()
                    if not self.stay_awake:
                        return
                    if self.prompt_user and self.translation.tts:
                        self.translation.tts.start(
                            "Say translate or detect to continue, or say stop listening to finish.",
                            timeout_s=self.tts_timeout,
                        )
                    continue

                if self.translation.source_lang != "en":
                    self.translation.delete_last_audio_file()
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
            # Full program shutdown only
            self.translation.shutdown_captions_overlay()


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
        "--stt-timeout",
        type=float,
        help="Optional timeout (seconds) for STT transcription to avoid hangs.",
    )
    parser.add_argument(
        "--tts-timeout",
        type=float,
        help="Optional timeout (seconds) for TTS playback to avoid hangs.",
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

    assistant = WakeWordTranslationAssistant(
        audio_dir=args.audio_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        qnn_encoder_dir=args.qnn_encoder_dir,
        qnn_decoder_dir=args.qnn_decoder_dir,
        stt_model=args.stt_model,
        wakeword_models=args.wakeword,
        wakeword_threshold=args.wake_threshold,
        wakeword_device_index=args.wake_mic_index,
        wakeword_debug=args.wake_debug,
        wakeword_debug_interval=args.wake_debug_interval,
        speak=not args.no_speak,
        prompt_user=not args.no_prompt,
        stay_awake=args.stay_awake,
        stt_timeout=args.stt_timeout,
        tts_timeout=args.tts_timeout,
        launch_captions_overlay=args.captions_auto_start,
        captions_monitor_index=args.captions_monitor_index,
    )
    assistant.run()


if __name__ == "__main__":
    main()
