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
import textwrap

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
        self._ocr_pages: list[str] = []
        self._ocr_page_index = 0
        self._ocr_original_text = ""
        self._ocr_display_text = ""
        self._ocr_is_translated = False
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
    def _pretty_wakeword_label(wakeword: str) -> str:
        return Path(wakeword).stem.replace("_", " ").replace("-", " ").strip().title()

    def _show_idle_wake_caption(self) -> None:
        if not self.wakeword_models:
            wake_phrase = '"Hey Jarvis"'
        else:
            wake_phrase = " or ".join(
                f'"{self._pretty_wakeword_label(w)}"' for w in self.wakeword_models
            )

        self.translation.show_caption(
            f"Say {wake_phrase} to wake up the pipeline.",
            ttl_ms=None,   # persistent until replaced/cleared
        )

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
    
    @staticmethod
    def _chunk_text_for_ocr_display(
        text: str,
        *,
        max_chars_per_line: int = 48,
        max_lines: int = 2,
    ) -> list[str]:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return []

        lines = textwrap.wrap(
            cleaned,
            width=max_chars_per_line,
            break_long_words=False,
            break_on_hyphens=False,
        )

        return [
            "\n".join(lines[i:i + max_lines])
            for i in range(0, len(lines), max_lines)
        ]

    @staticmethod
    def _get_ocr_command(transcription: str) -> str | None:
        normalized = " ".join((transcription or "").lower().split())

        if "stop reading" in normalized or normalized == "stop":
            return "stop"

        if "previous page" in normalized or normalized == "previous" or normalized == "back":
            return "previous"

        if "next page" in normalized or normalized == "next":
            return "next"

        if "translate text" in normalized or normalized == "translate":
            return "translate"

        return None


    def _set_ocr_display_text(self, text: str, *, translated: bool) -> None:
        self._ocr_display_text = text or ""
        self._ocr_pages = self._chunk_text_for_ocr_display(
            self._ocr_display_text,
            max_chars_per_line=48,
            max_lines=2,
        )
        self._ocr_page_index = 0
        self._ocr_is_translated = translated


    def _show_current_ocr_page(self) -> None:
        if not self._ocr_pages:
            return

        self.translation.show_caption(
            self._ocr_pages[self._ocr_page_index],
            ttl_ms=None,
            format_text=False,
        )


    def _translate_ocr_text(self, ocr_cycle: dict[str, object] | None = None) -> None:
        if not self._ocr_original_text.strip():
            return

        translate_start = time.perf_counter()
        try:
            translated = self.translation.translator.translate(
                self._ocr_original_text,
                source_lang="en",   # keep your current OCR demo assumption
                target_lang=self.translation.target_lang,
            )
            elapsed = time.perf_counter() - translate_start
        except Exception:
            elapsed = time.perf_counter() - translate_start
            if ocr_cycle is not None:
                ocr_cycle["error_stage"] = "ocr_translate"
                self.translation.performance.record_ocr_translation(ocr_cycle, "", elapsed)
            raise

        if ocr_cycle is not None:
            self.translation.performance.record_ocr_translation(ocr_cycle, translated, elapsed)

        if not translated or not translated.strip():
            if self.translation.tts:
                self.translation.speak_text(
                    "Translation failed.",
                    timeout_s=self.tts_timeout,
                    ttl_ms=2000,
                    show_caption=True,
                )
                self._show_current_ocr_page()
            return

        self._set_ocr_display_text(translated, translated=True)
        self._show_current_ocr_page()

    def _handle_translate_mode(self) -> str | None:
        if self.translation.tts:
            self.translation.speak_text(
                "Ready to translate. Please say what you want translated.",
                timeout_s=self.tts_timeout,
                show_caption=True,
            )
        cycle = self.translation.performance.start_speech_cycle()
        transcription = ""
        translated = ""
        time.sleep(0.1)

        record_start = time.perf_counter()
        audio_path = self.translation.record(filename="last_rec.wav")
        cycle["record_time_s"] = time.perf_counter() - record_start
        if audio_path:
            self.logger.info("Translation audio captured: %s", audio_path)
        if not audio_path:
            self.logger.warning("No audio captured for translation.")
            self.translation.performance.complete_speech_cycle(
                cycle,
                success=False,
                error_stage="record",
                transcription=transcription,
                translated=translated,
            )
            if self.translation.tts:
                self.translation.speak_text(
                    "I did not catch anything to translate. Please try again.",
                    timeout_s=self.tts_timeout,
                    show_caption=True,
                )
            return None

        self.logger.info("Transcribing translation audio...")
        stt_start = time.perf_counter()
        try:
            transcription = self.translation.transcribe(delete=self.translation.source_lang == "en")
            cycle["stt_time_s"] = time.perf_counter() - stt_start
        except Exception:
            cycle["stt_time_s"] = time.perf_counter() - stt_start
            self.logger.exception("Translation transcription failed.")
            self.translation.performance.complete_speech_cycle(
                cycle,
                success=False,
                error_stage="stt",
                transcription=transcription,
                translated=translated,
            )
            if self.translation.tts:
                self.translation.speak_text(
                    "I had trouble understanding that. Please try again.",
                    timeout_s=self.tts_timeout,
                    show_caption=True,
                )
            return None
        if not transcription or not transcription.strip():
            self.translation.performance.complete_speech_cycle(
                cycle,
                success=False,
                error_stage="stt",
                transcription=transcription,
                translated=translated,
            )
            if self.translation.source_lang != "en":
                self.translation.delete_last_audio_file()
            if self.translation.tts:
                self.translation.speak_text(
                    "I did not catch that. Please try again.",
                    timeout_s=self.tts_timeout,
                    show_caption=True,
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
                    self.translation.speak_text(
                        "Stopping. Goodbye.",
                        timeout_s=self.tts_timeout,
                        show_caption=True,
                    )
                else:
                    self.translation.speak_text(
                        "Stopping. Say the wake word when you need me again.",
                        timeout_s=self.tts_timeout,
                        show_caption=True,
                    )
            return stop_intent

        self.logger.info("Translating wake word transcription...")
        translation_start = time.perf_counter()
        try:
            translated = self.translation.translate_transcription(transcription, skip_tts=True)
            cycle["translation_time_s"] = time.perf_counter() - translation_start
        except Exception:
            cycle["translation_time_s"] = time.perf_counter() - translation_start
            self.logger.exception("Translation failed.")
            self.translation.performance.complete_speech_cycle(
                cycle,
                success=False,
                error_stage="translate",
                transcription=transcription,
                translated=translated,
            )
            return None

        if self.translation.speak and self.translation.tts:
            tts_start = time.perf_counter()
            try:
                self.translation.tts.start(translated, timeout_s=self.translation.tts_timeout)
                cycle["tts_time_s"] = time.perf_counter() - tts_start
            except Exception:
                cycle["tts_time_s"] = time.perf_counter() - tts_start
                self.logger.exception("TTS failed.")
                self.translation.performance.complete_speech_cycle(
                    cycle,
                    success=False,
                    error_stage="tts",
                    transcription=transcription,
                    translated=translated,
                )
                return None
        else:
            cycle["tts_time_s"] = 0.0

        self.translation.performance.complete_speech_cycle(
            cycle,
            success=True,
            error_stage=None,
            transcription=transcription,
            translated=translated,
        )
        return None

    def _handle_detect_mode(self):
        ocr_cycle = self.translation.performance.start_ocr_cycle()
        error_stage: str | None = None

        if self.translation.tts:
            self.translation.speak_text("Detecting text.", timeout_s=self.tts_timeout, show_caption=True)

        scan_start = time.perf_counter()
        try:
            text = self.ocr_scanner.scan_once(save_debug=True)
            ocr_cycle["ocr_scan_time_s"] = time.perf_counter() - scan_start
        except Exception as e:
            ocr_cycle["ocr_scan_time_s"] = time.perf_counter() - scan_start
            error_stage = "ocr_scan"
            if self.translation.tts:
                self.translation.speak_text("Camera or OCR failed.", timeout_s=self.tts_timeout, show_caption=True)
            print("Detect error:", e)
            self.translation.performance.complete_ocr_cycle(ocr_cycle, success=False, error_stage=error_stage)
            return

        cleaned = (text or "").strip()
        ocr_cycle["ocr_text_found"] = bool(cleaned)
        ocr_cycle["ocr_chars"] = len(cleaned)
        ocr_cycle["ocr_words"] = len(cleaned.split()) if cleaned else 0

        if not cleaned:
            if self.translation.tts:
                self.translation.speak_text("No readable text found.", timeout_s=self.tts_timeout, show_caption=True)
            self.translation.performance.complete_ocr_cycle(ocr_cycle, success=False, error_stage=None)
            return

        print("OCR text:\n", text)

        # Store original OCR text and show it first, without translating
        self._ocr_original_text = text
        self._set_ocr_display_text(text, translated=False)
        ocr_cycle["ocr_pages"] = len(self._ocr_pages)

        if not self._ocr_pages:
            error_stage = "ocr_display"
            if self.translation.tts:
                self.translation.speak_text("Nothing to show.", timeout_s=self.tts_timeout, show_caption=True)
            self.translation.performance.complete_ocr_cycle(ocr_cycle, success=False, error_stage=error_stage)
            return

        if self.translation.tts:
            self.translation.speak_text(
                "Reading mode. Say next page, previous page, translate text, or stop reading.",
                timeout_s=self.tts_timeout,
                show_caption=False,
            )
            self.translation.show_caption(
                "Reading mode. Say next page, previous page, translate text, or stop reading.",
                ttl_ms=3500,
            )
            time.sleep(3.5)

        try:
            self._show_current_ocr_page()
        except Exception:
            self.logger.exception("Failed to show OCR page.")
            self.translation.performance.complete_ocr_cycle(
                ocr_cycle,
                success=False,
                error_stage="ocr_display",
            )
            return

        self._run_ocr_reading_loop(ocr_cycle)
        final_error_stage = error_stage or ocr_cycle.get("error_stage")
        self.translation.performance.complete_ocr_cycle(
            ocr_cycle,
            success=True,
            error_stage=final_error_stage if isinstance(final_error_stage, str) else None,
        )

    def _run_ocr_reading_loop(self, ocr_cycle: dict[str, object]) -> None:
        while True:
            record_start = time.perf_counter()
            audio_path = self.translation.record(filename="ocr_command.wav")
            record_time = time.perf_counter() - record_start
            if not audio_path:
                continue

            stt_start = time.perf_counter()
            try:
                command_text = self.translation.transcribe(
                    language_override="en",
                    delete=True,
                )
                stt_time = time.perf_counter() - stt_start
            except Exception:
                stt_time = time.perf_counter() - stt_start
                self.logger.exception("OCR command transcription failed.")
                ocr_cycle["error_stage"] = ocr_cycle.get("error_stage") or "ocr_command_stt"
                self.translation.performance.record_ocr_command(
                    ocr_cycle,
                    record_time_s=record_time,
                    stt_time_s=stt_time,
                    intent=None,
                )
                if self.translation.tts:
                    self.translation.speak_text(
                        "Say next page, previous page, translate text, or stop reading.",
                        timeout_s=self.tts_timeout,
                        ttl_ms=2500,
                        show_caption=True,
                    )
                    self._show_current_ocr_page()
                continue

            print(f"OCR command: {command_text}")
            intent = self._get_ocr_command(command_text)
            self.translation.performance.record_ocr_command(
                ocr_cycle,
                record_time_s=record_time,
                stt_time_s=stt_time,
                intent=intent,
            )

            if intent == "next":
                if self._ocr_page_index < len(self._ocr_pages) - 1:
                    self._ocr_page_index += 1
                self._show_current_ocr_page()
                continue

            if intent == "previous":
                if self._ocr_page_index > 0:
                    self._ocr_page_index -= 1
                self._show_current_ocr_page()
                continue

            if intent == "translate":
                if not self._ocr_is_translated:
                    try:
                        self._translate_ocr_text(ocr_cycle)
                    except Exception:
                        self.logger.exception("OCR text translation failed.")
                        ocr_cycle["error_stage"] = "ocr_translate"
                        if self.translation.tts:
                            self.translation.speak_text(
                                "Translation failed.",
                                timeout_s=self.tts_timeout,
                                ttl_ms=2000,
                                show_caption=True,
                            )
                            self._show_current_ocr_page()
                else:
                    self._show_current_ocr_page()
                continue

            if intent == "stop":
                self._ocr_original_text = ""
                self._ocr_display_text = ""
                self._ocr_pages = []
                self._ocr_page_index = 0
                self._ocr_is_translated = False
                self.translation.clear_captions()
                if self.translation.tts:
                    self.translation.speak_text(
                        "Stopped reading.",
                        timeout_s=self.tts_timeout,
                        ttl_ms=1500,
                        show_caption=True,
                    )
                return

            if self.translation.tts:
                self.translation.speak_text(
                    "Say next page, previous page, translate text, or stop reading.",
                    timeout_s=self.tts_timeout,
                    ttl_ms=2500,
                    show_caption=True,
                )
                self._show_current_ocr_page()

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
        self.translation.clear_captions()
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
                    else "Ready. Say 'translate' for speech translation or 'detect' for text detection. Say stop listening to finish."
                )
                self.translation.speak_text(prompt_text, timeout_s=self.tts_timeout, show_caption=True)
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
                        self.translation.speak_text(
                            "I had trouble understanding that. Please try again.",
                            timeout_s=self.tts_timeout,
                            show_caption=True,
                        )
                    return
                if not prompt or not prompt.strip():
                    if self.translation.source_lang != "en":
                        self.translation.delete_last_audio_file()
                    if self.translation.tts:
                        self.translation.speak_text(
                            "I did not catch that. Please try again.",
                            timeout_s=self.tts_timeout,
                            show_caption=True,
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
                            self.translation.speak_text(
                                "Stopping. Goodbye.",
                                timeout_s=self.tts_timeout,
                                show_caption=True,
                            )
                        else:
                            self.translation.speak_text(
                                "Stopping. Say the wake word when you need me again.",
                                timeout_s=self.tts_timeout,
                                show_caption=True,
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
                                self.translation.speak_text(
                                    "Stopping. Goodbye.",
                                    timeout_s=self.tts_timeout,
                                    show_caption=True,
                                )
                            else:
                                self.translation.speak_text(
                                    "Stopping. Say the wake word when you need me again.",
                                    timeout_s=self.tts_timeout,
                                    show_caption=True,
                                )
                        if stop_intent == "stop_program":
                            self._should_exit = True
                        return
                    if english_prompt:
                        command_for_mode = english_prompt

                mode_intent = self._get_mode_intent(command_for_mode)
                if not mode_intent:
                    if self.translation.tts:
                        self.translation.speak_text(
                            "Please say translate or detect.",
                            timeout_s=self.tts_timeout,
                            show_caption=True,
                        )
                    continue

                if mode_intent == "detect":
                    if self.translation.source_lang != "en":
                        self.translation.delete_last_audio_file()
                    self._handle_detect_mode()
                    if not self.stay_awake:
                        return
                    if self.prompt_user and self.translation.tts:
                        self.translation.speak_text(
                            "Say translate or detect to continue, or say stop listening to finish.",
                            timeout_s=self.tts_timeout,
                            show_caption=False,
                        )
                    self.translation.show_caption(
                        "Say translate or detect to continue, or say stop listening to finish.",
                        ttl_ms=None,
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
                    self.translation.speak_text(
                        "Say translate or detect to continue, or say stop listening to finish.",
                        timeout_s=self.tts_timeout,
                        show_caption=False,
                    )
                self.translation.show_caption(
                    "Say translate or detect to continue, or say stop listening to finish.",
                    ttl_ms=None,
                )
        finally:
            if not self._should_exit:
                self.detector.start()
                if not self.stay_awake:
                    threading.Timer(10.0, self._show_idle_wake_caption).start()

    def run(self) -> None:
        """Start wake-word listening loop."""
        print(
            f"✅ Ready. Listening for wake words {self.wakeword_models} to start translation "
            f"({self.translation.source_lang} → {self.translation.target_lang})."
        )
        try:
            self.detector.start()
            self._show_idle_wake_caption()
            while not self._should_exit:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            self.detector.stop()
            self.detector.cleanup()
            self.translation.recorder.cleanup()
            self.translation.print_performance_summary()
            summary_latest, speech_latest, ocr_latest, _, _, _ = self.translation.save_performance_reports()
            print("Saved:")
            print(f"  {summary_latest}")
            print(f"  {speech_latest}")
            print(f"  {ocr_latest}")
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
