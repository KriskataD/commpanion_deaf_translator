"""
Translator pipeline using QNN Whisper STT, M2M100 translation, and optional TTS playback.

Speak into the microphone, the audio is recorded until silence is detected,
transcribed, translated, and the translated text is printed and spoken.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import threading
import time
import os
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from .npu.whisper_qnn_stt import WhisperSmallQuantizedQNNSTT
from .recorder import AudioRecorder
from .tts import _TTS


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
        stt_timeout: float | None = None,
        tts_timeout: float | None = None,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.speak = speak
        self.stt_timeout = stt_timeout
        self.tts_timeout = tts_timeout

        self.logger = logging.getLogger(__name__)
        self.recorder = AudioRecorder()
        self.stt = self._build_stt_backend(qnn_encoder_dir, qnn_decoder_dir)
        self.translator = MultiLanguageTranslator()
        self.tts = _TTS() if self.speak else None
        self.last_audio_path: Path | None = None

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
        return WhisperSmallQuantizedQNNSTT(
            encoder_dir=encoder_dir,
            decoder_dir=decoder_dir,
            debug=False,
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

    def delete_last_audio_file(self) -> None:
        if self.last_audio_path and self.last_audio_path.exists():
            self.last_audio_path.unlink()
        self.last_audio_path = None

    def delete_last_audio_file(self) -> None:
        if self.last_audio_path and self.last_audio_path.exists():
            self.last_audio_path.unlink()
        self.last_audio_path = None

    def set_languages(self, source_lang: str, target_lang: str) -> None:
        """Update the language pair for subsequent translations."""
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate_transcription(self, transcription: str) -> str:
        translated = self.translator.translate(transcription, self.source_lang, self.target_lang)
        if self.speak and translated and self.tts:
            self.tts.start(translated, timeout_s=self.tts_timeout)
        print(f"➡️  Translated ({self.source_lang} → {self.target_lang}): {translated}")
        return translated

    def translate_text(self, text: str) -> str:
        """Translate arbitrary text without invoking the STT step."""

        translated = self.translator.translate(text, self.source_lang, self.target_lang)
        if self.speak and translated and self.tts:
            self.tts.start(translated, timeout_s=self.tts_timeout)
        print(f"➡️  Translated text ({self.source_lang} → {self.target_lang}): {translated}")
        return translated

    def run_once(self) -> str:
        """Record audio once and return the translated text."""
        audio_path = self.record()
        if not audio_path:
            print("❌ Failed to capture audio.")
            return ""

        print("📝 Transcribing...")
        try:
            transcription = self.transcribe()
        except Exception as exc:
            self.logger.exception("Transcription failed.")
            print(f"❌ Transcription failed: {exc}")
            return ""
        print(f"Original text ({self.source_lang}): {transcription}")

        print("🌐 Translating...")
        try:
            return self.translate_transcription(transcription)
        except Exception as exc:
            self.logger.exception("Translation failed.")
            print(f"❌ Translation failed: {exc}")
            return ""

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

    args = parser.parse_args()

    pipeline = TranslatorPipeline(
        audio_dir=args.audio_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        speak=not args.no_speak,
        qnn_encoder_dir=args.qnn_encoder_dir,
        qnn_decoder_dir=args.qnn_decoder_dir,
        stt_timeout=args.stt_timeout,
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
