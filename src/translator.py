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

import subprocess
import sys

from .npu.whisper_qnn_stt import WhisperQnnSTT
from .recorder import AudioRecorder
from .tts import _TTS
from .captions_client import CaptionsClient

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
        self.tts = _TTS() if self.speak else None

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

    def show_caption(self, text: str, ttl_ms: int | None = 9000) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        try:
            if getattr(self, "captions", None):
                self.captions.send(cleaned, ttl_ms=ttl_ms)
        except Exception as e:
            self.logger.warning("Failed to send captions: %s", e)

    def speak_text(
        self,
        text: str,
        *,
        timeout_s: float | None = None,
        ttl_ms: int | None = 9000,
    ) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return

        # Always mirror spoken text to the captions overlay
        self.show_caption(cleaned, ttl_ms=ttl_ms)

        if self.speak and self.tts:
            self.tts.start(cleaned, timeout_s=timeout_s)

    def translate_transcription(self, transcription: str) -> str:
        translated = self.translator.translate(transcription, self.source_lang, self.target_lang)
        if translated:
            self.speak_text(translated, timeout_s=self.tts_timeout, ttl_ms=9000)
        print(f"➡️  Translated ({self.source_lang} → {self.target_lang}): {translated}")
        return translated

    def translate_text(self, text: str) -> str:
        """Translate arbitrary text without invoking the STT step."""

        translated = self.translator.translate(text, self.source_lang, self.target_lang)
        if translated:
            self.speak_text(translated, timeout_s=self.tts_timeout, ttl_ms=9000)
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
            transcription = self.transcribe(language_override=self.source_lang, delete=False)
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
