import os
import re
import threading
import time
import logging
from pathlib import Path
from queue import Queue

import pyttsx3

try:
    import pythoncom
    import win32com.client as win32_client
except ImportError:  # pragma: no cover - only needed on Windows
    pythoncom = None
    win32_client = None

CHUNK_REGEX = re.compile(r".*?[\.!?…](?:\s|$)")

logger = logging.getLogger(__name__)


class _TTS:
    """
    A wrapper around the pyttsx3 engine to convert text to speech, used to vocalize complete sentences.
    """

    def __init__(self, rate: int = 1, preferred_lang: str | None = None):
        """
        Initialize the pyttsx3 text-to-speech engine.

        Args:
            rate (int): Speed of speech in words per minute. Default is 170.
        """
        self._rate = rate
        self._preferred_lang = (preferred_lang or "").lower() or None
        self._logger = logging.getLogger(__name__)
        self._use_sapi = os.name == "nt" and win32_client is not None and pythoncom is not None
        self._sapi_voice = None
        if not self._use_sapi:
            self._initialize_engine()
        self._queue: Queue[tuple[str, float | None, threading.Event]] = Queue()
        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()

    def _initialize_engine(self) -> None:
        if os.name == "nt":
            _ensure_comtypes_cache()

        try:
            self.engine = pyttsx3.init()
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize text-to-speech. If you're on Windows, ensure "
                "`comtypes` and `pywin32` are installed and that the comtypes cache is "
                "writable. Try reinstalling with `pip install --upgrade comtypes pywin32`."
            ) from exc

        self.engine.setProperty("rate", self._rate)


    def start(self, text_: str, timeout_s: float | None = None):
        """
        Speak the given text out loud.

        Args:
            text_ (str): The sentence or phrase to be vocalized.
            timeout_s (float | None): Optional timeout for speech playback in seconds.
        """
        logger.info("TTS speaking: %s", text_)
        done = threading.Event()
        self._queue.put((text_, timeout_s, done))
        done.wait()

    def _get_sapi_voice(self):
        if self._sapi_voice is None:
            pythoncom.CoInitialize()
            voice = win32_client.Dispatch("SAPI.SpVoice")
            voice.Rate = self._rate
            voice.Volume = 100
            self._select_sapi_voice_for_language(voice)
            self._sapi_voice = voice
        return self._sapi_voice

    def set_language(self, lang_code: str | None) -> None:
        self._preferred_lang = (lang_code or "").lower() or None
        if self._use_sapi and self._sapi_voice is not None:
            self._select_sapi_voice_for_language(self._sapi_voice)

    def _select_sapi_voice_for_language(self, voice) -> None:
        if self._preferred_lang != "bg":
            return

        try:
            voices = voice.GetVoices()
        except Exception:
            self._logger.info("Unable to enumerate SAPI voices for Bulgarian selection.")
            return

        selected_voice = None
        for i in range(voices.Count):
            token = voices.Item(i)
            description = (token.GetDescription() or "").lower()
            name = (token.GetAttribute("Name") or "").lower()
            languages = (token.GetAttribute("Language") or "").lower()
            if (
                "bulgarian" in description
                or "bulgarian" in name
                or "bg-bg" in description
                or "bg-bg" in name
                or "0402" in languages
                or any(part.strip() == "402" for part in languages.replace(",", ";").split(";"))
            ):
                selected_voice = token
                break

        if selected_voice is None:
            self._logger.info("No Bulgarian SAPI voice found; using default system voice.")
            return

        try:
            voice.Voice = selected_voice
        except Exception:
            self._logger.info("Failed to select Bulgarian SAPI voice; using default system voice.")

    def _sapi_speak(self, text_: str, timeout_s: float | None) -> None:
        voice = self._get_sapi_voice()
        if timeout_s is None:
            voice.Speak(text_)
            return

        flags_async = 1  # SpeechVoiceSpeakFlags.SVSFlagsAsync
        flags_purge = 2  # SpeechVoiceSpeakFlags.SVSFPurgeBeforeSpeak
        voice.Speak(text_, flags_async)
        finished = voice.WaitUntilDone(int(timeout_s * 1000))
        if finished:
            return

        print(f"⚠️ TTS timeout after {timeout_s:.1f}s; stopping playback.")
        voice.Speak("", flags_async | flags_purge)

    def _run_worker(self) -> None:
        while True:
            text_, timeout_s, done = self._queue.get()
            try:
                if self._use_sapi:
                    self._sapi_speak(text_, timeout_s)
                else:
                    if timeout_s is not None:
                        self._initialize_engine()
                    self.engine.say(text_)
                    playback = threading.Thread(target=self.engine.runAndWait)
                    playback.daemon = True
                    playback.start()
                    if timeout_s is not None:
                        playback.join(timeout=timeout_s)
                        if playback.is_alive():
                            logger.warning("TTS timeout after %.1fs; stopping playback.", timeout_s)
                            self.engine.stop()
                    else:
                        playback.join()
            except Exception:
                if not self._use_sapi:
                    self._initialize_engine()
            finally:
                if timeout_s is not None and not self._use_sapi:
                    self._initialize_engine()
                done.set()

def _ensure_comtypes_cache() -> None:
    if os.environ.get("COMTYPES_GEN_DIR"):
        return

    base_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    gen_dir = base_dir / "comtypes_gen"
    gen_dir.mkdir(parents=True, exist_ok=True)
    init_file = gen_dir / "__init__.py"
    init_file.touch(exist_ok=True)
    os.environ["COMTYPES_GEN_DIR"] = str(gen_dir)


def talk_stream(stream) -> str:
    """
    Process a streaming LLM response and speak it sentence by sentence as soon as each is complete.

    Args:
        stream (iterable): An iterable or generator yielding response chunks.
                           Each chunk must be a string or an object with a `.content` attribute.

    Returns:
        str: The full concatenated response text that was spoken.
    """
    buffer = ""
    full_text = ""
    start_time = time.time()

    tts = _TTS()

    for chunk in stream:
        # Get text content from chunk (object or plain string)
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        if not content:
            continue

        buffer += content
        full_text += content

        # Speak all full sentences found in the buffer
        while (match := CHUNK_REGEX.match(buffer)):
            sentence = match.group(0).strip()
            if sentence:
                tts.start(sentence)
            buffer = buffer[match.end():]

    # Speak any remaining content in the buffer
    leftover = buffer.strip()
    if leftover:
        tts.start(leftover)

    logger.info("Response time: %.3f seconds", time.time() - start_time)
    return full_text
