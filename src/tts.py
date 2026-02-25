import os
import re
import threading
import time
from pathlib import Path
from queue import Queue

import pyttsx3

try:
    import pythoncom
    import win32com.client as win32_client
except ImportError:  # pragma: no cover - only needed on Windows
    pythoncom = None
    win32_client = None

CHUNK_REGEX = re.compile(r".*?[\.!?…](?:\s|$)")  # Regex to match complete sentence-like segments

import json, socket

class CaptionClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 37777) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def set_text(self, text: str) -> None:
        try:
            payload = {"text": text, "clear": False}
            self.sock.sendto(json.dumps(payload, ensure_ascii=False).encode("utf-8"), self.addr)
        except Exception:
            # Never break TTS if captions fail
            pass

    def clear(self) -> None:
        try:
            payload = {"text": "", "clear": True}
            self.sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)
        except Exception:
            pass

class _TTS:
    """
    A wrapper around the pyttsx3 engine to convert text to speech, used to vocalize complete sentences.
    """

    def __init__(self, rate: int = 1, enable_captions: bool = True, captions_port: int = 37777):
        """
        Initialize the pyttsx3 text-to-speech engine.

        Args:
            rate (int): Speed of speech in words per minute. Default is 170.
        """
        self._rate = rate
        self._use_sapi = os.name == "nt" and win32_client is not None and pythoncom is not None
        self._sapi_voice = None
        if not self._use_sapi:
            self._initialize_engine()
        self._queue: Queue[tuple[str, float | None, threading.Event]] = Queue()
        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()
        self._captions = CaptionClient(port=captions_port) if enable_captions else None

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
        print(f"🎤 Vocal synthesis: {text_}")
        done = threading.Event()
        self._queue.put((text_, timeout_s, done))
        done.wait()

    def _get_sapi_voice(self):
        if self._sapi_voice is None:
            pythoncom.CoInitialize()
            voice = win32_client.Dispatch("SAPI.SpVoice")
            voice.Rate = self._rate
            voice.Volume = 100
            self._sapi_voice = voice
        return self._sapi_voice

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
                if self._captions is not None:
                    self._captions.set_text(text_)

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
                            print(f"⚠️ TTS timeout after {timeout_s:.1f}s; stopping playback.")
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

    print(f"\n\nResponse time: {time.time() - start_time:.3f} seconds")
    return full_text
