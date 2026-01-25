import os
import re
import threading
import time
from pathlib import Path
from queue import Queue

import pyttsx3

CHUNK_REGEX = re.compile(r".*?[\.!?…](?:\s|$)")  # Regex to match complete sentence-like segments

class _TTS:
    """
    A wrapper around the pyttsx3 engine to convert text to speech, used to vocalize complete sentences.
    """

    def __init__(self, rate: int = 200):
        """
        Initialize the pyttsx3 text-to-speech engine.

        Args:
            rate (int): Speed of speech in words per minute. Default is 200.
        """
        self._rate = rate
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
        print(f"🎤 Vocal synthesis: {text_}")
        done = threading.Event()
        self._queue.put((text_, timeout_s, done))
        done.wait()

    def _run_worker(self) -> None:
        while True:
            text_, timeout_s, done = self._queue.get()
            timer: threading.Timer | None = None
            try:
                if timeout_s is not None:
                    self._initialize_engine()
                    timer = threading.Timer(timeout_s, self._stop_engine_on_timeout, args=(timeout_s,))
                    timer.start()
                self.engine.say(text_)
                self.engine.runAndWait()
            except Exception:
                self._initialize_engine()
            finally:
                if timer:
                    timer.cancel()
                if timeout_s is not None:
                    self._initialize_engine()
                done.set()

    def _stop_engine_on_timeout(self, timeout_s: float) -> None:
        print(f"⚠️ TTS timeout after {timeout_s:.1f}s; stopping playback.")
        self.engine.stop()

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
