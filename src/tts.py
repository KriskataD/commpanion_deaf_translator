import re
import time
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
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)

    def start(self, text_: str):
        """
        Speak the given text out loud.

        Args:
            text_ (str): The sentence or phrase to be vocalized.
        """
        print(f"🎤 Vocal synthesis: {text_}")
        self.engine.say(text_)
        self.engine.runAndWait()


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