from pathlib import Path

import importlib.util

import torch


class SpeechToTextApplication:
    """Speech-to-text using the OpenAI Whisper open-source model. """

    def __init__(
        self,
        audio_records_path: Path | str | None = None,
        model_name: str = "base",
        device: str | None = None,
        language: str | None = None,
        task: str = "transcribe",
    ) -> None:
        if not is_openai_whisper_available():
            raise RuntimeError(
                "openai-whisper is not installed. Install it with `pip install openai-whisper`."
            )
        if isinstance(audio_records_path, str):
            self.audio_records_path: Path | None = Path(audio_records_path)
        else:
            self.audio_records_path = audio_records_path
        self.last_audio_file: Path | None = None

        import whisper

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_name, device=resolved_device)
        self.language = language
        self.task = task

    def _get_audio_file(self) -> Path:
        """
        Retrieve the first .wav audio file from the records directory.

        Returns:
            Path: Path to the audio file.

        Raises:
            FileNotFoundError: If no audio files are found.
        """
        if self.audio_records_path is None:
            raise ValueError("Audio records path is not set.")
        audio_files = list(self.audio_records_path.glob("*.wav"))
        if not audio_files:
            raise FileNotFoundError("No audio files found.")
        self.last_audio_file = audio_files[0]
        return audio_files[0]

    def _delete_audio_file(self) -> None:
        """Delete the last processed audio file."""
        if self.last_audio_file and self.last_audio_file.exists():
            self.last_audio_file.unlink()
            print(f"Deleted audio file: {self.last_audio_file}")
            self.last_audio_file = None
        else:
            print("No audio file to delete or file does not exist.")

    def transcribe(self) -> str:
        """
        Transcribe the first audio file in the records directory.

        Returns:
            str: The transcription result.

        Raises:
            ValueError: If audio_records_path is not set.
            FileNotFoundError: If no audio files are found.
        """
        audio_file = self._get_audio_file()
        result = self.model.transcribe(
            str(audio_file),
            language=self.language,
            task=self.task,
        )
        transcription = result.get("text", "").strip()
        print(f"Transcription result: {transcription}")
        self._delete_audio_file()
        return transcription


def is_openai_whisper_available() -> bool:
    """Return True if the OpenAI Whisper package can be imported."""
    return importlib.util.find_spec("whisper") is not None
