from pathlib import Path

import importlib
import importlib.util

import torch


class _AudioRecordMixin:
    def __init__(self, audio_records_path: Path | str | None) -> None:
        if isinstance(audio_records_path, str):
            self.audio_records_path: Path | None = Path(audio_records_path)
        else:
            self.audio_records_path = audio_records_path
        self.last_audio_file: Path | None = None

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


class SpeechToTextApplication(_AudioRecordMixin):
    """
    Application for transcribing speech from audio files using WhisperBase models.
    """

    def __init__(
        self,
        audio_records_path: Path | str | None = None,
        models_dir: Path | str | None = None,
        model_name: str = "whisper_base",
    ) -> None:
        """
        Initialize the SpeechToTextApplication.

        Args:
            audio_records_path (Path | str | None): Path to the directory containing audio files.
            models_dir (Path | str | None): Directory that contains the ONNX encoder/decoder
                exported from WhisperBaseEn (`*_encoderinf.onnx` / `*_decoderinf.onnx`).
        """
        super().__init__(audio_records_path)

        whisper_model, encoder_filename, decoder_filename = _load_qai_whisper_assets(model_name)

        base_models_dir = (
            Path(models_dir) if models_dir is not None else Path(__file__).parent / "models"
        )
        encoder_path = base_models_dir / encoder_filename
        decoder_path = base_models_dir / decoder_filename

        WhisperApp, OnnxModelTorchWrapper = _load_qai_whisper_runtime()
        self.app = WhisperApp(
            OnnxModelTorchWrapper.OnNPU(str(encoder_path)),
            OnnxModelTorchWrapper.OnNPU(str(decoder_path)),
            num_decoder_blocks=whisper_model.num_decoder_blocks,
            num_decoder_heads=whisper_model.num_decoder_heads,
            attention_dim=whisper_model.attention_dim,
            mean_decode_len=whisper_model.mean_decode_len,
        )

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
        transcription = self.app.transcribe(str(audio_file), audio_sample_rate=None)
        print(f"Transcription result: {transcription}")
        self._delete_audio_file()
        return transcription


class OpenAIWhisperSpeechToText(_AudioRecordMixin):
    """Speech-to-text using the OpenAI Whisper open-source model."""

    def __init__(
        self,
        audio_records_path: Path | str | None = None,
        model_name: str = "base",
        device: str | None = None,
        language: str | None = None,
        task: str = "transcribe",
    ) -> None:
        super().__init__(audio_records_path)
        if not is_openai_whisper_available():
            raise RuntimeError(
                "openai-whisper is not installed. Install it with `pip install openai-whisper`."
            )
        import whisper

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_name, device=resolved_device)
        self.language = language
        self.task = task

    def transcribe(self) -> str:
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


def is_whisper_base_available() -> bool:
    """Return True if the multilingual whisper_base module can be imported."""
    return (
        importlib.util.find_spec("qai_hub_models.models.whisper_base") is not None
        or importlib.util.find_spec("qai_hub_models.models.whisper_base.model") is not None
    )


def _load_whisper_base_model():
    try:
        whisper_base_module = importlib.import_module("qai_hub_models.models.whisper_base")
    except ModuleNotFoundError as exc:
        try:
            whisper_base_module = importlib.import_module(
                "qai_hub_models.models.whisper_base.model"
            )
        except ModuleNotFoundError as nested_exc:
            raise RuntimeError(
                "whisper_base is unavailable in the installed qai-hub-models package. "
                "Please upgrade qai-hub-models or install a version that includes "
                "qai_hub_models.models.whisper_base."
            ) from nested_exc
    model_cls = getattr(whisper_base_module, "Model", None) or getattr(
        whisper_base_module, "WhisperBase", None
    )
    if model_cls is None:
        raise RuntimeError(
            "whisper_base model class not found. Expected Model or WhisperBase in "
            "qai_hub_models.models.whisper_base."
        )
    return model_cls.from_pretrained()


def is_openai_whisper_available() -> bool:
    """Return True if the OpenAI Whisper package can be imported."""
    return importlib.util.find_spec("whisper") is not None


def _load_qai_whisper_runtime():
    try:
        from qai_hub_models.models._shared.whisper.app import WhisperApp
        from qai_hub_models.utils.onnx_torch_wrapper import OnnxModelTorchWrapper
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qai-hub-models is not installed. Install it with `pip install qai-hub-models` "
            "to use the ONNX Whisper backend."
        ) from exc
    return WhisperApp, OnnxModelTorchWrapper


def _load_qai_whisper_assets(model_name: str):
    if model_name == "whisper_base":
        model = _load_whisper_base_model()
        encoder_filename = "whisper_base-whisperencoderinf.onnx"
        decoder_filename = "whisper_base-whisperdecoderinf.onnx"
    else:
        raise ValueError("Unsupported Whisper model name. Use 'whisper_base'.")
    return model, encoder_filename, decoder_filename
