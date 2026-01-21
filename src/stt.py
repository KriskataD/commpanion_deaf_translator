from pathlib import Path

import importlib

from qai_hub_models.models._shared.whisper.app import WhisperApp
from qai_hub_models.models.whisper_base.model import WhisperBase
from qai_hub_models.models.whisper_base_en.model import WhisperBaseEn
from qai_hub_models.utils.onnx_torch_wrapper import OnnxModelTorchWrapper


class SpeechToTextApplication:
    """
    Application for transcribing speech from audio files using WhisperBase models.
    """

    def __init__(
        self,
        audio_records_path: Path | str | None = None,
        models_dir: Path | str | None = None,
        model_name: str = "whisper_base_en",
    ) -> None:
        """
        Initialize the SpeechToTextApplication.

        Args:
            audio_records_path (Path | str | None): Path to the directory containing audio files.
            models_dir (Path | str | None): Directory that contains the ONNX encoder/decoder
                exported from WhisperBaseEn (`*_encoderinf.onnx` / `*_decoderinf.onnx`).
        """
        if model_name == "whisper_base_en":
            self.model = WhisperBaseEn.from_pretrained()
            encoder_filename = "whisper_base_en-whisperencoderinf.onnx"
            decoder_filename = "whisper_base_en-whisperdecoderinf.onnx"
        elif model_name == "whisper_base":
            whisper_base_module = importlib.import_module(
                "qai_hub_models.models.whisper_base.model"
            )
            self.model = whisper_base_module.WhisperBase.from_pretrained()
            encoder_filename = "whisper_base-whisperencoderinf.onnx"
            decoder_filename = "whisper_base-whisperdecoderinf.onnx"
        else:
            raise ValueError(
                "Unsupported Whisper model name. Use 'whisper_base_en' or 'whisper_base'."
            )

        base_models_dir = (
            Path(models_dir) if models_dir is not None else Path(__file__).parent / "models"
        )
        encoder_path = base_models_dir / encoder_filename
        decoder_path = base_models_dir / decoder_filename

        self.app = WhisperApp(
            OnnxModelTorchWrapper.OnNPU(str(encoder_path)),
            OnnxModelTorchWrapper.OnNPU(str(decoder_path)),
            num_decoder_blocks=self.model.num_decoder_blocks,
            num_decoder_heads=self.model.num_decoder_heads,
            attention_dim=self.model.attention_dim,
            mean_decode_len=self.model.mean_decode_len,
        )
        if isinstance(audio_records_path, str):
            self.audio_records_path: Path | None = Path(audio_records_path)
        else:
            self.audio_records_path: Path | None = audio_records_path
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
        """
        Delete the last processed audio file.
        """
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
        transcription = self.app.transcribe(str(audio_file), audio_sample_rate=None)
        print(f"Transcription result: {transcription}")
        self._delete_audio_file()
        return transcription
