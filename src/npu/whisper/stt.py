from __future__ import annotations

from pathlib import Path

from .audio_features import WhisperAudioFeaturesMixin
from .decoder_runtime import WhisperDecoderRuntimeMixin
from .init_runtime import WhisperInitRuntimeMixin, dump_model_io
from .kv_debug import WhisperKvDebugMixin
from .token_selection import WhisperTokenSelectionMixin


class WhisperQnnSTT(
    WhisperInitRuntimeMixin,
    WhisperAudioFeaturesMixin,
    WhisperDecoderRuntimeMixin,
    WhisperTokenSelectionMixin,
    WhisperKvDebugMixin,
):
    """Run Whisper QNN STT (small-quantized or large-v3-turbo) with ONNX Runtime QNN."""


class WhisperSmallQuantizedQNNSTT(WhisperQnnSTT):
    """Backward-compatible wrapper for the small quantized Whisper profile."""

    def __init__(
        self,
        encoder_dir: str | Path,
        decoder_dir: str | Path,
        prefer_qnn: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(
            encoder_dir=encoder_dir,
            decoder_dir=decoder_dir,
            stt_model="small-quantized",
            prefer_qnn=prefer_qnn,
            debug=debug,
        )


class WhisperLargeV3TurboQNNSTT(WhisperQnnSTT):
    """Convenience wrapper for the large-v3-turbo Whisper profile."""

    def __init__(
        self,
        encoder_dir: str | Path,
        decoder_dir: str | Path,
        prefer_qnn: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(
            encoder_dir=encoder_dir,
            decoder_dir=decoder_dir,
            stt_model="large-v3-turbo",
            prefer_qnn=prefer_qnn,
            debug=debug,
        )
