# Commpanion — Real-Time Speech Translator for Deaf and Hard-of-Hearing Users

An edge AI assistant that listens for a wake word, transcribes speech using a **quantized Whisper model running on a Qualcomm NPU**, translates it with Meta's M2M100 model, and speaks the result aloud — entirely on-device, with no cloud dependency for the STT step.

This project was developed as a Final Year Project at University College Cork (UCC), with a focus on running inference on-device hardware accelerators (NPU/DSP) via Qualcomm's QNN SDK.

---

## Key Technical Highlights

- **NPU-accelerated inference** — Whisper Small quantized to INT8/FP16 and executed via ONNX Runtime's `QNNExecutionProvider` on a Qualcomm Snapdragon NPU/DSP.
- **Custom autoregressive decoder loop** — manually manages KV self-cache, cross-attention cache, attention masks, and position IDs to drive the quantized decoder step-by-step outside of HuggingFace's generate loop.
- **Multilingual translation** — Meta's M2M100 (418M) supports direct translation between 100+ language pairs, no pivot through English required.
- **Wake-word activated pipeline** — openWakeWord listens continuously in the background; the pipeline only activates on a detected wake word, minimising power use.
- **Thread-safe audio handling** — recording, wake-word detection, TTS playback, and the main loop each run on separate threads with lock-guarded state.

---

## Architecture

```
Microphone
    │
    ▼
WakeWordDetector (openWakeWord, background thread)
    │  wake word detected
    ▼
AudioRecorder (PyAudio + silence detection)
    │  .wav file
    ▼
WhisperSmallQuantizedQNNSTT (ONNX Runtime + QNNExecutionProvider)
    │  transcription text
    ▼
MultiLanguageTranslator (facebook/m2m100_418M, HuggingFace)
    │  translated text
    ▼
TTS (_TTS via Windows SAPI / pyttsx3)
    │
    ▼
Speaker output
```

The `WakeWordTranslationAssistant` orchestrates the full loop; `TranslatorPipeline` owns the STT/translation/TTS components.

---

## Features

- Wake-word activation (`hey_jarvis` by default; configurable)
- **Stay-awake mode** — keep translating after each utterance without re-triggering the wake word (`--stay-awake`)
- **OCR detection mode** — point AR glasses at printed text; EasyOCR (quantized, QNN-accelerated) reads and translates it, overlaying subtitles via a captions client
- **AR subtitle overlay** — translated text is displayed as on-screen captions in real time
- Voice stop commands — say *"stop listening"* or *"stop Jarvis"* to exit
- Non-English source language support — stop commands are re-verified with an English transcription pass
- Performance tracking — per-session STT, translation, and OCR timing metrics logged on shutdown
- Configurable TTS timeout to prevent playback hangs
- Debug logging for wake word scores, encoder/decoder IO shapes, and token selection

---

## Tech Stack

| Layer | Technology |
|---|---|
| Wake word | [openWakeWord](https://github.com/dscripka/openWakeWord) |
| Speech recognition | OpenAI Whisper Small / Large-v3-Turbo (quantized) via ONNX Runtime + QNN |
| OCR | EasyOCR (quantized detector + recogniser) via ONNX Runtime + QNN |
| NPU runtime | Qualcomm QNN SDK / `QNNExecutionProvider` |
| Translation | `facebook/m2m100_418M` (HuggingFace Transformers) |
| AR subtitles | Custom captions overlay client |
| Text-to-speech | Windows SAPI via `pywin32`; `pyttsx3` fallback |
| Audio I/O | PyAudio |
| ML frameworks | PyTorch, ONNX Runtime |

---

## Setup

**Requirements:** Python 3.10+, Windows (SAPI TTS), Qualcomm device with QNN runtime for NPU inference (CPU fallback available via env vars).

```bash
pip install -r requirements.txt
```

PyAudio may require OS-specific build tooling (e.g. `pipwin install pyaudio` on Windows).

### Models

| Model | How to obtain |
|---|---|
| QNN Whisper encoder | Export from `openai/whisper-small` with Qualcomm AI Hub or ONNX export tools; quantize to INT8/FP16 |
| QNN Whisper decoder | Same export pipeline as encoder |
| M2M100 | Downloaded automatically from HuggingFace on first run |
| Wake word | `openwakeword` downloads `hey_jarvis` automatically on first run |

Place the ONNX model directories at:
```
models/
  whisper_small_quantized_encoder_optimized_onnx/
    model.onnx
    model.bin
  whisper_small_quantized_decoder_optimized_onnx/
    model.onnx
    model.bin
```

Or override with `--qnn-encoder-dir` / `--qnn-decoder-dir` flags, or the `QNN_ENCODER_DIR` / `QNN_DECODER_DIR` environment variables.

---

## Usage

### Full wake-word pipeline

```bash
python -m src.wake_translation_assistant \
  --source-lang en \
  --target-lang fr \
  --qnn-encoder-dir models/whisper_small_quantized_encoder_optimized_onnx \
  --qnn-decoder-dir models/whisper_small_quantized_decoder_optimized_onnx
```

Say **"hey Jarvis"**, then say **"translate"**, then speak your phrase. The translation is spoken back via TTS.

### Stay-awake mode (continuous translation)

```bash
python -m src.wake_translation_assistant \
  --source-lang bg \
  --target-lang en \
  --qnn-encoder-dir models/whisper_small_quantized_encoder_optimized_onnx \
  --qnn-decoder-dir models/whisper_small_quantized_decoder_optimized_onnx \
  --stay-awake \
  --no-prompt \
  --tts-timeout 5
```

### Single translation (no wake word)

```bash
python -m src.translator \
  --source-lang en \
  --target-lang fr \
  --once \
  --qnn-encoder-dir models/whisper_small_quantized_encoder_optimized_onnx \
  --qnn-decoder-dir models/whisper_small_quantized_decoder_optimized_onnx
```

### Useful flags

| Flag | Description |
|---|---|
| `--stay-awake` | Keep translating after each phrase without re-triggering the wake word |
| `--no-speak` | Disable TTS output |
| `--no-prompt` | Skip the spoken prompt before recording |
| `--tts-timeout N` | Stop TTS playback after N seconds |
| `--wake-debug` | Log wake word scores every second for microphone/detection debugging |
| `--wake-mic-index N` | Force a specific PyAudio input device index for wake word detection |

### CPU fallback (debugging without QNN hardware)

```bash
set QNN_ENCODER_CPU=1
set QNN_DECODER_CPU=1
python -m src.wake_translation_assistant ...
```

---

## Project Structure

```
src/
├── wake_translation_assistant.py   # Top-level orchestrator: wake word → route → pipeline
├── translator.py                   # TranslatorPipeline: record → STT → translate → TTS + OCR
├── recorder.py                     # PyAudio recorder with silence detection
├── wakeword_detector.py            # openWakeWord wrapper with callback registration
├── tts.py                          # Windows SAPI / pyttsx3 TTS with worker queue
├── captions_client.py              # AR subtitle overlay client
├── captions_overlay.py             # Subtitle rendering logic
├── yolov8Objects.py                # YOLOv8 object locator (future sign-language extension)
├── ocr/
│   ├── easyocr_qnn.py              # EasyOCR detector + recogniser via QNN
│   └── scan_once.py                # Single-frame OCR scan helper
└── npu/
    ├── ort_qnn.py                  # ONNX Runtime session factory (QNNExecutionProvider)
    ├── whisper_qnn_stt.py          # Public API facade for Whisper QNN STT
    └── whisper/                    # Whisper inference internals
        ├── stt.py                  # WhisperQnnSTT / model profiles
        ├── decoder_runtime.py      # Autoregressive decoder loop + KV cache
        ├── audio_features.py       # Mel spectrogram extraction
        ├── token_selection.py      # Greedy decoding + repeat guards
        └── profiles.py             # Model profile definitions (small, large-v3-turbo)
scripts/
└── check_ort_qnn.py                # Utility: verify QNN provider availability
models/                             # (gitignored) ONNX model directories
audio/                              # (gitignored) Temporary WAV recordings
```
