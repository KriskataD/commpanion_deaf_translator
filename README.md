# Commpanion Deaf Translator

Wake-word controlled pipeline that records speech, transcribes it with QNN Whisper (NPU), translates with M2M100, and plays the result with TTS. YOLOv8 helpers are included for future sign-language/vision work.

## Layout
- `src/wake_translation_assistant.py` – wake-word loop that routes to translation (sign-language branch placeholder).
- `src/translator.py` – record → STT → translate → TTS pipeline.
- `src/npu/whisper_qnn_stt.py` – QNN Whisper (NPU) integration.
- `src/tts.py` – pyttsx3 helper.
- `src/recorder.py` – microphone capture with silence detection.
- `src/wakeword_detector.py` – openWakeWord wrapper.
- `src/yolov8Objects.py` – YOLOv8 object locator (kept for vision/sign language work).

## Setup
1) Python 3.10+ recommended.  
2) Install deps: `pip install -r requirements.txt` (PyAudio may need OS-specific tooling).
3) Models:
   - QNN Whisper: export the quantized encoder/decoder ONNX models and set the directories with
     `--qnn-encoder-dir`/`--qnn-decoder-dir` or the `QNN_ENCODER_DIR`/`QNN_DECODER_DIR` environment variables.
   - Wake word: `openwakeword` downloads defaults on first run; to use a custom model, place it in `src/models/` and pass `--wakeword path/to/model.onnx`.
   - YOLO: download a YOLOv8 weights file (e.g., `yolov8l-oiv7.pt`) into `src/models/` and update `src/yolov8Objects.py` if you want to run it.

## Run the translation pipeline
```bash
python -m src.wake_translation_assistant --source-lang en --target-lang fr \
  --qnn-encoder-dir models/whisper_small_quantized_encoder_optimized_onnx \
  --qnn-decoder-dir models/whisper_small_quantized_decoder_optimized_onnx
```
- Say the wake word (default: `hey_jarvis`).  
- After the prompt, speak the phrase to translate; translation is spoken back via TTS.  
- Saying something about "sign language" will currently reply that the branch is not ready.
- If wake word detection is not triggering, try specifying the input device index with
  `--wake-mic-index` (see PyAudio device list on your system).
- By default the wake word listener now uses the same default mic selected for recording.
- To debug detection, add `--wake-debug` to log the top wake word score every second and
  verify that audio is reaching the model (adjust `--wake-debug-interval` as needed).
- If you pass a model path (e.g., `hey_jarvis_v0.1.onnx`), callbacks are normalized to
  `hey_jarvis` so detections still trigger.
- During capture the wake-word listener is paused to avoid microphone contention.
- Use `--no-prompt` to skip the spoken prompt and `--no-speak` to disable spoken translations.
- Use `--stay-awake` to keep translating without saying the wake word again (say "stop listening" to exit).
- If TTS playback hangs, use `--no-speak` or set `--tts-timeout 5` to stop speech after a few seconds.

For a translation-only loop without wake word you can also run:
```bash
python -m src.translator --source-lang en --target-lang fr --once \
  --qnn-encoder-dir models/whisper_small_quantized_encoder_optimized_onnx \
  --qnn-decoder-dir models/whisper_small_quantized_decoder_optimized_onnx
```

Explicit STT profile selection examples:
```bash
python -m src.translator --once --stt-model small-quantized
python -m src.translator --once --stt-model large-v3-turbo
```
You can still override model directories with `--qnn-encoder-dir` and `--qnn-decoder-dir`.

To run the full pipeline:
```bash
python -m src.wake_translation_assistant   --source-lang bg   --target-lang en   --wakeword hey_jarvis   --stay-awake   --tts-timeout 30   --stt-model large-v3-turbo   --captions-auto-start   --captions-monitor-index 1
```

## Notes
- This repo focuses only on STT, translation, TTS, wake word, and YOLOv8 helpers extracted from `commpanion-blind-deaf`. Other intents (OCR, BLIP, collision detection, etc.) are omitted.  
- The pipeline uses the first available microphone detected by PyAudio; adjust `MicrophoneSelector` or `TranslatorPipeline` if you need a specific device.  
- The YOLO module depends on `lmstudio` only if you want to run the LLM-based object locator.
