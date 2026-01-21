# Commpanion Deaf Translator

Wake-word controlled pipeline that records speech, transcribes it with Whisper (ONNX), translates with M2M100, and plays the result with TTS. YOLOv8 helpers are included for future sign-language/vision work.

## Layout
- `src/wake_translation_assistant.py` – wake-word loop that routes to translation (sign-language branch placeholder).
- `src/translator.py` – record → STT → translate → TTS pipeline.
- `src/stt.py` – Whisper ONNX runner (place ONNX exports in `src/models/`).
- `src/tts.py` – pyttsx3 helper.
- `src/recorder.py` – microphone capture with silence detection.
- `src/wakeword_detector.py` – openWakeWord wrapper.
- `src/yolov8Objects.py` – YOLOv8 object locator (kept for vision/sign language work).

## Setup
1) Python 3.10+ recommended.  
2) Install deps: `pip install -r requirements.txt` (PyAudio may need OS-specific tooling).  
3) Models:
   - Whisper ONNX:
     - English-only: export `whisper_base_en-whisperencoderinf.onnx` and `whisper_base_en-whisperdecoderinf.onnx` into `src/models/`.
     - Multilingual (recommended for non-English input): export `whisper_base-whisperencoderinf.onnx` and `whisper_base-whisperdecoderinf.onnx` into `src/models/`.
   - Wake word: `openwakeword` downloads defaults on first run; to use a custom model, place it in `src/models/` and pass `--wakeword path/to/model.onnx`.
   - YOLO: download a YOLOv8 weights file (e.g., `yolov8l-oiv7.pt`) into `src/models/` and update `src/yolov8Objects.py` if you want to run it.

## Run the translation pipeline
```bash
python -m src.wake_translation_assistant --source-lang en --target-lang fr
```
- Say the wake word (default: `hey_jarvis`).  
- After the prompt, speak the phrase to translate; translation is spoken back via TTS.  
- Saying something about "sign language" will currently reply that the branch is not ready.

For a translation-only loop without wake word you can also run:
```bash
python -m src.translator --source-lang en --target-lang fr --once
```

For non-English source languages (e.g., Bulgarian), either let the default auto-selection
choose the multilingual Whisper model or set it explicitly:
```bash
python -m src.translator --source-lang bg --target-lang en --stt-model whisper_base --once
```

## Notes
- This repo focuses only on STT, translation, TTS, wake word, and YOLOv8 helpers extracted from `commpanion-blind-deaf`. Other intents (OCR, BLIP, collision detection, etc.) are omitted.  
- The pipeline uses the first available microphone detected by PyAudio; adjust `MicrophoneSelector` or `TranslatorPipeline` if you need a specific device.  
- The YOLO module depends on `lmstudio` only if you want to run the LLM-based object locator.
