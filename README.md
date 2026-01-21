# Commpanion Deaf Translator

Wake-word controlled pipeline that records speech, transcribes it with OpenAI Whisper (local), translates with M2M100, and plays the result with TTS. YOLOv8 helpers are included for future sign-language/vision work.

## Layout
- `src/wake_translation_assistant.py` – wake-word loop that routes to translation (sign-language branch placeholder).
- `src/translator.py` – record → STT → translate → TTS pipeline.
- `src/stt.py` – OpenAI Whisper (local) integration.
- `src/tts.py` – pyttsx3 helper.
- `src/recorder.py` – microphone capture with silence detection.
- `src/wakeword_detector.py` – openWakeWord wrapper.
- `src/yolov8Objects.py` – YOLOv8 object locator (kept for vision/sign language work).

## Setup
1) Python 3.10+ recommended.  
2) Install deps: `pip install -r requirements.txt` (PyAudio may need OS-specific tooling).  
   - OpenAI Whisper (local) requires `ffmpeg` available on your system for audio decoding.
3) Models:
   - OpenAI Whisper (local): models download automatically on first run (default: `base`). Ensure `ffmpeg` is installed.
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

To use the OpenAI Whisper open-source model locally:
```bash
python -m src.translator --source-lang en --target-lang fr --stt-model openai_whisper:base --once
```

To use the OpenAI Whisper open-source model locally:
```bash
python -m src.translator --source-lang en --target-lang fr --stt-model openai_whisper:base --once
```

## Notes
- This repo focuses only on STT, translation, TTS, wake word, and YOLOv8 helpers extracted from `commpanion-blind-deaf`. Other intents (OCR, BLIP, collision detection, etc.) are omitted.  
- The pipeline uses the first available microphone detected by PyAudio; adjust `MicrophoneSelector` or `TranslatorPipeline` if you need a specific device.  
- The YOLO module depends on `lmstudio` only if you want to run the LLM-based object locator.
