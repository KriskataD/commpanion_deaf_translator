"""Performance benchmark runner for the translator branch."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


MODES = ("text_only", "full_pipeline")


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg": None, "min": None, "max": None, "std": None}
    avg = sum(values) / len(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"avg": avg, "min": min(values), "max": max(values), "std": std}


def _word_count(text: str) -> int:
    return len([token for token in text.strip().split() if token])


def _sample_resources(process: Any) -> tuple[float | None, float | None]:
    if process is None:
        return None, None
    rss_mb = process.memory_info().rss / (1024 * 1024)
    cpu_pct = process.cpu_percent(interval=None)
    return rss_mb, cpu_pct


def run_benchmark(
    *,
    mode: str,
    source_lang: str,
    target_lang: str,
    speak: bool,
    tts_timeout: float | None,
    cycles: int,
    text_inputs: list[str],
    mock_translate: bool,
) -> dict[str, Any]:
    translator = None
    tts = None
    pipeline = None

    if mode == "text_only":
        if not mock_translate:
            from .translator import MultiLanguageTranslator
            from .tts import _TTS

            translator = MultiLanguageTranslator()
            tts = _TTS() if speak else None
    else:
        from .translator import TranslatorPipeline

        pipeline = TranslatorPipeline(
            source_lang=source_lang,
            target_lang=target_lang,
            speak=speak,
            tts_timeout=tts_timeout,
        )

    process = psutil.Process() if psutil else None
    if process:
        process.cpu_percent(interval=None)

    attempted_cycles = cycles
    successful_cycles = 0

    record_times: list[float] = []
    stt_times: list[float] = []
    translation_times: list[float] = []
    tts_times: list[float] = []
    cycle_times: list[float] = []

    input_words: list[int] = []
    output_words: list[int] = []
    input_chars: list[int] = []
    output_chars: list[int] = []

    rss_samples: list[float] = []
    cpu_samples: list[float] = []

    session_start = time.perf_counter()

    for index in range(cycles):
        cycle_start = time.perf_counter()
        translated = ""

        if mode == "text_only":
            record_times.append(0.0)
            stt_times.append(0.0)

            text = text_inputs[index % len(text_inputs)]

            translation_start = time.perf_counter()
            translated = text if mock_translate else translator.translate(text, source_lang, target_lang)
            translation_times.append(time.perf_counter() - translation_start)

            tts_start = time.perf_counter()
            if tts and translated:
                tts.start(translated, timeout_s=tts_timeout)
            tts_times.append(time.perf_counter() - tts_start)
        else:
            record_start = time.perf_counter()
            audio_path = pipeline.record(filename=f"benchmark_{index}.wav")
            record_times.append(time.perf_counter() - record_start)
            if not audio_path:
                stt_times.append(0.0)
                translation_times.append(0.0)
                tts_times.append(0.0)
                cycle_times.append(time.perf_counter() - cycle_start)
                continue

            stt_start = time.perf_counter()
            text = pipeline.transcribe()
            stt_times.append(time.perf_counter() - stt_start)

            translation_start = time.perf_counter()
            translated = pipeline.translator.translate(text, source_lang, target_lang)
            translation_times.append(time.perf_counter() - translation_start)

            tts_start = time.perf_counter()
            if pipeline.speak and translated and pipeline.tts:
                pipeline.tts.start(translated, timeout_s=tts_timeout)
            tts_times.append(time.perf_counter() - tts_start)

        if translated.strip():
            successful_cycles += 1

        input_words.append(_word_count(text))
        output_words.append(_word_count(translated))
        input_chars.append(len(text))
        output_chars.append(len(translated))

        rss_mb, cpu_pct = _sample_resources(process)
        if rss_mb is not None:
            rss_samples.append(rss_mb)
        if cpu_pct is not None:
            cpu_samples.append(cpu_pct)

        cycle_times.append(time.perf_counter() - cycle_start)

    session_duration = time.perf_counter() - session_start
    translations_per_minute = (
        successful_cycles / (session_duration / 60.0) if session_duration > 0 else math.nan
    )

    effective_mode = mode
    if mode == "text_only" and mock_translate:
        effective_mode = "text_only_mock"

    return {
        "session_duration_s": session_duration,
        "speech_translation": {
            "mode": effective_mode,
            "attempted_cycles": attempted_cycles,
            "successful_cycles": successful_cycles,
            "success_rate": (successful_cycles / attempted_cycles) if attempted_cycles else 0.0,
            "translations_per_minute": translations_per_minute,
            "latency_s": {
                "record_time_s": _stats(record_times),
                "stt_time_s": _stats(stt_times),
                "translation_time_s": _stats(translation_times),
                "tts_time_s": _stats(tts_times),
                "cycle_total_time_s": _stats(cycle_times),
            },
            "text_volume": {
                "avg_input_words": (sum(input_words) / len(input_words)) if input_words else 0,
                "avg_output_words": (sum(output_words) / len(output_words)) if output_words else 0,
                "avg_input_chars": (sum(input_chars) / len(input_chars)) if input_chars else 0,
                "avg_output_chars": (sum(output_chars) / len(output_chars)) if output_chars else 0,
            },
            "resources": {
                "psutil_available": psutil is not None,
                "avg_rss_memory_mb": (sum(rss_samples) / len(rss_samples)) if rss_samples else None,
                "peak_rss_memory_mb": max(rss_samples) if rss_samples else None,
                "avg_cpu_percent": (sum(cpu_samples) / len(cpu_samples)) if cpu_samples else None,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run translator performance metrics and emit JSON output."
    )
    parser.add_argument("--mode", choices=MODES, default="text_only")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="fr")
    parser.add_argument("--no-speak", action="store_true")
    parser.add_argument("--tts-timeout", type=float, default=None)
    parser.add_argument("--cycles", type=int, default=3, help="Number of benchmark cycles.")
    parser.add_argument(
        "--input-text",
        action="append",
        dest="input_texts",
        help="Input text for text-only benchmark mode. Can be passed multiple times.",
    )
    parser.add_argument(
        "--mock-translate",
        action="store_true",
        help="Use identity translation to benchmark without loading models (text_only mode only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to write metrics JSON.",
    )

    args = parser.parse_args()
    input_texts = args.input_texts or [
        "Hello, how are you today?",
        "Please let me know if you need help.",
        "I can translate your message into another language.",
    ]

    metrics = run_benchmark(
        mode=args.mode,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        speak=not args.no_speak,
        tts_timeout=args.tts_timeout,
        cycles=max(args.cycles, 1),
        text_inputs=input_texts,
        mock_translate=args.mock_translate,
    )

    rendered = json.dumps(metrics, indent=2)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
