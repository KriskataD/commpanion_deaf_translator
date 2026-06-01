import numpy as np
import pyaudio
from openwakeword.model import Model
from openwakeword.utils import download_models
import threading
import queue
import os
import re
from pathlib import Path
import time
from typing import Callable, Optional, Dict, Any
import logging

class WakeWordDetector:
    """Modular wake word detector using openWakeWord."""

    _MODEL_SUFFIX_RE = re.compile(r"_v\d+(?:\.\d+)*$")
    
    def __init__(
        self,
        wakeword_models: list[str] = None,
        inference_framework: str = 'onnx',
        threshold: float = 0.25,
        chunk_size: int = 1280,
        sample_rate: int = 16000,
        channels: int = 1,
        input_device_index: Optional[int] = None,
        log_predictions: bool = False,
        log_interval_s: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the wake word detector.
        
        Args:
            wakeword_models: List of models to load (default: ['hey_jarvis'])
            inference_framework: Inference framework ('onnx' or 'tflite')
            threshold: Detection threshold (0-1)
            chunk_size: Audio chunk size
            sample_rate: Sample rate
            channels: Number of audio channels
            logger: Custom logger
        """
        self.wakeword_models = wakeword_models or ['hey_jarvis']
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.input_device_index = input_device_index
        self.log_predictions = log_predictions
        self.log_interval_s = log_interval_s
        self.logger = logger or logging.getLogger(__name__)
        
        # Model initialization
        self.model = Model(
            wakeword_models=self.wakeword_models,
            inference_framework=inference_framework
        )
        
        # Audio configuration
        self.audio_format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Threading and state
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.callbacks: Dict[str, Callable] = {}
        self._last_log_time = time.monotonic()

    def list_input_devices(self) -> list[dict[str, Any]]:
        """Return available input devices from PyAudio."""
        devices: list[dict[str, Any]] = []
        for index in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0) > 0:
                devices.append(info)
        return devices
        
    def register_callback(self, wakeword: str, callback: Callable[[str, float], Any]):
        """
        Register a callback function for a specific wake word.
        
        Args:
            wakeword: Wake word name
            callback: Function to call (receives the detected word and the score)
        """
        normalized = self._normalize_wakeword_name(wakeword)
        self.callbacks[normalized] = callback
        if normalized != wakeword:
            self.callbacks[wakeword] = callback
            self.logger.info(
                "Callback registered for '%s' (normalized from '%s')",
                normalized,
                wakeword,
            )
        else:
            self.logger.info("Callback registered for '%s'", wakeword)

    def _normalize_wakeword_name(self, wakeword: str) -> str:
        """Normalize wake word names to match openWakeWord prediction keys."""
        stem = Path(wakeword).stem
        return self._MODEL_SUFFIX_RE.sub("", stem)
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for the audio stream."""
        if self.is_listening:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self):
        """Audio processing thread."""
        self.logger.info("Starting audio processing")
        
        while self.is_listening:
            try:
                # Get audio data
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Prediction
                predictions = self.model.predict(audio_array)

                if self.log_predictions:
                    now = time.monotonic()
                    if now - self._last_log_time >= self.log_interval_s:
                        best = max(predictions.items(), key=lambda item: item[1], default=None)
                        if best:
                            self.logger.info(
                                "Wake word scores (top): %s=%.3f",
                                best[0],
                                best[1],
                            )
                        self._last_log_time = now
                
                # Check detections
                for wakeword, score in predictions.items():
                    normalized = self._normalize_wakeword_name(wakeword)
                    if score > self.threshold:
                        self.logger.info(
                            "Wake word detected: %s (score: %.2f)",
                            normalized,
                            score,
                        )
                        # Call the callback if available
                        if normalized in self.callbacks:
                            try:
                                self.callbacks[normalized](normalized, score)
                            except Exception as e:
                                self.logger.error(f"Error in callback: {e}")
                        elif wakeword in self.callbacks:
                            try:
                                self.callbacks[wakeword](wakeword, score)
                            except Exception as e:
                                self.logger.error(f"Error in callback: {e}")
                        
                        # Reset to avoid multiple detections
                        self.model.reset()
                        
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error during audio processing: {e}")

    @classmethod
    def download_models(cls):
        if not os.path.exists("resources/models"):
            logging.getLogger(__name__).info("Downloading openWakeWord models.")
            download_models()
    
    def start(self):
        """Start wake word listening."""

        print("Starting wake word detector")

        if self.is_listening:
            self.logger.warning("The detector is already running")
            return

        if self.input_device_index is not None:
            self.logger.info(
                "Using wake word input device index: %s",
                self.input_device_index,
            )
        else:
            self.logger.info("Using default input device for wake word detection.")
        
        self.logger.info("Starting wake word detector")
        self.is_listening = True
        
        # Open the audio stream
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.input_device_index,
            stream_callback=self._audio_callback,
        )
        
        # Start the processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.start()
        
        self.logger.info("Detector started and listening")
    
    def stop(self):
        """Stop listening."""
        if not self.is_listening:
            return
        
        self.logger.info("Stopping detector")
        self.is_listening = False
        
        # Wait for the thread to finish
        if hasattr(self, 'processing_thread'):
            if threading.current_thread() is not self.processing_thread:
                self.processing_thread.join()
        
        # Close the stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Empty the queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        self.logger.info("Detector stopped")
    
    def __enter__(self):
        """Context manager for automatic start."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager for automatic stop."""
        self.stop()
        
    def cleanup(self):
        """Resource cleanup."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
