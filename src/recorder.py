import logging
import pyaudio
import wave
import threading
import numpy as np
from collections import deque


class MicrophoneSelector:
    """Detect and manage available microphones."""

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.microphones = []
        self._detect_microphones()

    def _detect_microphones(self):
        self.microphones = []
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')

        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                self.microphones.append({
                    'index': i,
                    'name': device_info.get('name'),
                    'channels': device_info.get('maxInputChannels'),
                    'sample_rate': device_info.get('defaultSampleRate'),
                })

    def get_microphones(self):
        return self.microphones

    def get_default_microphone(self):
        if self.microphones:
            return self.microphones[0]
        return None

    def cleanup(self):
        self.audio.terminate()


class SilenceDetector:
    """Detect silence in an audio stream."""

    def __init__(self, silence_threshold=1000, silence_duration=2.0, sample_rate=44100):
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.sample_rate = sample_rate
        self.silence_frames = int(silence_duration * sample_rate / 1024)
        self.recent_volumes = deque(maxlen=self.silence_frames)
        self.is_recording_started = False
        self.speech_detected = False

    def _calculate_volume(self, audio_data):
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return 0.0
            mean_square = np.mean(audio_array.astype(np.float64) ** 2)
            if np.isnan(mean_square) or np.isinf(mean_square) or mean_square < 0:
                return 0.0
            volume = np.sqrt(mean_square)
            if np.isnan(volume) or np.isinf(volume):
                return 0.0
            return float(volume)
        except Exception:
            return 0.0

    def process_audio_chunk(self, audio_data):
        """Return True if recording should continue, False to stop."""
        volume = self._calculate_volume(audio_data)
        self.recent_volumes.append(volume)

        if volume > self.silence_threshold:
            self.speech_detected = True
            self.is_recording_started = True

        if not self.speech_detected:
            return True

        if len(self.recent_volumes) >= self.silence_frames:
            if all(vol < self.silence_threshold for vol in self.recent_volumes):
                return False

        return True

    def reset(self):
        self.recent_volumes.clear()
        self.is_recording_started = False
        self.speech_detected = False


class AudioRecorder:
    """Record audio from a microphone with automatic silence detection."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chunk_size = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 44100
        self.microphone_index = None

        self.is_recording = False
        self.audio_data = []
        self.recording_thread = None

        self._recording_lock = threading.Lock()
        self._stop_requested = False

        self.mic_selector = MicrophoneSelector()
        self.silence_detector = SilenceDetector(
            silence_threshold=350,
            silence_duration=1.5,
            sample_rate=self.sample_rate,
        )

        self.on_recording_start = None
        self.on_recording_stop = None
        self.on_volume_update = None

    def set_microphone(self, mic_index):
        self.microphone_index = mic_index

    def set_silence_settings(self, threshold, duration):
        self.silence_detector.silence_threshold = threshold
        self.silence_detector.silence_duration = duration
        self.silence_detector.silence_frames = int(duration * self.sample_rate / self.chunk_size)
        self.silence_detector.recent_volumes = deque(maxlen=self.silence_detector.silence_frames)

    def start_recording(self):
        with self._recording_lock:
            if self.is_recording:
                return False

            if self.microphone_index is None:
                self.logger.error("No microphone selected.")
                return False

            self.is_recording = True
            self._stop_requested = False
            self.audio_data = []
            self.silence_detector.reset()

            try:
                self.recording_thread = threading.Thread(target=self._record_audio)
                self.recording_thread.daemon = True
                self.recording_thread.start()
                self._safe_callback(self.on_recording_start)
                return True
            except Exception:
                self.logger.exception("Error starting the recording thread.")
                self.is_recording = False
                return False

    def stop_recording(self):
        self.logger.info("Stop requested for recording.")
        with self._recording_lock:
            if not self.is_recording:
                self.logger.info("No recording in progress.")
                return
            self._stop_requested = True
            self.is_recording = False

        if self.recording_thread and self.recording_thread.is_alive():
            self.logger.info("Waiting for the recording thread to finish.")
            self.recording_thread.join(timeout=3.0)
            if self.recording_thread.is_alive():
                self.logger.warning("Recording thread did not stop in time.")
            else:
                self.logger.info("Recording thread stopped cleanly.")

    def _safe_callback(self, callback, *args):
        if callback:
            try:
                callback(*args)
            except Exception:
                self.logger.exception("Error in recording callback.")

    def _record_audio(self):
        audio = None
        stream = None

        try:
            self.logger.info("Initializing recording.")
            audio = pyaudio.PyAudio()

            stream = audio.open(
                format=self.sample_format,
                channels=self.channels,
                rate=self.sample_rate,
                frames_per_buffer=self.chunk_size,
                input=True,
                input_device_index=self.microphone_index,
            )

            self.logger.info("Recording started.")

            while True:
                with self._recording_lock:
                    if self._stop_requested or not self.is_recording:
                        self.logger.info("Stop detected in the recording loop.")
                        break

                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_data.append(data)

                    with self._recording_lock:
                        if not self._stop_requested:
                            should_continue = self.silence_detector.process_audio_chunk(data)
                            if not should_continue:
                                self.logger.info("Silence detected — stopping recording.")
                                self.is_recording = False
                                break

                    volume = self.silence_detector._calculate_volume(data)
                    self._safe_callback(self.on_volume_update, volume)

                except Exception:
                    self.logger.exception("Error while reading audio.")
                    break

        except Exception:
            self.logger.exception("Recording error.")
        finally:
            self.logger.info("Cleaning up audio resources.")

            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    self.logger.exception("Error while closing the audio stream.")

            if audio:
                try:
                    audio.terminate()
                except Exception:
                    self.logger.exception("Error while terminating PyAudio.")

            with self._recording_lock:
                self.is_recording = False

            self.logger.info("Recording thread finished.")
            self._safe_callback(self.on_recording_stop)

    def save_recording(self, filename):
        if not self.audio_data:
            return False
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(pyaudio.get_sample_size(self.sample_format))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b''.join(self.audio_data))
            self.logger.info("Recording saved to %s.", filename)
            return True
        except Exception:
            self.logger.exception("Error saving recording.")
            return False

    def get_recording_duration(self):
        if not self.audio_data:
            return 0
        total_frames = len(self.audio_data) * self.chunk_size
        return total_frames / self.sample_rate

    def cleanup(self):
        self.stop_recording()
