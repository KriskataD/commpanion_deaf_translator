import pyaudio
import wave
import threading
import numpy as np
from collections import deque

class MicrophoneSelector:
    """Class to detect and manage available microphones"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.microphones = []
        self._detect_microphones()
    
    def _detect_microphones(self):
        """Detect all available microphones"""
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
                    'sample_rate': device_info.get('defaultSampleRate')
                })
    
    def get_microphones(self):
        """Return the list of available microphones"""
        return self.microphones
    
    def get_default_microphone(self):
        """Return the default microphone"""
        if self.microphones:
            return self.microphones[0]
        return None
    
    def cleanup(self):
        """Clean up PyAudio resources"""
        self.audio.terminate()


class SilenceDetector:
    """Class to detect silence in audio"""
    
    def __init__(self, silence_threshold=1000, silence_duration=2.0, sample_rate=44100):
        self.silence_threshold = silence_threshold  # Amplitude threshold to detect silence
        self.silence_duration = silence_duration    # Silence duration before stopping (seconds)
        self.sample_rate = sample_rate
        self.silence_frames = int(silence_duration * sample_rate / 1024)  # Number of silence frames
        self.recent_volumes = deque(maxlen=self.silence_frames)
        self.is_recording_started = False
        self.speech_detected = False
    
    def _calculate_volume(self, audio_data):
        """Safely compute RMS volume"""
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Ensure the array is not empty
            if len(audio_array) == 0:
                return 0.0
            
            # Compute volume (RMS) with protection against invalid values
            mean_square = np.mean(audio_array.astype(np.float64) ** 2)
            
            # Ensure the value is valid
            if np.isnan(mean_square) or np.isinf(mean_square) or mean_square < 0:
                return 0.0
            
            volume = np.sqrt(mean_square)
            
            # Validate the final result
            if np.isnan(volume) or np.isinf(volume):
                return 0.0
                
            return float(volume)
            
        except Exception as e:
            print(f"Error while computing volume: {e}")
            return 0.0
    
    def process_audio_chunk(self, audio_data):
        """
        Analyze an audio chunk and decide whether there is silence.
        Returns True if recording should continue, False to stop.
        """
        # Compute volume safely
        volume = self._calculate_volume(audio_data)
        self.recent_volumes.append(volume)
        
        # Detect if speech starts
        if volume > self.silence_threshold:
            self.speech_detected = True
            self.is_recording_started = True
        
        # If speech hasn't started yet, keep recording
        if not self.speech_detected:
            return True
        
        # Check if all recent frames are below the threshold
        if len(self.recent_volumes) >= self.silence_frames:
            if all(vol < self.silence_threshold for vol in self.recent_volumes):
                return False  # Stop recording
        
        return True  # Continue recording
    
    def reset(self):
        """Reset the detector for a new recording"""
        self.recent_volumes.clear()
        self.is_recording_started = False
        self.speech_detected = False


class AudioRecorder:
    """Main class for audio recording"""
    
    def __init__(self):
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
        
        # Components
        self.mic_selector = MicrophoneSelector()
        self.silence_detector = SilenceDetector(
            silence_threshold=500, 
            silence_duration=1.0,
            sample_rate=self.sample_rate
        )
        
        # Callbacks
        self.on_recording_start = None
        self.on_recording_stop = None
        self.on_volume_update = None
    
    def set_microphone(self, mic_index):
        """Set the microphone to use"""
        self.microphone_index = mic_index
    
    def set_silence_settings(self, threshold, duration):
        """Configure silence detection parameters"""
        self.silence_detector.silence_threshold = threshold
        self.silence_detector.silence_duration = duration
        self.silence_detector.silence_frames = int(duration * self.sample_rate / self.chunk_size)
        self.silence_detector.recent_volumes = deque(maxlen=self.silence_detector.silence_frames)
    
    def start_recording(self):
        with self._recording_lock:
            if self.is_recording:
                return False
            
            if self.microphone_index is None:
                print("Error: No microphone selected")
                return False
            
            self.is_recording = True
            self._stop_requested = False
            self.audio_data = []
            self.silence_detector.reset()
            
            try:  # 🔒 NEW: Error handling
                self.recording_thread = threading.Thread(target=self._record_audio)
                self.recording_thread.daemon = True
                self.recording_thread.start()
                
                self._safe_callback(self.on_recording_start)
                
                return True
            except Exception as e:  # 🔒 NEW: Error handling
                print(f"Error starting the recording thread: {e}")
                self.is_recording = False
                return False
    
    def stop_recording(self):
        """Stop audio recording"""
        print("🛑 Stop request for recording...")
    
        with self._recording_lock:  # 🔒 NEW: Lock
            if not self.is_recording:
                print("ℹ️ No recording in progress")
                return
            
            self._stop_requested = True
            self.is_recording = False

        if self.recording_thread and self.recording_thread.is_alive():
            print("⏳ Waiting for the recording thread to finish...")
            self.recording_thread.join(timeout=3.0)  # Timeout!
            
            if self.recording_thread.is_alive():
                print("⚠️ The recording thread did not stop in time")
            else:
                print("✅ Recording thread stopped cleanly")

    def _safe_callback(self, callback, *args):
        """Call a callback safely"""
        if callback:
            try:
                callback(*args)
            except Exception as e:
                print(f"Error in callback: {e}")
    
    def _record_audio(self):
        """Recording function executed in a thread with full protection"""
        audio = None
        stream = None
        
        try:
            print("🎤 Initializing recording...")
            audio = pyaudio.PyAudio()
            
            stream = audio.open(
                format=self.sample_format,
                channels=self.channels,
                rate=self.sample_rate,
                frames_per_buffer=self.chunk_size,
                input=True,
                input_device_index=self.microphone_index
            )
            
            print("🔴 Recording started")
            
            while True:
                # Check if stopping was requested (thread-safe)
                with self._recording_lock:
                    if self._stop_requested or not self.is_recording:
                        print("🛑 Stop detected in the recording loop")
                        break
                
                try:
                    # Read audio data
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_data.append(data)
                    
                    # Detect silence only if no manual stop was requested
                    with self._recording_lock:
                        if not self._stop_requested:
                            should_continue = self.silence_detector.process_audio_chunk(data)
                            if not should_continue:
                                print("🤫 Silence detected - auto stop")
                                self.is_recording = False
                                break
                    
                    # Notify volume for the UI (safe callback)
                    volume = self.silence_detector._calculate_volume(data)
                    self._safe_callback(self.on_volume_update, volume)
                    
                except Exception as e:
                    print(f"Error while reading audio: {e}")
                    break
            
        except Exception as e:
            print(f"Recording error: {e}")
        finally:
            # Safe cleanup of resources
            print("🧹 Cleaning up audio resources...")
            
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                    print("✅ Audio stream closed")
                except Exception as e:
                    print(f"Error while closing the stream: {e}")
            
            if audio:
                try:
                    audio.terminate()
                    print("✅ PyAudio terminated")
                except Exception as e:
                    print(f"Error while terminating PyAudio: {e}")
            
            # Update final state
            with self._recording_lock:
                self.is_recording = False
            
            print("🏁 Recording thread finished")
            
            # Call the stop callback
            self._safe_callback(self.on_recording_stop)
    
    def save_recording(self, filename):
        """Save the recording to a WAV file"""
        if not self.audio_data:
            return False
        
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(pyaudio.get_sample_size(self.sample_format))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b''.join(self.audio_data))
                print("File saved")
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False
    
    def get_recording_duration(self):
        """Return the recording duration in seconds"""
        if not self.audio_data:
            return 0
        total_frames = len(self.audio_data) * self.chunk_size
        return total_frames / self.sample_rate
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_recording()
        #self.mic_selector.cleanup()
