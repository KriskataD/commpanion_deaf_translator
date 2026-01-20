import pyaudio
import wave
import threading
import numpy as np
from collections import deque

class MicrophoneSelector:
    """Classe pour détecter et gérer les microphones disponibles"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.microphones = []
        self._detect_microphones()
    
    def _detect_microphones(self):
        """Détecte tous les microphones disponibles"""
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
        """Retourne la liste des microphones disponibles"""
        return self.microphones
    
    def get_default_microphone(self):
        """Retourne le microphone par défaut"""
        if self.microphones:
            return self.microphones[0]
        return None
    
    def cleanup(self):
        """Nettoie les ressources PyAudio"""
        self.audio.terminate()


class SilenceDetector:
    """Classe pour détecter le silence dans l'audio"""
    
    def __init__(self, silence_threshold=1000, silence_duration=2.0, sample_rate=44100):
        self.silence_threshold = silence_threshold  # Seuil d'amplitude pour détecter le silence
        self.silence_duration = silence_duration    # Durée de silence avant arrêt (secondes)
        self.sample_rate = sample_rate
        self.silence_frames = int(silence_duration * sample_rate / 1024)  # Nombre de frames de silence
        self.recent_volumes = deque(maxlen=self.silence_frames)
        self.is_recording_started = False
        self.speech_detected = False
    
    def _calculate_volume(self, audio_data):
        """Calcule le volume RMS de manière sécurisée"""
        try:
            # Convertir les données audio en numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Vérifier que le array n'est pas vide
            if len(audio_array) == 0:
                return 0.0
            
            # Calculer le volume (RMS) avec protection contre les valeurs invalides
            mean_square = np.mean(audio_array.astype(np.float64) ** 2)
            
            # Vérifier que la valeur est valide
            if np.isnan(mean_square) or np.isinf(mean_square) or mean_square < 0:
                return 0.0
            
            volume = np.sqrt(mean_square)
            
            # Vérifier le résultat final
            if np.isnan(volume) or np.isinf(volume):
                return 0.0
                
            return float(volume)
            
        except Exception as e:
            print(f"Erreur dans le calcul du volume: {e}")
            return 0.0
    
    def process_audio_chunk(self, audio_data):
        """
        Analyse un chunk audio et détermine s'il y a du silence
        Retourne True si l'enregistrement doit continuer, False pour arrêter
        """
        # Calculer le volume de manière sécurisée
        volume = self._calculate_volume(audio_data)
        self.recent_volumes.append(volume)
        
        # Détecter si on commence à parler
        if volume > self.silence_threshold:
            self.speech_detected = True
            self.is_recording_started = True
        
        # Si on n'a pas encore commencé à parler, continuer l'enregistrement
        if not self.speech_detected:
            return True
        
        # Vérifier si toutes les frames récentes sont en dessous du seuil
        if len(self.recent_volumes) >= self.silence_frames:
            if all(vol < self.silence_threshold for vol in self.recent_volumes):
                return False  # Arrêter l'enregistrement
        
        return True  # Continuer l'enregistrement
    
    def reset(self):
        """Remet à zéro le détecteur pour un nouvel enregistrement"""
        self.recent_volumes.clear()
        self.is_recording_started = False
        self.speech_detected = False


class AudioRecorder:
    """Classe principale pour l'enregistrement audio"""
    
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
        
        # Composants
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
        """Définit le microphone à utiliser"""
        self.microphone_index = mic_index
    
    def set_silence_settings(self, threshold, duration):
        """Configure les paramètres de détection de silence"""
        self.silence_detector.silence_threshold = threshold
        self.silence_detector.silence_duration = duration
        self.silence_detector.silence_frames = int(duration * self.sample_rate / self.chunk_size)
        self.silence_detector.recent_volumes = deque(maxlen=self.silence_detector.silence_frames)
    
    def start_recording(self):
        with self._recording_lock:
            if self.is_recording:
                return False
            
            if self.microphone_index is None:
                print("Erreur: Aucun microphone sélectionné")
                return False
            
            self.is_recording = True
            self._stop_requested = False
            self.audio_data = []
            self.silence_detector.reset()
            
            try:  # 🔒 NOUVEAU : Gestion d'erreur
                self.recording_thread = threading.Thread(target=self._record_audio)
                self.recording_thread.daemon = True
                self.recording_thread.start()
                
                self._safe_callback(self.on_recording_start)
                
                return True
            except Exception as e:  # 🔒 NOUVEAU : Gestion d'erreur
                print(f"Erreur lors du démarrage du thread d'enregistrement: {e}")
                self.is_recording = False
                return False
    
    def stop_recording(self):
        """Arrête l'enregistrement audio"""
        print("🛑 Demande d'arrêt de l'enregistrement...")
    
        with self._recording_lock:  # 🔒 NOUVEAU : Lock
            if not self.is_recording:
                print("ℹ️ Aucun enregistrement en cours")
                return
            
            self._stop_requested = True
            self.is_recording = False

        if self.recording_thread and self.recording_thread.is_alive():
            print("⏳ Attente de la fin du thread d'enregistrement...")
            self.recording_thread.join(timeout=3.0)  # Timeout !
            
            if self.recording_thread.is_alive():
                print("⚠️ Le thread d'enregistrement ne s'est pas arrêté dans les temps")
            else:
                print("✅ Thread d'enregistrement arrêté proprement")

    def _safe_callback(self, callback, *args):
        """Appelle un callback de manière sécurisée"""
        if callback:
            try:
                callback(*args)
            except Exception as e:
                print(f"Erreur dans le callback: {e}")
    
    def _record_audio(self):
        """Fonction d'enregistrement exécutée dans un thread avec protection complète"""
        audio = None
        stream = None
        
        try:
            print("🎤 Initialisation de l'enregistrement...")
            audio = pyaudio.PyAudio()
            
            stream = audio.open(
                format=self.sample_format,
                channels=self.channels,
                rate=self.sample_rate,
                frames_per_buffer=self.chunk_size,
                input=True,
                input_device_index=self.microphone_index
            )
            
            print("🔴 Enregistrement démarré")
            
            while True:
                # Vérifier si l'arrêt a été demandé (avec protection thread-safe)
                with self._recording_lock:
                    if self._stop_requested or not self.is_recording:
                        print("🛑 Arrêt détecté dans la boucle d'enregistrement")
                        break
                
                try:
                    # Lire les données audio
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_data.append(data)
                    
                    # Détecter le silence seulement si pas d'arrêt manuel demandé
                    with self._recording_lock:
                        if not self._stop_requested:
                            should_continue = self.silence_detector.process_audio_chunk(data)
                            if not should_continue:
                                print("🤫 Silence détecté - arrêt automatique")
                                self.is_recording = False
                                break
                    
                    # Notifier le volume pour l'interface (callback sécurisé)
                    volume = self.silence_detector._calculate_volume(data)
                    self._safe_callback(self.on_volume_update, volume)
                    
                except Exception as e:
                    print(f"Erreur lors de la lecture audio: {e}")
                    break
            
        except Exception as e:
            print(f"Erreur d'enregistrement: {e}")
        finally:
            # Nettoyage sécurisé des ressources
            print("🧹 Nettoyage des ressources audio...")
            
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                    print("✅ Stream audio fermé")
                except Exception as e:
                    print(f"Erreur lors de la fermeture du stream: {e}")
            
            if audio:
                try:
                    audio.terminate()
                    print("✅ PyAudio terminé")
                except Exception as e:
                    print(f"Erreur lors de la terminaison de PyAudio: {e}")
            
            # Mettre à jour l'état final
            with self._recording_lock:
                self.is_recording = False
            
            print("🏁 Thread d'enregistrement terminé")
            
            # Appeler le callback d'arrêt
            self._safe_callback(self.on_recording_stop)
    
    def save_recording(self, filename):
        """Sauvegarde l'enregistrement dans un fichier WAV"""
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
            print(f"Erreur de sauvegarde: {e}")
            return False
    
    def get_recording_duration(self):
        """Retourne la durée de l'enregistrement en secondes"""
        if not self.audio_data:
            return 0
        total_frames = len(self.audio_data) * self.chunk_size
        return total_frames / self.sample_rate
    
    def cleanup(self):
        """Nettoie les ressources"""
        self.stop_recording()
        #self.mic_selector.cleanup()

