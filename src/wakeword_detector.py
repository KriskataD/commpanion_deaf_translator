import numpy as np
import pyaudio
from openwakeword.model import Model
from openwakeword.utils import download_models
import threading
import queue
import os
from typing import Callable, Optional, Dict, Any
import logging

class WakeWordDetector:
    """Détecteur modulaire de mot de réveil utilisant openWakeWord."""
    
    def __init__(
        self,
        wakeword_models: list[str] = None,
        inference_framework: str = 'onnx',
        threshold: float = 0.25,
        chunk_size: int = 1280,
        sample_rate: int = 16000,
        channels: int = 1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise le détecteur de mot de réveil.
        
        Args:
            wakeword_models: Liste des modèles à charger (par défaut: ['hey_jarvis'])
            inference_framework: Framework d'inférence ('onnx' ou 'tflite')
            threshold: Seuil de détection (0-1)
            chunk_size: Taille des chunks audio
            sample_rate: Taux d'échantillonnage
            channels: Nombre de canaux audio
            logger: Logger personnalisé
        """
        self.wakeword_models = wakeword_models or ['hey_jarvis']
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialisation du modèle
        self.model = Model(
            wakeword_models=self.wakeword_models,
            inference_framework=inference_framework
        )
        
        # Configuration audio
        self.audio_format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Threading et état
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.callbacks: Dict[str, Callable] = {}
        
    def register_callback(self, wakeword: str, callback: Callable[[str, float], Any]):
        """
        Enregistre une fonction callback pour un mot de réveil spécifique.
        
        Args:
            wakeword: Nom du mot de réveil
            callback: Fonction à appeler (reçoit le mot détecté et le score)
        """
        self.callbacks[wakeword] = callback
        self.logger.info(f"Callback enregistré pour '{wakeword}'")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback pour le stream audio."""
        if self.is_listening:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self):
        """Thread de traitement audio."""
        self.logger.info("Démarrage du traitement audio")
        
        while self.is_listening:
            try:
                # Récupération des données audio
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Conversion en numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Prédiction
                predictions = self.model.predict(audio_array)
                
                # Vérification des détections
                for wakeword, score in predictions.items():
                    if score > self.threshold:
                        self.logger.info(f"Mot de réveil détecté: {wakeword} (score: {score:.2f})")
                        # Appel du callback si disponible
                        if wakeword in self.callbacks:
                            try:
                                self.callbacks[wakeword](wakeword, score)
                            except Exception as e:
                                self.logger.error(f"Erreur dans le callback: {e}")
                        
                        # Réinitialisation pour éviter les détections multiples
                        self.model.reset()
                        
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Erreur dans le traitement audio: {e}")

    @classmethod
    def download_models(self):
        if not os.path.exists("resources/models"):
            print("Download et installation de openWakeWord")
            # One-time download of all pre-trained models (or only select models)
            download_models()
            print("✅ openWakeWord installé avec succès\n")
    
    def start(self):
        """Démarre l'écoute du mot de réveil."""

        print("Démarrage du détecteur de mot de réveil")

        if self.is_listening:
            self.logger.warning("Le détecteur est déjà en cours d'exécution")
            return
        
        self.logger.info("Démarrage du détecteur de mot de réveil")
        self.is_listening = True
        
        # Ouverture du stream audio
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        # Démarrage du thread de traitement
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.start()
        
        self.logger.info("Détecteur démarré et en écoute")
    
    def stop(self):
        """Arrête l'écoute."""
        if not self.is_listening:
            return
        
        self.logger.info("Arrêt du détecteur")
        self.is_listening = False
        
        # Attente de la fin du thread
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        
        # Fermeture du stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Vidage de la queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        self.logger.info("Détecteur arrêté")
    
    def __enter__(self):
        """Context manager pour démarrage automatique."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager pour arrêt automatique."""
        self.stop()
        
    def cleanup(self):
        """Nettoyage des ressources."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
