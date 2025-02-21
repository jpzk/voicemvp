import torch
from kokoro import KModel, KPipeline
import pyaudio
import numpy as np
import time

class TTSGenerator:
    def __init__(self, default_voice='af_heart'):
        """Initialize the TTS generator with models and default settings."""
        print("\nInitializing TTS system...")
        # Initialize model and pipeline
        self.model = KModel().to('cpu').eval()  # Using CPU for simplicity
        self.pipelines = {}
        self.voice_packs = {}
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Pre-load the default voice
        self.load_voice(default_voice)
        print("TTS system ready!")
        
    def load_voice(self, voice):
        """Load a voice if it hasn't been loaded before."""
        lang_code = voice[0]
        
        # Create pipeline for this language if needed
        if lang_code not in self.pipelines:
            self.pipelines[lang_code] = KPipeline(lang_code=lang_code, model=False)
            
        # Load voice pack if needed
        if voice not in self.voice_packs:
            self.voice_packs[voice] = self.pipelines[lang_code].load_voice(voice)
            
        return self.pipelines[lang_code], self.voice_packs[voice]
    
    def play_audio(self, audio_data, sample_rate=24000):
        """Play audio using PyAudio."""
        # Open stream
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            output=True
        )
        
        try:
            # Play the audio
            stream.write(audio_data.astype(np.float32).tobytes())
        finally:
            # Clean up
            stream.stop_stream()
            stream.close()
    
    def generate_speech(self, text, voice='af_heart', speed=1):
        """Generate and play speech from text."""
        pipeline, pack = self.load_voice(voice)
        
        # Generate audio
        for _, ps, _ in pipeline(text, voice, speed):
            ref_s = pack[len(ps)-1]
            audio = self.model(ps, ref_s, speed)
            
            # Convert to numpy array and play
            audio_numpy = audio.numpy()
            self.play_audio(audio_numpy)
            
            return audio_numpy
            
    def cleanup(self):
        """Clean up PyAudio resources."""
        if hasattr(self, 'p'):
            self.p.terminate()

def create_tts_generator():
    """Create and return a TTSGenerator instance."""
    return TTSGenerator()

if __name__ == "__main__":
    # Example usage
    tts = TTSGenerator()
    try:
        tts.generate_speech("Hello, this is a test of the text to speech system.")
    finally:
        tts.cleanup() 