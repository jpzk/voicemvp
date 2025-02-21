import torch
from kokoro import KModel, KPipeline
import pyaudio
import numpy as np

class TTSGenerator:
    def __init__(self, default_voice='af_heart'):
        print("Initializing TTS...")
        self.model = KModel().to('cpu').eval()
        self.pipeline = KPipeline(lang_code='a', model=False)
        self.voice_pack = self.pipeline.load_voice(default_voice)
        self.p = pyaudio.PyAudio()
        print("TTS ready!")
    
    def generate_speech(self, text):
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=24000,
            output=True
        )
        
        try:
            for _, ps, _ in self.pipeline(text, 'af_heart', 1):
                ref_s = self.voice_pack[len(ps)-1]
                audio = self.model(ps, ref_s, 1)
                stream.write(audio.numpy().astype(np.float32).tobytes())
        finally:
            stream.stop_stream()
            stream.close()
            
    def cleanup(self):
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