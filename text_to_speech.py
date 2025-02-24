import torch
from kokoro import KModel, KPipeline
import sounddevice as sd
import numpy as np

class TTSGenerator:
    def __init__(self, default_voice='af_heart'):
        print("Initializing TTS...")
        self.model = KModel().to('cpu').eval()
        self.pipeline = KPipeline(lang_code='a', model=False)
        self.voice_pack = self.pipeline.load_voice(default_voice)
        print("TTS ready!")
    
    def generate_speech(self, text):
        try:
            for _, ps, _ in self.pipeline(text, 'af_heart', 1):
                ref_s = self.voice_pack[len(ps)-1]
                audio = self.model(ps, ref_s, 1)
                audio_data = audio.numpy().astype(np.float32)
                sd.play(audio_data, samplerate=24000, blocking=True)
                sd.wait()  # Wait until audio is finished playing
        except KeyboardInterrupt:
            sd.stop()
            
    def cleanup(self):
        pass  # No cleanup needed for sounddevice

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