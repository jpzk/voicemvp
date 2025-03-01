import torch
from kokoro import KModel, KPipeline
import sounddevice as sd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('text_to_speech')

class TTSGenerator:
    def __init__(self, default_voice='af_heart'):
        logger.info("Initializing TTS...")
        self.model = KModel().to('cpu').eval()
        self.pipeline = KPipeline(lang_code='a', model=False)
        self.voice_pack = self.pipeline.load_voice(default_voice)
        logger.info("TTS ready!")
    
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
