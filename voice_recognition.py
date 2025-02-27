import sys
from typing import Optional
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from pvrecorder import PvRecorder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def select_audio_device() -> int:
    """Select an audio device and return its index."""
    devices = PvRecorder.get_available_devices()
    print("\nAvailable audio devices:")
    for i, device in enumerate(devices):
        print(f"{i}: {device}")
    
    device_id = -1  # Default device
    try:
        selection = input("\nEnter device number (press Enter for default): ").strip()
        if selection and 0 <= int(selection) < len(devices):
            device_id = int(selection)
    except (ValueError, EOFError):
        pass
    
    return device_id

class VoiceRecognizer:
    def __init__(self, device_id: int = -1, frame_length: int = 512):
        self.device_id = device_id
        self.frame_length = frame_length
        self.sample_rate = 16000  # Required by Whisper
        self.recorder = None
        
        logging.info("Initializing Voice Recognition...")
        self._setup_recorder()
        self._setup_whisper()
        
    def _setup_recorder(self) -> None:
        try:
            self.recorder = PvRecorder(device_index=self.device_id, frame_length=self.frame_length)
            device_name = PvRecorder.get_available_devices()[self.device_id]
            logging.info(f"Using audio device: {device_name}")
        except Exception as e:
            logging.error(f"Error setting up recorder: {e}")
            raise

    def _setup_whisper(self) -> None:
        logging.info("Loading Whisper model...")
        try:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logging.info("Using CUDA for inference")
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            raise

    @staticmethod
    def list_audio_devices():
        devices = PvRecorder.get_available_devices()
        print("\nAvailable audio devices:")
        for i, device in enumerate(devices):
            print(f"{i}: {device}")
        return devices

    def record_audio(self, voice_threshold: float = 150, silence_threshold: float = 140, silence_duration: float = 2.0) -> Optional[np.ndarray]:
        if not self.recorder:
            logging.error("Recorder not initialized")
            return None

        audio_data = []
        silence_count = 0
        max_silence_count = int(silence_duration * self.sample_rate / self.frame_length)
        voice_detected = False

        try:
            self.recorder.start()
            logging.info("Waiting for voice...")
            
            while True:
                try:
                    frame = self.recorder.read()
                    volume = np.abs(np.array(frame, dtype=np.int16)).mean()
                    
                    # Wait for voice to start recording
                    if not voice_detected:
                        print("." if volume < voice_threshold else "!", end="", flush=True)
                        if volume > voice_threshold:
                            voice_detected = True
                            print("\nVoice detected, recording...")
                        continue
                    
                    # Once voice is detected, start recording
                    audio_data.extend(frame)
                    print("█" if volume > silence_threshold else ".", end="", flush=True)
                    
                    # Silence detection after voice
                    if volume < silence_threshold:
                        silence_count += 1
                        if silence_count >= max_silence_count:
                            print("\nSilence detected, stopping...")
                            break
                    else:
                        silence_count = 0
                        
                except (KeyboardInterrupt, EOFError):
                    print("\nRecording interrupted...")
                    sys.exit()
                    return None

        finally:
            try:
                self.recorder.stop()
            except:
                pass

        if not audio_data:
            logging.warning("No audio recorded")
            return None

        return np.array(audio_data, dtype=np.float32) / 32768.0  # Convert to float32 [-1.0, 1.0]

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        try:
            input_features = self.processor(
                audio_data, 
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features

            if torch.cuda.is_available():
                input_features = input_features.to("cuda")

            predicted_ids = self.model.generate(
                input_features,
                language="en",  # Force English language
                task="transcribe"  # Ensure we're doing transcription
            )
            return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return ""

    def cleanup(self):
        if self.recorder:
            self.recorder.delete()

def main():
    recognizer = None
    try:
        # List available devices and get user selection
        devices = VoiceRecognizer.list_audio_devices()
        device_id = -1  # Default device
        
        try:
            selection = input("\nEnter device number (press Enter for default): ").strip()
            if selection and 0 <= int(selection) < len(devices):
                device_id = int(selection)
        except (ValueError, EOFError):
            pass

        # Initialize recognizer
        recognizer = VoiceRecognizer(device_id=device_id)
        
        print("\nCommands:")
        print("- Speak to transcribe")
        print("- Say 'quit', 'exit', 'goodbye', or 'bye' to end")
        print("- Press Ctrl+C or Ctrl+D to stop\n")
        
        while True:
            try:
                audio_data = recognizer.record_audio()
                if audio_data is not None:
                    text = recognizer.transcribe_audio(audio_data)
                    print(f"\nTranscribed: {text}")
                    
                    if text.lower() in {'quit', 'exit', 'goodbye', 'bye'}:
                        break
            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except EOFError:
                print("\nReceived EOF, stopping...")
                break

    except (KeyboardInterrupt, EOFError):
        print("\nProgram terminated by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        if recognizer:
            recognizer.cleanup()
            logging.info("Cleanup complete")

if __name__ == "__main__":
    main() 