import sounddevice as sd
import numpy as np
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from scipy.io import wavfile

def list_audio_devices():
    devices = sd.query_devices()
    print("\nAvailable input devices:")
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']} (inputs: {device['max_input_channels']})")
            input_devices.append(i)
    return input_devices

def select_audio_device():
    input_devices = list_audio_devices()
    while True:
        try:
            device_id = input("\nEnter the number of the input device to use (or press Enter for default): ").strip()
            if not device_id:
                return None
            device_id = int(device_id)
            if device_id in input_devices:
                return device_id
            print("Invalid device number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

class VoiceRecognizer:
    def __init__(self, silence_threshold=0.01, silence_duration=2, device_id=None):
        self.device_id = device_id
        print("\nInitializing Voice Recognition...")
        self.setup_audio_recording(silence_threshold, silence_duration)
        self.setup_whisper()
        print("Voice Recognition initialization complete!")
        
    def setup_audio_recording(self, silence_threshold, silence_duration):
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.chunk_size = 1024
        self.channels = 1
        self.is_recording = False
        
        if self.device_id is not None:
            device_info = sd.query_devices(self.device_id)
            print(f"\nUsing selected input device: {device_info['name']}")
            sd.default.device = self.device_id
        else:
            print("\nUsing default input device")

    def setup_whisper(self):
        print("Loading Whisper model...")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        print("Whisper model loaded!")

    def record_audio(self):
        print("\nListening... (speak now)")
        print("Volume level: ", end="", flush=True)
        
        audio_data = []
        silence_start = None
        self.is_recording = True
        recording_started = False
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f'Error: {status}')
            audio_chunk = indata[:, 0]  # Get first channel
            volume = np.abs(audio_chunk).mean()
            
            # Print volume indicator
            if volume > self.silence_threshold:
                print("█", end="", flush=True)
                nonlocal recording_started
                recording_started = True
                nonlocal silence_start
                silence_start = None
            else:
                print(".", end="", flush=True)
                if not recording_started:
                    return  # Wait for voice to start recording
                
                nonlocal silence_start
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= self.silence_duration:
                    self.is_recording = False
                    raise sd.CallbackStop()
            
            # Only store audio data if we've started recording
            if recording_started:
                audio_data.append(audio_chunk)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=audio_callback
            ):
                while self.is_recording:
                    sd.sleep(100)  # Sleep to prevent busy-waiting
                print("\nSilence detected, processing speech...")
                
        except KeyboardInterrupt:
            print("\nStopping recording...")
        
        if not audio_data:
            print("\nNo audio recorded!")
            return None
        
        # Combine all audio chunks
        audio_data = np.concatenate(audio_data)
        
        # Save the audio file
        wavfile.write("recorded_audio.wav", self.sample_rate, audio_data)
        print(f"\nRecorded {len(audio_data)} samples")
        
        return audio_data

    def transcribe_audio(self, audio_data):
        # Normalize audio data
        audio_data = audio_data.astype(np.float32)
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Process audio with Whisper
        input_features = self.processor(
            audio_data, 
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features

        if torch.cuda.is_available():
            input_features = input_features.to("cuda")

        # Generate token ids
        predicted_ids = self.model.generate(
            input_features,
            language="<|en|>",
            task="transcribe"
        )
        
        # Decode the token ids to text
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]

        return transcription.strip()

    def cleanup(self):
        pass  # No cleanup needed for sounddevice

if __name__ == "__main__":
    # Example usage
    device_id = select_audio_device()
    recognizer = VoiceRecognizer(
        silence_threshold=0.002,
        silence_duration=1.0,
        device_id=device_id
    )
    
    try:
        while True:
            audio_data = recognizer.record_audio()
            if audio_data is not None:
                text = recognizer.transcribe_audio(audio_data)
                print(f"\nTranscribed text: {text}")
                
                if text.lower() in ['quit', 'exit', 'goodbye', 'bye']:
                    break
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        recognizer.cleanup() 