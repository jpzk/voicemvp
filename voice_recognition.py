import pyaudio
import numpy as np
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from scipy.io import wavfile

def list_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    print("\nAvailable input devices:")
    input_devices = []
    for i in range(num_devices):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0:
            print(f"{i}: {device_info.get('name')} (inputs: {device_info.get('maxInputChannels')})")
            input_devices.append(i)
    p.terminate()
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
        self.audio_format = pyaudio.paFloat32
        self.channels = 1
        self.is_recording = False
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        if self.device_id is not None:
            device_info = self.p.get_device_info_by_index(self.device_id)
            print(f"\nUsing selected input device: {device_info.get('name')}")
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
        
        # Open stream in blocking mode
        stream = self.p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_id,
            frames_per_buffer=self.chunk_size
        )
        
        audio_data = []
        silence_start = None
        self.is_recording = True
        recording_started = False
        
        try:
            while self.is_recording:
                # Read chunk of audio data
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                # Calculate volume
                volume = np.abs(audio_chunk).mean()
                
                # Print volume indicator
                if volume > self.silence_threshold:
                    print("█", end="", flush=True)
                    recording_started = True
                    silence_start = None
                else:
                    print(".", end="", flush=True)
                    if not recording_started:
                        continue  # Wait for voice to start recording
                    
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= self.silence_duration:
                        print("\nSilence detected, processing speech...")
                        break
                
                # Only store audio data if we've started recording
                if recording_started:
                    audio_data.append(audio_chunk)
                
        except KeyboardInterrupt:
            print("\nStopping recording...")
        finally:
            stream.stop_stream()
            stream.close()
        
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
        if hasattr(self, 'p'):
            self.p.terminate()

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