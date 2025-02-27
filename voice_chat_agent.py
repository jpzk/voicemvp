from voice_recognition import VoiceRecognizer, select_audio_device
from llm_thinking import LLMThinker
from text_to_speech import TTSGenerator

class VoiceAgent:
    def __init__(self, device_id=1):
        print("\nInitializing Voice Chat Agent...")
        self.recognizer = VoiceRecognizer(device_id)
        self.thinker = LLMThinker()
        self.tts = TTSGenerator(default_voice='af_heart')
        print("Voice Chat Agent initialization complete!")

    def cleanup(self):
        """Clean up all resources."""
        if hasattr(self, 'recognizer'):
            self.recognizer.cleanup()
        if hasattr(self, 'tts'):
            self.tts.cleanup()

    def chat_loop(self):
        print("\nVoice Chat Agent ready! Press Ctrl+C to exit")
        print("Make sure LM Studio is running and the API is active!")
        print("Speak clearly into your microphone. You should see █ when voice is detected.")
        
        try:
            while True:
                try:
                    # Record and transcribe audio
                    audio_data = self.recognizer.record_audio()
                    
                    if audio_data is not None and len(audio_data) > 0:
                        # Convert speech to text
                        text = self.recognizer.transcribe_audio(audio_data)
                        print(f"\nYou said: {text}")
                        
                        if text.lower() in ['quit', 'exit', 'goodbye', 'bye']:
                            print("\nGoodbye!")
                            break
                        
                        if not text.strip():
                            print("\nNo speech detected, trying again...")
                            continue
                        
                        # Get LLM response
                        response = self.thinker.get_response(text)
                        
                        # Convert response to speech
                        print("\nSpeaking...")
                        self.tts.generate_speech(response)
                    else:
                        print("\nNo audio recorded, trying again...")
                    
                except Exception as e:
                    print(f"\nError in conversation loop: {e}")
                    print("Restarting recording...")
                
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"\nError: {e}")
            print("Make sure LM Studio is running and the API is active!")
        finally:
            self.cleanup()

def main():
    device_id = select_audio_device()
    
    agent = VoiceAgent(
        device_id=device_id       # Selected device
    )
    agent.chat_loop()

if __name__ == "__main__":
    main() 