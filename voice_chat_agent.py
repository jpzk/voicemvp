from voice_recognition import VoiceRecognizer, select_audio_device
from llm_thinking import LLMThinker
from text_to_speech import TTSGenerator
import argparse
import logging
from rag_system import process_documents, query_documents

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('voice_chat_agent')

class VoiceAgent:
    def __init__(self, llm_server, device_id=None, use_rag=True, rag_directory=None):
        logger.info("Initializing Voice Chat Agent...")
        # Use default device (-1) if device_id is None
        device_id_int = int(device_id) if device_id is not None else -1
        self.recognizer = VoiceRecognizer(device_id=device_id_int)
        self.thinker = LLMThinker(llm_server=llm_server, use_rag=use_rag, rag_directory=rag_directory)
        self.tts = TTSGenerator(default_voice='af_heart')
        logger.info("Voice Chat Agent initialization complete!")

    def cleanup(self):
        """Clean up all resources."""
        if hasattr(self, 'recognizer'):
            self.recognizer.cleanup()
        if hasattr(self, 'tts'):
            self.tts.cleanup()

    def chat_loop(self):
        logger.info("Voice Chat Agent ready! Press Ctrl+C to exit")
        logger.info("Make sure LM Studio is running and the API is active!")
        logger.info("Speak clearly into your microphone. You should see █ when voice is detected.")
        logger.info("You can ask questions about your documents or general knowledge.")
        
        try:
            while True:
                try:
                    # Record and transcribe audio
                    audio_data = self.recognizer.record_audio()
                    
                    if audio_data is not None and len(audio_data) > 0:
                        # Convert speech to text
                        text = self.recognizer.transcribe_audio(audio_data)
                        logger.info(f"You said: {text}")
                        
                        if text.lower() in ['quit', 'exit', 'goodbye', 'bye']:
                            logger.info("Goodbye!")
                            self.tts.generate_speech("Goodbye!")
                            break
                        
                        if not text.strip():
                            logger.info("No speech detected, trying again...")
                            continue
                        
                        # Get LLM response
                        response = self.thinker.get_response(text)
                        
                        # Convert response to speech
                        logger.info("Speaking...")
                        self.tts.generate_speech(response)
                    else:
                        logger.info("No audio recorded, trying again...")
                    
                except Exception as e:
                    logger.error(f"Error in conversation loop: {e}")
                    logger.info("Restarting recording...")
                
        except KeyboardInterrupt:
            logger.info("Exiting...")
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error("Make sure LM Studio is running and the API is active!")
        finally:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(
        description="Voice Chat Agent with RAG capabilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add global arguments
    parser.add_argument("--llm-server", required=True, help="URL of the LLM server (e.g., http://localhost:1234/v1)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process command
    process_parser = subparsers.add_parser(
        "process", 
        help="Process markdown files and store in vector database"
    )
    process_parser.add_argument(
        "directory", 
        help="Directory containing markdown files"
    )
    process_parser.add_argument(
        "--vectorstore", 
        help="Directory to persist vector store",
        required=True
    )
    process_parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000, 
        help="Chunk size for splitting documents"
    )
    process_parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=200, 
        help="Chunk overlap for splitting documents"
    )
    # Query command
    query_parser = subparsers.add_parser(
        "query", 
        help="Query the vector database directly"
    )
    query_parser.add_argument(
        "query", 
        help="Query string"
    )
    query_parser.add_argument(
        "--vectorstore", 
        help="Directory of vector store",
        required=True
    )
    query_parser.add_argument(
        "--k", 
        type=int, 
        default=5, 
        help="Number of documents to retrieve"
    )
    # Chat command
    chat_parser = subparsers.add_parser(
        "chat", 
        help="Start interactive voice chat with the agent"
    )
    chat_parser.add_argument(
        "--vectorstore", 
        help="Directory of vector store",
        required=True
    )
    chat_parser.add_argument(
        "--no-rag", 
        action="store_true", 
        help="Disable RAG capabilities"
    )
    
    args = parser.parse_args()
    
    if args.command == "process":
        process_documents(args)
    elif args.command == "query":
        query_documents(args)
    elif args.command == "chat" or args.command is None:
        # Default to chat if no command is specified
        device_id = select_audio_device()
        
        agent = VoiceAgent(
            llm_server=args.llm_server,
            device_id=device_id,
            use_rag=not getattr(args, 'no_rag', False),
            rag_directory=getattr(args, 'vectorstore')
        )
        agent.chat_loop()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 