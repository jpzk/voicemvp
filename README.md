# Voice Chat Agent

A voice-based chat application that allows users to interact with an AI assistant using speech. The application leverages OpenAI's Whisper for accurate speech recognition and Kokoro-TTS for natural-sounding voice synthesis. It will use the LLM Studio local model that needs to be served at http://localhost:1234 to generate responses.

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/voice-chat-agent.git
   cd voice-chat-agent
   ```

2. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python voice_chat_agent.py
   ```
