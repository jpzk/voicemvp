# Voice Chat Agent

A voice-based chat application that allows users to interact with an AI assistant using speech. The application leverages OpenAI's [Whisper small](https://huggingface.co/openai/whisper-small) for accurate speech recognition and [Kokoro-TTS](https://huggingface.co/hexgrad/Kokoro-82M) for natural-sounding voice synthesis. It will use the [LM Studio local model](https://lms.dev/) that needs to be served at http://localhost:1234 to generate responses.

Hacked together with Claude and Cursor. 

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/jpzk/voicemvp.git
   cd voicemvp
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

4. The first run might take a while as it needs to download the models.