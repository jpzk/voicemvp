# Local Voice Chat MVP with Tool-use and Memory

A voice-based chat application that allows users to interact with an AI assistant using speech. The application leverages OpenAI's [Whisper small](https://huggingface.co/openai/whisper-small) for accurate speech recognition and [Kokoro-TTS](https://huggingface.co/hexgrad/Kokoro-82M) for natural-sounding voice synthesis. You can use a [LM Studio local model](https://lms.dev/) that needs to be served at http://localhost:1234 to generate response. Requires meta-llama-3.1-8b-instruct for tool-use. Works great on Macbook M1 Pro, but Linux works too.  

**Use python3.12 to run this application, because of PyTorch dependency**.

Hacked together with Claude and Cursor. 

## Quick Start

1. Clone the repository:
   ```bash
   $ git clone https://github.com/jpzk/voicemvp.git
   $ cd voicemvp
   ```

2. Install system dependencies (macOS):
   ```bash
   $ brew install portaudio
   ```

3. Install dependencies:
   ```bash
   $ python3.12 -m venv env
   $ source env/bin/activate
   $ python3.12 -m pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   $ python3.12 voice_chat_agent.py
   ```

5. The first run might take a while as it needs to download the models.
