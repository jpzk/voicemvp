from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class LLMThinker:
    def __init__(self):
        print("\nInitializing LLM Thinking System...")
        self.setup_llm()
        print("LLM Thinking System ready!")
        
    def setup_llm(self):
        self.chat = ChatOpenAI(
            model_name="local-model",
            openai_api_base="http://localhost:1234/v1",
            openai_api_key="not-needed",
            streaming=True
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant engaging in a voice conversation. "
                      "Keep your responses natural and conversational, as if speaking to a friend. "
                      "IMPORTANT: Always respond in a single, flowing sentence without any line breaks "
                      "or paragraphs. Keep responses concise and easy to speak. Avoid using special "
                      "characters, abbreviations, or complex formatting that might affect text-to-speech quality."),
            ("human", "{input}")
        ])

    def clean_response_for_tts(self, text):
        """Clean and format the response to be TTS-friendly."""
        # Remove all types of line breaks
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Remove any markdown formatting
        text = text.replace('*', '').replace('_', '').replace('`', '')
        
        # Remove any list formatting
        text = text.replace('- ', '').replace('* ', '').replace('• ', '')
        
        # Remove any common quote characters
        text = text.replace('"', '').replace('"', '').replace('"', '')
        
        # Remove parenthetical asides (often used for clarification)
        while '(' in text and ')' in text:
            start = text.find('(')
            end = text.find(')')
            if start < end:
                text = text[:start] + text[end + 1:]
            else:
                break
                
        # Ensure proper spacing around punctuation
        text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        
        # Ensure the response ends with proper punctuation
        if not text.rstrip()[-1] in '.!?':
            text = text.rstrip() + '.'
            
        return text.strip()

    def get_response(self, text):
        """Get a response from the LLM and clean it for TTS."""
        print("\nThinking...")
        chain = self.prompt | self.chat
        response = chain.invoke({"input": text})
        cleaned_response = self.clean_response_for_tts(response.content)
        print(f"Assistant: {cleaned_response}")
        return cleaned_response

if __name__ == "__main__":
    # Example usage
    thinker = LLMThinker()
    while True:
        user_input = input("\nEnter your message (or 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit', 'goodbye', 'bye']:
            break
        response = thinker.get_response(user_input) 