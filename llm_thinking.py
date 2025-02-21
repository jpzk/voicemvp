from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class LLMThinker:
    def __init__(self):
        print("Initializing LLM...")
        self.chat = ChatOpenAI(
            model_name="local-model",
            openai_api_base="http://localhost:1234/v1",
            openai_api_key="not-needed",
            streaming=True
        )
        
        # Initialize conversation history
        self.conversation_history = []
        
        # System prompt remains constant
        self.system_prompt = """You are a helpful AI assistant engaging in natural conversation. 
        Your responses should be casual and conversational, suitable for being read aloud.
        Avoid using markdown, special formatting, or structured lists.
        Keep responses flowing naturally as if speaking to a friend.
        Use conversation history to maintain context and have a more natural dialogue.
        Phrase things in a way that sounds natural when spoken."""
        print("LLM ready!")

    def get_response(self, text):
        print("\nThinking...")
        
        # Build messages list starting with system prompt
        messages = [("system", self.system_prompt)]
        
        # Add conversation history
        for message in self.conversation_history:
            messages.append(message)
            
        # Add current user input
        messages.append(("human", text))
        
        # Create prompt template with all messages
        prompt = ChatPromptTemplate.from_messages(messages)
        
        # Get response
        chain = prompt | self.chat
        response = chain.invoke({"input": text})
        cleaned = ' '.join(response.content.replace('\n', ' ').split())
        
        # Store the exchange in conversation history
        self.conversation_history.append(("human", text))
        self.conversation_history.append(("assistant", cleaned))
        
        # Keep only last 10 exchanges (20 messages) to prevent context from growing too large
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
            
        print(f"Assistant: {cleaned}")
        return cleaned

if __name__ == "__main__":
    # Example usage
    thinker = LLMThinker()
    while True:
        user_input = input("\nEnter your message (or 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit', 'goodbye', 'bye']:
            break
        response = thinker.get_response(user_input) 