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
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Keep responses natural and concise."),
            ("human", "{input}")
        ])
        print("LLM ready!")

    def get_response(self, text):
        print("\nThinking...")
        chain = self.prompt | self.chat
        response = chain.invoke({"input": text})
        cleaned = ' '.join(response.content.replace('\n', ' ').split())
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