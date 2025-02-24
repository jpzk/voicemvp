from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

class LLMThinker:
    def __init__(self):
        print("Initializing LLM...")
        self.chat = ChatOpenAI(
            model_name="local-model",
            openai_api_base="http://localhost:1234/v1",
            openai_api_key="not-needed",
            streaming=True
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # System prompt remains constant
        self.system_prompt = """You are a helpful AI assistant engaging in natural conversation. 
        Your responses should be casual and conversational, suitable for being read aloud.
        Avoid using markdown, special formatting, or structured lists.
        Keep responses flowing naturally as if speaking to a friend.
        Use conversation history to maintain context and have a more natural dialogue.
        Phrase things in a way that sounds natural when spoken."""
        
        # Create the agent with tools (empty list for now, can be extended later)
        tools = []
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.system_prompt}\nPrevious conversation:\n{{chat_history}}"),
            ("human", "{input}")
        ])
        
        self.agent = create_react_agent(self.chat, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            memory=self.memory,
            verbose=True
        )
        print("LLM ready!")

    def get_response(self, text):
        print("\nThinking...")
        
        # Get response using agent executor
        response = self.agent_executor.invoke({"input": text})
        cleaned = ' '.join(response['output'].replace('\n', ' ').split())
            
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