from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import initialize_agent, AgentType

def get_openai_client():
    """
    Returns a ChatOpenAI instance configured with environment variables if OPENAI_API_KEY
    is present, otherwise uses local parameters.
    
    Returns:
        ChatOpenAI: Configured OpenAI client
    """
    import os
   
    # Check if OPENAI_API_KEY exists in environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key:
        # Use API key from environment variables
        print("Using OpenAI API with key from environment variables")
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=api_key
        )
    else:
        # Use local parameters as before
        print("Using local OpenAI parameters")
        return ChatOpenAI(
            base_url="http://192.168.1.6:1234/v1",
            model="meta-llama-3.1-8b-instruct",
            api_key="your-api-key-here"  # Replace with your actual local key if needed
        )

class LLMThinker:
    def __init__(self):
        print("Initializing LLM...")
        self.chat = get_openai_client()
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create tools
        wikipedia = WikipediaAPIWrapper(wiki_client=None)
        search = DuckDuckGoSearchAPIWrapper()
        
        tools = [
            Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description="Useful for queries about historical facts, general knowledge, and detailed information. Input should be a search query."
            ),
            Tool(
                name="DuckDuckGo",
                func=search.run,
                description="Useful for queries about current events, news, and real-time information. Input should be a search query."
            )
        ]
        
        # Initialize the agent with a simpler setup
        self.agent_executor = initialize_agent(
            tools=tools,
            llm=self.chat,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
        print("LLM ready!")

    def get_response(self, text):
        print("\nThinking...")
        try:
            response = self.agent_executor.invoke({"input": text})
            cleaned = ' '.join(response['output'].replace('\n', ' ').split())
            print(f"Assistant: {cleaned}")
            return cleaned
        except Exception as e:
            error_msg = "I apologize, but I encountered an error. Could you rephrase your question?"
            print(f"Assistant: {error_msg}")
            return error_msg

if __name__ == "__main__":
    # Example usage
    thinker = LLMThinker()
    while True:
        user_input = input("\nEnter your message (or 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit', 'goodbye', 'bye']:
            break
        response = thinker.get_response(user_input) 