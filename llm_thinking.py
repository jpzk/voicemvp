from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_community.utilities import (
    WikipediaAPIWrapper,
    DuckDuckGoSearchAPIWrapper
)

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
        self.system_prompt = """You are a helpful AI assistant engaging in natural \
        conversation. 
        Your responses should be casual and conversational, suitable for being read \
        aloud.
        Avoid using markdown, special formatting, or structured lists.
        Keep responses flowing naturally as if speaking to a friend.
        Use conversation history to maintain context and have a more natural dialogue.
        Phrase things in a way that sounds natural when spoken.
        
        For information retrieval:
        - Use the Wikipedia tool for facts, historical information, and general \
        knowledge
        - Use the DuckDuckGo tool for recent events, news, and current information
        Always summarize the information in a conversational way."""
        
        # Create the agent with Wikipedia and DuckDuckGo tools
        wikipedia = WikipediaAPIWrapper()
        search = DuckDuckGoSearchAPIWrapper()
        
        tools = [
            Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description=(
                    "Use this tool for searching facts, historical information, and "
                    "general knowledge. Input should be a search query. Returns "
                    "detailed information from Wikipedia articles."
                ),
                return_direct=False
            ),
            Tool(
                name="DuckDuckGo",
                func=search.run,
                description=(
                    "Use this tool for searching recent events, news, and current "
                    "information. Input should be a search query. Returns real-time "
                    "search results from the web."
                ),
                return_direct=False
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("system", "Previous conversation:\n{chat_history}"),
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