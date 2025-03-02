from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
import logging

from pydantic import SecretStr
from rag_system import RAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('llm_thinking')

def get_openai_client(base_url):
    import os
    logger.info(f"Using local LLM server at {base_url}")
    return ChatOpenAI(
        base_url=base_url,
        model="meta-llama-3.1-8b-instruct",
        api_key=SecretStr("not-needed")
    )

class LLMThinker:
    def __init__(self, llm_server, rag_directory=None, use_rag=True):
        logger.info("Initializing LLM...")
        self.chat = ChatOpenAI(
            base_url=llm_server,
            model="meta-llama-3.1-8b-instruct",
            api_key=SecretStr("not-needed")
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create standard tools
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
        
        # Add RAG tool if enabled
        if use_rag:
            try:
                logger.info(f"Initializing RAG system from {rag_directory}")
                self.rag_system = RAGSystem(base_url=llm_server, vectorstore=rag_directory)
                rag_tool = self.rag_system.get_retrieval_tool()
                tools.append(rag_tool)
                logger.info("RAG system initialized and tool added")
            except Exception as e:
                logger.error(f"Failed to initialize RAG system: {e}")
        
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
        logger.info("LLM ready!")

    def get_response(self, text):
        logger.info("Thinking...")
        try:
            response = self.agent_executor.invoke({"input": text})
            cleaned = ' '.join(response['output'].replace('\n', ' ').split())
            logger.info(f"Response: {cleaned[:100]}...")
            return cleaned
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            error_msg = "I apologize, but I encountered an error. Could you rephrase your question?"
            return error_msg
