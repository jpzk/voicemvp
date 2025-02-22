from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import tool, AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory

class LLMThinker:
    def __init__(self):
        print("Initializing LLM...")
        self.chat = ChatOpenAI(
            model_name="qwen2.5-7b-instruct",
            openai_api_base="http://localhost:1234/v1",
            disable_streaming=True,
            openai_api_key="not-needed"  # Remove tools from constructor
        )
        
        # Initialize tools
        self.wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        self.ddg_search = DuckDuckGoSearchRun()
        
        # Initialize conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Updated system prompt for tool usage
        self.system_prompt = """You are a helpful AI assistant
     
        You have access to two tools for real-time information and live searches:
        1. search_wikipedia: for searching on wikipedia
        2. search_web: for searching the web

        ALWAYS use a tool, you MUST respond with the following format:
        <tool_call>
        {"name": "<tool-name>", query": "<tool-argument>" }
        </tool_call>

        The human will call the tool and then you will respond with the result.
        incorporate the information naturally into your response.
        """

        # Create the agent with tools
        tools = [self.search_wikipedia, self.search_web]
        prompt = ChatPromptTemplate.from_messages([
            #("placeholder", "{chat_history}"),
            ("system", self.system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        self.agent = create_openai_functions_agent(self.chat, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            max_iterations=2  # Limit tool usage iterations
        )
        
        print("LLM ready!")

    @tool
    def search_wikipedia(self, query: str) -> str:
        """Search Wikipedia for encyclopedic information about a topic."""
        print(f"\n[Tool Used] Searching Wikipedia for: {query}")
        try:
            result = self.wikipedia.run(query)
            print("[Tool Success] Found Wikipedia information")
            return result
        except Exception as e:
            error_msg = f"Sorry, I couldn't find that information on Wikipedia. Error: {str(e)}"
            print(f"[Tool Error] {error_msg}")
            return error_msg

    @tool
    def search_web(self, query: str) -> str:
        """Search the web using DuckDuckGo for current information."""
        print(f"\n[Tool Used] Searching DuckDuckGo for: {query}")
        try:
            result = self.ddg_search.run(query)
            print("[Tool Success] Found web search results")
            return result
        except Exception as e:
            error_msg = f"Sorry, I couldn't search the web for that information. Error: {str(e)}"
            print(f"[Tool Error] {error_msg}")
            return error_msg

    def get_response(self, text):
        print("\nThinking...")
        
        # Get initial response
        response = self.chat.invoke(text)
        content = response.content
        
        print("Debug: ", content)

        # Check for tool call
        if "<tool_call>" in content and "</tool_call>" in content:
            # Extract the tool call JSON
            tool_start = content.find("<tool_call>") + len("<tool_call>")
            tool_end = content.find("</tool_call>")
            tool_json_str = content[tool_start:tool_end].strip()
            
            try:
                import json
                tool_request = json.loads(tool_json_str)
                
                # Call the appropriate tool
                if tool_request["name"] == "search_wikipedia":
                    tool_output = self.search_wikipedia(tool_request["query"])
                elif tool_request["name"] == "search_web":
                    tool_output = self.search_web(tool_request["query"])
                else:
                    tool_output = "Error: Unknown tool requested"
                
                # Get final response with tool output
                final_response = self.chat.invoke(
                    f"Here is the information you requested: {tool_output}\n\n"
                    "Please provide a natural, conversational response incorporating this information."
                )
                cleaned = ' '.join(final_response.content.replace('\n', ' ').split())
            
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[Error] Failed to process tool call: {e}")
                cleaned = ' '.join(content.replace('\n', ' ').split())
        else:
            # No tool call, use original response
            cleaned = ' '.join(content.replace('\n', ' ').split())
        
        # Store in conversation history
        self.memory.save_context({"input": text}, {"output": cleaned})
        
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