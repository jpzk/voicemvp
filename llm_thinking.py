from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

class LLMThinker:
    def __init__(self):
        print("Initializing LLM...")
        self.chat = ChatOpenAI(
            model_name="local-model",
            openai_api_base="http://localhost:1234/v1",
            openai_api_key="not-needed",
            streaming=True
        )
        
        # Initialize Wikipedia tool
        wikipedia = WikipediaAPIWrapper()
        self.wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Keep responses natural and concise.
            When you need to look up factual information, you can use the Wikipedia tool.
            To use it, write [WIKI] followed by your search query, then [/WIKI].
            Example: To look up information about Python, write: [WIKI]Python programming language[/WIKI]"""),
            ("human", "{input}")
        ])
        print("LLM ready!")

    def get_response(self, text):
        print("\nThinking...")
        chain = self.prompt | self.chat
        response = chain.invoke({"input": text})
        content = response.content
        
        # Check if response contains Wikipedia queries
        while "[WIKI]" in content and "[/WIKI]" in content:
            start = content.find("[WIKI]") + 6
            end = content.find("[/WIKI]")
            if start < end:
                query = content[start:end]
                try:
                    wiki_result = self.wiki_tool.run(query)
                    # Replace the wiki query with the result
                    content = content[:start-6] + wiki_result + content[end+7:]
                except Exception as e:
                    content = content[:start-6] + f"(Wikipedia search failed: {str(e)})" + content[end+7:]
            else:
                break
        
        cleaned = ' '.join(content.replace('\n', ' ').split())
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