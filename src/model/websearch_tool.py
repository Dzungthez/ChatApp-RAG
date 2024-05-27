from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '../../config/.env')
load_dotenv(dotenv_path)
api_key = os.getenv('TAVILY_API_KEY')


def get_websearch_tool(top_k=3):
    return TavilySearchResults(k=top_k, api_key=api_key)


def test():
    tool = get_websearch_tool()
    results = tool.invoke({"query": "What is the capital of France?"})
    print(results)


if __name__ == "__main__":
    test()
