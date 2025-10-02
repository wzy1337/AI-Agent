from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, SerpAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="sample.env")

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# SerpAPI tool for Google search results
# Load environment variables and create SerpAPI wrapper
load_dotenv(dotenv_path="sample.env")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

if serpapi_api_key:
    serpapi_wrapper = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    serpapi_tool = Tool(
        name="google_search",
        func=serpapi_wrapper.run,
        description="Search Google for current information and news using SerpAPI. Provides more structured results than DuckDuckGo.",
    )
else:
    print("Warning: SERPAPI_API_KEY not found. SerpAPI tool will not be available.")
    serpapi_tool = None

