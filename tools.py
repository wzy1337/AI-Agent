from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="sample.env")

# -----------------------------
# Input Schemas (THE FIX)
# These schemas tell the Agent exactly what inputs are allowed
# -----------------------------
class SingleStringInput(BaseModel):
    query: str = Field(description="The search query or input text.")

class FileSaveInput(BaseModel):
    data: str = Field(description="The content to save to the file.")
    filename: Optional[str] = Field(default="research_output.txt", description="The filename to save to.")

class UrlInput(BaseModel):
    url: str = Field(description="The complete URL string to extract data from.")

# -----------------------------
# Save Tool
# -----------------------------
@tool("save_text_to_file", args_schema=FileSaveInput)
def save_tool(data: str, filename: str = "research_output.txt"):
    """Saves structured research data to a text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)
        return f"Data successfully saved to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

# -----------------------------
# Tavily AI Search (PRIMARY SEARCH TOOL)
# -----------------------------
tavily_api_key = os.getenv("TAVILY_API_KEY")

@tool("tavily_search", args_schema=SingleStringInput)
def tavily_tool(query: str) -> str:
    """
    PRIMARY SEARCH TOOL - Advanced AI-powered search that returns COMPLETE article URLs from credible sources.
    Use this FIRST for all research queries.
    Returns structured results with full URLs, publication dates, and relevance scores.
    """
    try:
        # Define dates dynamically
        end_date = datetime.now()
        start_date = datetime(end_date.year - 1, 1, 1)  # Previous year Jan 1
        
        if not tavily_api_key:
            return "❌ Tavily API key not configured. Please add TAVILY_API_KEY to secrets."
        
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_api_key)
        
        # Advanced search targeting credible sources
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=10,
            # Note: Tavily Python SDK expects strings for dates in YYYY-MM-DD
            start_published_date=start_date.strftime("%Y-%m-%d"), 
            end_published_date=end_date.strftime("%Y-%m-%d"),
            include_answer=True,
            include_raw_content=True
        )
        
        # Format results with COMPLETE URLs
        results = []
        for idx, r in enumerate(response.get('results', []), 1):
            result_str = f"\n{'='*80}\n"
            result_str += f"RESULT {idx}\n"
            result_str += f"{'='*80}\n"
            result_str += f"📰 Title: {r.get('title', 'N/A')}\n"
            result_str += f"🔗 COMPLETE URL: {r.get('url', 'N/A')}\n"
            result_str += f"📅 Published: {r.get('published_date', 'Date not available')}\n"
            result_str += f"⭐ Relevance Score: {r.get('score', 'N/A')}\n"
            result_str += f"\n📝 Content Summary:\n{r.get('content', 'N/A')}\n"
            results.append(result_str)
        
        if not results:
            return f"⚠️ No results found from credible sources for: '{query}'"
        
        header = f"\n🔍 TAVILY SEARCH RESULTS FOR: '{query}'\n"
        header += f"📊 Found {len(results)} high-quality results from credible sources\n"
        header += f"⚠️ IMPORTANT: Copy the COMPLETE URLs exactly as shown below\n"
        
        return header + "\n".join(results)
        
    except ImportError:
        return "❌ Tavily library not installed. pip install tavily-python"
    except Exception as e:
        return f"❌ Tavily search error: {str(e)}"

# -----------------------------
# Tavily Extract (METADATA EXTRACTION)
# -----------------------------

def extract_year_from_url(url: str) -> Optional[int]:
    """Helper: Extract publication year from URL patterns (FREE)."""
    # Dynamic year range: previous year to next year
    current_year = datetime.now().year
    min_year = current_year - 1  # Previous year
    max_year = current_year + 1  # Allow slight future dates
    
    # Pattern 1: /YYYY/ in path
    year_match = re.search(r'/(\d{4})/', url)
    if year_match:
        year = int(year_match.group(1))
        if min_year <= year <= max_year: return year
    
    # Pattern 2: YYYY-MM-DD anywhere
    date_match = re.search(r'(\d{4})-\d{2}-\d{2}', url)
    if date_match:
        year = int(date_match.group(1))
        if min_year <= year <= max_year: return year
    
    # Pattern 3: /YYYYMMDD/ format
    compact_date = re.search(r'/(\d{4})\d{4}/', url)
    if compact_date:
        year = int(compact_date.group(1))
        if min_year <= year <= max_year: return year
    
    return None

@tool("tavily_extract", args_schema=UrlInput)
def tavily_extract_tool(url: str) -> str:
    """
    COST-EFFICIENT metadata extraction tool. 
    First tries FREE URL pattern matching to filter old articles, then uses API only when needed.
    Returns publication date, author, source, and full article content.
    """
    try:
        # TIER 1: Try free URL-based date extraction first
        url_year = extract_year_from_url(url)
        tier1_passed = False
        min_year = datetime.now().year - 1  # Dynamic: previous year
        
        if url_year:
            if url_year < min_year:
                return f"❌ ARTICLE REJECTED - URL indicates year {url_year} (before {min_year}). URL: {url}"
            else:
                tier1_passed = True
        
        # TIER 2: Use Tavily Extract API
        if not tavily_api_key:
            return "❌ Tavily API key not configured."
        
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_api_key)
        
        response = client.extract(urls=[url])
        
        if not response or 'results' not in response or not response['results']:
            return f"❌ No content could be extracted from URL: {url}"
        
        result = response['results'][0]
        published_date = result.get('published_date')
        
        # Check Date
        if published_date:
            try:
                article_date = datetime.strptime(published_date, "%Y-%m-%d")
                min_year = datetime.now().year - 1  # Dynamic: previous year
                cutoff_date = datetime(min_year, 1, 1)
                if article_date < cutoff_date:
                    return f"❌ ARTICLE REJECTED - Published {published_date} (before {min_year}). URL: {url}"
            except ValueError:
                pass
        
        # Format Output
        output = f"\n{'='*80}\n✅ ARTICLE EXTRACTED\n{'='*80}\n"
        output += f"🔗 URL: {result.get('url', url)}\n"
        output += f"📰 Title: {result.get('title', 'N/A')}\n"
        output += f"📅 Published: {published_date if published_date else '⚠️ Date not available'}\n"
        
        raw_content = result.get('raw_content', '')
        content_preview = raw_content[:3000] + "..." if len(raw_content) > 3000 else raw_content
        output += f"\n📝 Content:\n{content_preview}\n"
        
        return output
        
    except Exception as e:
        return f"❌ Tavily extract error: {str(e)}"

# -----------------------------
# DuckDuckGo Search (FALLBACK)
# -----------------------------
search_run = DuckDuckGoSearchRun()

@tool("duckduckgo_search", args_schema=SingleStringInput)
def search_tool(query: str):
    """FALLBACK search tool - Use ONLY if tavily_search fails."""
    return search_run.run(query)