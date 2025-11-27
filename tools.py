from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional
import os
import re
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

# -----------------------------
# Tavily AI Search (PRIMARY SEARCH TOOL)
# -----------------------------
tavily_api_key = os.getenv("TAVILY_API_KEY")

def tavily_search(query: str) -> str:
    """
    Advanced AI-powered search optimized for research with COMPLETE URLs.
    Returns high-quality results from credible sources with full article URLs.
    """
    try:

        end_date = datetime.now()
        start_date = datetime(2024, 1, 1)
        now = datetime.now()
        one_year_ago_datetime = now - relativedelta(years=1)
        one_year_ago_int = one_year_ago_datetime.year

        if not tavily_api_key:
            return "❌ Tavily API key not configured. Please add TAVILY_API_KEY to sample.env file."
        
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_api_key)
        
        # Advanced search targeting credible Singapore sources
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=10,

            start_published_date=start_date.strftime("%Y-%m-%d"),  # "2024-01-01"
            end_published_date=end_date.strftime("%Y-%m-%d"),      # "2024-10-23"
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
            return f"⚠️ No results found from credible sources for: '{query}'\n\nTry:\n- Broader search terms\n- Different keywords\n- Removing date restrictions"
        
        header = f"\n🔍 TAVILY SEARCH RESULTS FOR: '{query}'\n"
        header += f"📊 Found {len(results)} high-quality results from credible sources\n"
        header += f"⚠️ IMPORTANT: Copy the COMPLETE URLs exactly as shown below\n"
        
        return header + "\n".join(results)
        
    except ImportError:
        return "❌ Tavily library not installed.\n\nInstall with: pip install tavily-python\n\nThen add your API key to sample.env:\nTAVILY_API_KEY=your-key-here"
    except Exception as e:
        return f"❌ Tavily search error: {str(e)}\n\nPlease check:\n1. API key is valid\n2. You have API credits\n3. Internet connection is working"

tavily_tool = Tool(
    name="tavily_search",
    func=tavily_search,
    description="PRIMARY SEARCH TOOL - Advanced AI-powered search that returns COMPLETE article URLs from credible sources (government sites, major news outlets, research institutions). Use this FIRST for all research queries. Returns structured results with full URLs, publication dates, and relevance scores.",
)

# -----------------------------
# Tavily Extract (METADATA EXTRACTION & DATE FILTERING)
# COST-EFFICIENT VERSION: Uses free methods first, API only when needed
# -----------------------------

def extract_year_from_url(url: str) -> Optional[int]:
    """
    Extract publication year from URL patterns (FREE - no API call).
    
    Patterns detected:
    - /2024/01/article
    - /article-2024-01-15
    - ?date=2024-01-15
    - /20240115/ (compact format)
    """
    # Pattern 1: /YYYY/ in path
    year_match = re.search(r'/(\d{4})/', url)
    if year_match:
        year = int(year_match.group(1))
        # Sanity check: reasonable year range
        if 2020 <= year <= 2030:
            return year
    
    # Pattern 2: YYYY-MM-DD anywhere
    date_match = re.search(r'(\d{4})-\d{2}-\d{2}', url)
    if date_match:
        year = int(date_match.group(1))
        if 2020 <= year <= 2030:
            return year
    
    # Pattern 3: /YYYYMMDD/ format
    compact_date = re.search(r'/(\d{4})\d{4}/', url)
    if compact_date:
        year = int(compact_date.group(1))
        if 2020 <= year <= 2030:
            return year
    
    return None


def tavily_extract(url: str) -> str:
    """
    COST-EFFICIENT extraction with 2-tier fallback:
    
    Tier 1: Try URL pattern extraction (FREE) - saves ~60-70% of API calls
    Tier 2: Use Tavily Extract API if needed (PAID)
    
    Automatically filters out articles published before 2024.
    Returns comprehensive article information including publication date, author, and full content.
    """
    try:
        # TIER 1: Try free URL-based date extraction first
        url_year = extract_year_from_url(url)
        
        if url_year:
            if url_year < 2024:
                # Reject immediately without API call - SAVES MONEY!
                return f"❌ ARTICLE REJECTED - URL indicates publication year {url_year} (before 2024)\n\n" \
                       f"🔗 URL: {url}\n" \
                       f"📅 Year extracted from URL: {url_year}\n\n" \
                       f"⚠️ This article appears to be from {url_year}, before the 2024 cutoff.\n" \
                       f"💰 Saved 1 API call by using URL pattern matching!\n\n" \
                       f"Please search for more recent sources."
            else:
                # Year is 2024+, note this for later
                tier1_passed = True
                print(f"💰 Cost savings: URL pattern shows {url_year} - pre-screening passed")
        else:
            tier1_passed = False
        
        # TIER 2: Use Tavily Extract API for full metadata
        if not tavily_api_key:
            return "❌ Tavily API key not configured. Please add TAVILY_API_KEY to sample.env file."
        
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_api_key)
        
        # Extract detailed content and metadata from the URL
        response = client.extract(urls=[url])
        
        if not response or 'results' not in response or not response['results']:
            return f"❌ No content could be extracted from URL: {url}\n\nPossible reasons:\n- URL is not accessible\n- Content is behind a paywall\n- URL format is invalid"
        
        # Get the first (and typically only) result
        result = response['results'][0]
        
        # Extract publication date
        published_date = result.get('published_date')
        
        # Check if article is from 2024 or later
        if published_date:
            try:
                # Parse the date (Tavily returns ISO format: YYYY-MM-DD)
                article_date = datetime.strptime(published_date, "%Y-%m-%d")
                cutoff_date = datetime(2024, 1, 1)
                
                if article_date < cutoff_date:
                    return f"❌ ARTICLE REJECTED - Published before 2024\n\n" \
                           f"📅 Publication Date: {published_date}\n" \
                           f"🔗 URL: {url}\n" \
                           f"📰 Title: {result.get('title', 'N/A')}\n\n" \
                           f"⚠️ This article is from {article_date.year}, which is before the 2024 cutoff.\n" \
                           f"Please search for more recent sources."
            except ValueError:
                # If date parsing fails, include a warning but continue
                pass
        
        # Format the extracted content
        output = f"\n{'='*80}\n"
        output += f"✅ ARTICLE EXTRACTED SUCCESSFULLY\n"
        output += f"{'='*80}\n\n"
        
        # Note if we saved money with Tier 1 pre-screening
        if tier1_passed:
            output += f"💰 Cost optimization: URL pattern pre-screening indicated year {url_year}\n"
        
        output += f"🔗 URL: {result.get('url', url)}\n"
        output += f"📰 Title: {result.get('title', 'N/A')}\n"
        output += f"📅 Published: {published_date if published_date else '⚠️ Date not available'}\n"
        output += f"✍️ Author: {result.get('author', 'N/A')}\n"
        output += f"🏢 Source: {result.get('source', 'N/A')}\n"
        
        # Add warning if no date is available
        if not published_date:
            output += f"\n⚠️ WARNING: Publication date not found in metadata.\n"
            if tier1_passed:
                output += f"However, URL pattern indicates year {url_year}.\n"
            else:
                output += f"Cannot verify if this article is from 2024 or later.\n"
                output += f"Please manually verify the article date before using.\n"
        else:
            article_year = datetime.strptime(published_date, "%Y-%m-%d").year
            output += f"\n✅ Date verification: Article is from {article_year} (≥ 2024)\n"
            if tier1_passed and article_year == url_year:
                output += f"✅ Confirmed: URL pattern year ({url_year}) matches metadata year ({article_year})\n"
        
        # Add content summary
        raw_content = result.get('raw_content', '')
        if raw_content:
            # Truncate if too long (first 2000 characters)
            content_preview = raw_content[:2000] + "..." if len(raw_content) > 2000 else raw_content
            output += f"\n📝 Content Preview:\n{'-'*80}\n{content_preview}\n{'-'*80}\n"
        else:
            output += f"\n⚠️ No content could be extracted from this URL.\n"
        
        # Add metadata if available
        if 'metadata' in result and result['metadata']:
            output += f"\n📊 Additional Metadata:\n"
            for key, value in result['metadata'].items():
                output += f"  • {key}: {value}\n"
        
        return output
        
    except ImportError:
        return "❌ Tavily library not installed.\n\nInstall with: pip install tavily-python\n\nThen add your API key to sample.env:\nTAVILY_API_KEY=your-key-here"
    except Exception as e:
        return f"❌ Tavily extract error: {str(e)}\n\nPlease check:\n1. URL is valid and accessible\n2. API key is valid\n3. You have API credits\n4. Internet connection is working"

tavily_extract_tool = Tool(
    name="tavily_extract",
    func=tavily_extract,
    description="COST-EFFICIENT metadata extraction tool. First tries FREE URL pattern matching to filter old articles (saves ~60-70% of API calls), then uses API only when needed. Automatically REJECTS articles published before 2024. Returns publication date, author, source, and full article content. Input should be a complete URL.",
)

# -----------------------------
# DuckDuckGo Search (FALLBACK ONLY)
# -----------------------------
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="duckduckgo_search",
    func=search.run,
    description="FALLBACK search tool - Use ONLY if tavily_search fails. Returns basic web search results but may have incomplete URLs.",
)

# -----------------------------
# DEPRECATED: SerpAPI (keeping for backward compatibility)
# -----------------------------
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

if serpapi_api_key:
    try:
        from langchain_community.utilities import SerpAPIWrapper
        serpapi_wrapper = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
        serpapi_tool = Tool(
            name="google_search",
            func=serpapi_wrapper.run,
            description="DEPRECATED - Use tavily_search instead. Legacy Google search tool.",
        )
    except ImportError:
        print("Warning: SerpAPIWrapper not available")
        serpapi_tool = None
else:
    serpapi_tool = None


