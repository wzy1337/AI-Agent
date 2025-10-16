from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
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
        if not tavily_api_key:
            return "❌ Tavily API key not configured. Please add TAVILY_API_KEY to sample.env file."
        
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_api_key)
        
        # Advanced search targeting credible Singapore sources
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=10,
            include_domains=[
                # Singapore Government
                "cpf.gov.sg", "mas.gov.sg", "mom.gov.sg", "singstat.gov.sg",
                "pmo.gov.sg", "dos.gov.sg", "mof.gov.sg", "mti.gov.sg", "mha.gov.sg",
                "mnd.gov.sg", "smartnation.gov.sg",
                # International Government/Official
                "imf.org", "worldbank.org", "oecd.org", "un.org", "adb.org", "bis.org",
                # Singapore News
                "straitstimes.com", "channelnewsasia.com", "businesstimes.com.sg",
                "todayonline.com", "zaobao.sg",
                # International News
                "bloomberg.com", "reuters.com", "ft.com", "economist.com", "wsj.com",
                "nikkei.com", "scmp.com", "cnbc.com", "forbes.com",
                # Research & Think Tanks (Singapore)
                "rsis.edu.sg", "lkyspp.nus.edu.sg", "ips.org.sg", "iseas.edu.sg",
                "nus.edu.sg", "smu.edu.sg", "ntu.edu.sg",
                # Research & Think Tanks (International)
                "brookings.edu", "chathamhouse.org", "csis.org", "cfr.org", "weforum.org",
                "nber.org", "rand.org", "piie.com", "carnegieendowment.org",
                # Financial Institutions (Singapore)
                "gic.com.sg", "temasek.com.sg", "dbs.com", "ocbc.com", "uob.com",
                # Financial Institutions (International)
                "hsbc.com", "jpmorgan.com", "goldmansachs.com", "morganstanley.com",
                # Consulting
                "mckinsey.com", "pwc.com", "deloitte.com", "ey.com", "bcg.com",
                "bain.com", "kpmg.com", "accenture.com",
                # Industry Bodies
                "fintech.org.sg", "sba.org.sg", "sgtech.org.sg", "siatp.org.sg"
            ],
            include_answer=True,
            include_raw_content=False
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


