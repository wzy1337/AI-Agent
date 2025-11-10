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

        end_date = datetime.now()
        start_date = datetime(2024, 1, 1)

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
            
            include_domains=[
            # ========================================
            # TIER 1: SINGAPORE CORE (Unchanged)
            # ========================================
            "cpf.gov.sg", "mas.gov.sg", "mom.gov.sg", "singstat.gov.sg",
            "pmo.gov.sg", "dos.gov.sg", "mof.gov.sg", "mti.gov.sg", "mha.gov.sg",
            "mnd.gov.sg", "smartnation.gov.sg", "moh.gov.sg", "msf.gov.sg",
            "straitstimes.com", "channelnewsasia.com", "businesstimes.com.sg",
            "todayonline.com", "zaobao.sg", "thenewpaper.sg",
            
            # ========================================
            # TIER 2: WEAK SIGNALS & ALTERNATIVE MEDIA (EXPANDED)
            # ========================================
            # Singapore Alternative
            "ricemedia.co", "mothership.sg", "yahoo.com/news/singapore",
            "theonlinecitizen.com", "trs.sg", "theindependent.sg",
            # Forums & Social
            "reddit.com", "hardwarezone.com.sg",
            # International Alternative Media
            "medium.com", "substack.com", "substackcdn.com",
            "blog.google", "blogs.imf.org", "blogs.worldbank.org",
            # Think Pieces & Opinion
            "project-syndicate.org", "foreignaffairs.com", "foreignpolicy.com",
            "newstatesman.com", "spectator.co.uk", "newrepublic.com",
            
            # ========================================
            # TIER 3: GLOBAL PENSION SPECIALISTS (EXPANDED)
            # ========================================
            "pensionsage.com", "ipe.com", "pionline.com",
            "top1000funds.com", "institutionalinvestor.com",
            "thepensionsregulator.gov.uk", "pensionpolicyinternational.com",
            "globalaging.org", "gapensionsummit.com",
            "worldpensionscouncil.com", "fiduciary-investors.com",
            
            # ========================================
            # TIER 4: GLOBAL GOVERNMENTS & PENSION FUNDS (EXPANDED)
            # ========================================
            # Asia-Pacific
            "kwsp.gov.my", "bnm.gov.my", "treasury.gov.my",
            "mpfa.org.hk", "hkma.gov.hk", "fstb.gov.hk",
            "treasury.gov.au", "apra.gov.au", "ato.gov.au",
            "mhlw.go.jp", "gpif.go.jp",
            "nps.or.kr", "moel.go.kr",
            "nssf.gov.kh", "socso.gov.my",
            # North America
            "ssa.gov", "pbgc.gov", "dol.gov", "treasury.gov",
            "canada.ca", "osfi-bsif.gc.ca", "cppinvestments.com",
            # Europe
            "gov.uk", "pensionsregulator.gov.uk",
            "service-public.fr", "oecd.org/pensions",
            "bundesregierung.de", "dnb.nl",
            # Nordics (Innovation Leaders)
            "pensionsmyndigheten.se", "etk.fi", "nav.no",
            "borger.dk", "atp.dk",
            
            # ========================================
            # TIER 5: INTERNATIONAL INSTITUTIONS (EXPANDED)
            # ========================================
            "imf.org", "worldbank.org", "oecd.org", "adb.org",
            "bis.org", "un.org", "weforum.org",
            "ilo.org", "unescap.org", "undp.org",
            "issa.int", "iops.org",  # Int'l Social Security / Pension Supervisors
            
            # ========================================
            # TIER 6: GLOBAL THINK TANKS (EXPANDED)
            # ========================================
            # US Think Tanks
            "brookings.edu", "csis.org", "cfr.org", "nber.org",
            "rand.org", "piie.com", "carnegieendowment.org",
            "urban.org", "taxpolicycenter.org", "crfb.org",
            "americanprogress.org", "aei.org", "heritage.org",
            # European Think Tanks
            "chathamhouse.org", "bruegel.org", "ceps.eu",
            "cer.eu", "epim.info", "eurozone.europa.eu",
            # Asian Think Tanks
            "rsis.edu.sg", "lkyspp.nus.edu.sg", "ips.org.sg", "iseas.edu.sg",
            "thinkchina.sg", "eastasiaforum.org", "lowyinstitute.org",
            
            # ========================================
            # TIER 7: ACADEMIC & RESEARCH (EXPANDED)
            # ========================================
            # Singapore Universities
            "nus.edu.sg", "smu.edu.sg", "ntu.edu.sg", "sutd.edu.sg",
            # Top Global Universities
            "mit.edu", "stanford.edu", "harvard.edu", "yale.edu",
            "oxford.ac.uk", "cambridge.ac.uk", "lse.ac.uk",
            "princeton.edu", "berkeley.edu", "columbia.edu",
            # Journals
            "nature.com", "science.org", "plos.org",
            "frontiersin.org", "mdpi.com", "ssrn.com",
            "papers.ssrn.com", "arxiv.org", "biorxiv.org",
            
            # ========================================
            # TIER 8: GLOBAL MEDIA (EXPANDED)
            # ========================================
            # Premium Business
            "bloomberg.com", "reuters.com", "ft.com", "economist.com",
            "wsj.com", "cnbc.com", "forbes.com", "businessinsider.com",
            "marketwatch.com", "barrons.com", "morningstar.com",
            # Quality News
            "theguardian.com", "bbc.com", "bbc.co.uk",
            "nytimes.com", "washingtonpost.com", "apnews.com",
            "axios.com", "politico.com", "thehill.com",
            # Asia-Pacific
            "scmp.com", "nikkei.com", "japantimes.co.jp",
            "bangkokpost.com", "thestar.com.my", "afr.com",
            
            # ========================================
            # TIER 9: FINANCIAL INSTITUTIONS (EXPANDED)
            # ========================================
            # Singapore
            "gic.com.sg", "temasek.com.sg", "dbs.com", "ocbc.com", "uob.com",
            # Global Asset Managers
            "blackrock.com", "vanguard.com", "statestreet.com",
            "fidelity.com", "schroders.com", "jpmorgan.com",
            "goldmansachs.com", "morganstanley.com", "ubs.com",
            "credit-suisse.com", "amundi.com", "allianzgi.com",
            # Pension Fund Managers
            "calpers.ca.gov", "calstrs.com", "nycers.org",
            "ussif.org", "railpen.com",
            
            # ========================================
            # TIER 10: CONSULTING & ADVISORY (EXPANDED)
            # ========================================
            "mckinsey.com", "pwc.com", "deloitte.com", "ey.com",
            "bcg.com", "bain.com", "kpmg.com", "accenture.com",
            "oliverwyman.com", "mercer.com", "aon.com",
            "willislehman.com", "milliman.com", "cer.eu",
            
            # ========================================
            # TIER 11: TECHNOLOGY & DISRUPTION (NEW)
            # ========================================
            # Tech News
            "techcrunch.com", "wired.com", "technologyreview.com",
            "venturebeat.com", "zdnet.com", "theverge.com",
            "arstechnica.com", "engadget.com",
            # AI & Future of Work
            "openai.com", "anthropic.com", "deepmind.com",
            "futureoflife.org", "iftf.org", "singularityhub.com",
            # Fintech & Web3
            "coindesk.com", "cointelegraph.com", "theblock.co",
            "decrypt.co", "blockworks.co", "a16z.com",
            
            # ========================================
            # TIER 12: HEALTHCARE & LONGEVITY (NEW)
            # ========================================
            "thelancet.com", "nejm.org", "who.int",
            "healthaffairs.org", "kff.org", "commonwealthfund.org",
            "ageing.ox.ac.uk", "ageing.stanford.edu",
            "longevity.technology", "sens.org",
            
            # ========================================
            # TIER 13: CLIMATE & ESG (NEW)
            # ========================================
            "ipcc.ch", "carbonbrief.org", "climatecentral.org",
            "climateaction100.org", "unpri.org", "sasb.org",
            "tcfd-hub.org", "cdp.net", "msci.com/esg",
            
            # ========================================
            # TIER 14: PODCASTS & VIDEO PLATFORMS (NEW)
            # ========================================
            # Note: These may not work well with Tavily, but worth trying
            "youtube.com/watch", "vimeo.com",
            "podcasts.apple.com", "open.spotify.com",
        ],
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


