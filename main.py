from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, tavily_tool
import datetime
from dateutil.relativedelta import relativedelta
import json
import re
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# Import API keys from sample.env file
load_dotenv(dotenv_path="sample.env")
 
#Find Current Date for LLM context
now = datetime.now()
one_year_ago_datetime = now - relativedelta(years=1)
one_year_ago_int = one_year_ago_datetime.year
current_datetime_str = now.strftime("%Y-%m-%d, %A. Time: %H:%M:%S. Current timezone is UTC+8.")


# -----------------------------
# URL Date Filtering Function
# -----------------------------
def extract_year_from_url(url: str) -> Optional[int]:
    """
    Extract publication year from URL patterns.
    Returns year as int if found, None otherwise.
    
    Common patterns:
    - /2024/01/article
    - /article-2024-01-15
    - ?date=2024-01-15
    """
    # Pattern 1: /YYYY/ in path
    year_match = re.search(r'/(\d{4})/', url)
    if year_match:
        return int(year_match.group(1))
    
    # Pattern 2: YYYY-MM-DD anywhere
    date_match = re.search(r'(\d{4})-\d{2}-\d{2}', url)
    if date_match:
        return int(date_match.group(1))
    
    # Pattern 3: /YYYYMMDD/ format
    compact_date = re.search(r'/(\d{4})\d{4}/', url)
    if compact_date:
        return int(compact_date.group(1))
    
    # Pattern 4: Query parameters (e.g., ?date=2024-01-15)
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        for key in ['date', 'published', 'publishDate', 'pubdate']:
            if key in params:
                date_str = params[key][0]
                date_match = re.search(r'(\d{4})', date_str)
                if date_match:
                    return int(date_match.group(1))
    except:
        pass
    
    return None

def extract_date_from_tavily_metadata(output: str, url: str) -> Optional[datetime]:
    """
    Extract published date from Tavily search result metadata.
    Looks for the 'Published: YYYY-MM-DD' line after the URL.
    """
    try:
        # Find the section containing this URL
        url_escaped = re.escape(url)
        pattern = rf'🔗 COMPLETE URL: {url_escaped}\s*\n📅 Published: ([^\n]+)'
        match = re.search(pattern, output)
        
        if match:
            date_str = match.group(1).strip()
            if date_str and date_str != 'Date not available':
                # Try to parse the date
                try:
                    return date_parser.parse(date_str)
                except:
                    pass
    except:
        pass
    
    return None

def extract_dates_from_search_output(output: str) -> Dict[str, datetime]:
    """
    Extract all URLs and their published dates from Tavily search output.
    Returns dict mapping URL -> datetime object.
    """
    url_dates = {}
    
    # Pattern to match URL and its published date
    pattern = r'🔗 COMPLETE URL: ([^\n]+)\s*\n📅 Published: ([^\n]+)'
    matches = re.findall(pattern, output)
    
    for url, date_str in matches:
        url = url.strip()
        date_str = date_str.strip()
        
        if date_str and date_str != 'Date not available':
            try:
                parsed_date = date_parser.parse(date_str)
                url_dates[url] = parsed_date
            except:
                pass
    
    return url_dates

def is_url_from_2024_onwards(url: str, verbose: bool = False, metadata_date: Optional[datetime] = None) -> bool:
    """
    Check if URL is from 2024 or later.
    
    Priority:
    1. Use metadata_date if provided (from Tavily)
    2. Extract year from URL pattern
    3. Default to True (benefit of doubt)
    
    Returns True if year >= 2024 or year cannot be determined.
    Returns False if year < 2024.
    """
    # Priority 1: Use metadata date if available
    if metadata_date:
        year = metadata_date.year
        if verbose:
            date_source = "metadata"
            if year >= 2024:
                print(f"   ✅ Year {year} from {date_source} (valid): {url[:60]}...")
            else:
                print(f"   ❌ Year {year} from {date_source} (DISCARDED): {url[:60]}...")
        return year >= 2024
    
    # Priority 2: Extract from URL
    year = extract_year_from_url(url)
    
    if year is None:
        if verbose:
            print(f"   ⚠️ No date found (allowing): {url[:60]}...")
        return True  # Cannot determine, allow it
    
    if year >= 2024:
        if verbose:
            print(f"   ✅ Year {year} from URL (valid): {url[:60]}...")
        return True
    else:
        if verbose:
            print(f"   ❌ Year {year} from URL (DISCARDED): {url[:60]}...")
        return False

def filter_urls_by_date(urls: List[str], min_year: int = 2024, verbose: bool = False, 
                        url_metadata: Optional[Dict[str, datetime]] = None) -> Tuple[List[str], Dict[str, str]]:
    """
    Filter list of URLs to only include those from min_year onwards.
    URLs without detectable dates are included (benefit of doubt).
    
    Args:
        urls: List of URLs to filter
        min_year: Minimum year to keep (default: 2024)
        verbose: Print filtering details
        url_metadata: Optional dict mapping URL -> datetime from search metadata
    
    Returns:
        Tuple of (filtered_urls, date_info) where date_info maps URL -> date source
    """
    filtered = []
    discarded = []
    date_info = {}  # Track where date came from
    
    for url in urls:
        # Check metadata first
        metadata_date = url_metadata.get(url) if url_metadata else None
        
        if is_url_from_2024_onwards(url, verbose=verbose, metadata_date=metadata_date):
            filtered.append(url)
            
            # Track date source
            if metadata_date:
                date_info[url] = f"metadata:{metadata_date.strftime('%Y-%m-%d')}"
            else:
                year = extract_year_from_url(url)
                if year:
                    date_info[url] = f"url:{year}"
                else:
                    date_info[url] = "unknown"
        else:
            discarded.append(url)
    
    if verbose and discarded:
        print(f"\n🗑️ DISCARDED {len(discarded)} URLs from before {min_year}:")
        for url in discarded[:5]:  # Show first 5
            print(f"   - {url}")
        if len(discarded) > 5:
            print(f"   ... and {len(discarded) - 5} more")
    
    return filtered, date_info


# -----------------------------
# PDF Export Function
# -----------------------------
def save_research_output_as_pdf(research_response, filename: str = None):
    """
    Save the research output as a formatted PDF.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_output_{timestamp}.pdf"
    
    # Ensure filename has .pdf extension
    if not filename.endswith('.pdf'):
        filename += '.pdf'
    
    # Create PDF
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )
    
    # Container for PDF elements
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a5490'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1a5490'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c5f8d'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#2c5f8d'),
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        leading=14
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['BodyText'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=6,
        leading=14
    )
    
    # Helper function to clean text for PDF
    def clean_text(text):
        """Remove problematic characters and escape XML special chars"""
        if not text:
            return ""
        # Replace common problematic characters
        text = str(text)
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        # Remove emoji and special unicode
        text = re.sub(r'[^\x00-\x7F\u00A0-\uFFFF]+', '', text)
        return text
    
    # Title Page
    story.append(Paragraph(clean_text(research_response.topic), title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Metadata
    metadata_data = [
        ['Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ['Total Events:', str(len(research_response.events))],
        ['Action Items:', str(len(research_response.action_items) if research_response.action_items else 0)],
    ]
    
    metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
    metadata_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    story.append(metadata_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading1_style))
    story.append(Paragraph(clean_text(research_response.summary), body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Sources
    if research_response.source:
        story.append(Paragraph("Primary Sources", heading2_style))
        for idx, source in enumerate(research_response.source, 1):
            story.append(Paragraph(f"{idx}. {clean_text(source)}", bullet_style))
    
    story.append(PageBreak())
    
    # Events Section
    story.append(Paragraph("Detailed Analysis of Events", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    for idx, event in enumerate(research_response.events, 1):
        # Event Header
        story.append(Paragraph(f"Event {idx}: {clean_text(event.event)}", heading2_style))
        
        # Event Metadata Table
        event_metadata = [
            ['Category:', clean_text(event.category)],
            ['Location:', clean_text(event.location) if event.location else 'N/A'],
            ['Date Range:', clean_text(', '.join(event.date)) if event.date else 'N/A'],
            ['Relevance:', clean_text(event.relevance)],
            ['Confidence:', clean_text(event.confidence)],
            ['Signal Strength:', clean_text(event.signal_strength)],
        ]
        
        event_table = Table(event_metadata, colWidths=[1.5*inch, 4.5*inch])
        event_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1a5490')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        story.append(event_table)
        story.append(Spacer(1, 0.15*inch))
        
        # Description
        story.append(Paragraph("<b>Description:</b>", heading3_style))
        story.append(Paragraph(clean_text(event.description), body_style))
        
        # Impact
        story.append(Paragraph("<b>Impact Analysis:</b>", heading3_style))
        story.append(Paragraph(clean_text(event.impact), body_style))
        
        # Scenario
        story.append(Paragraph("<b>Scenarios:</b>", heading3_style))
        story.append(Paragraph(clean_text(event.scenario), body_style))
        
        # Actors
        story.append(Paragraph("<b>Key Actors:</b>", heading3_style))
        actors_text = ', '.join(event.actors) if event.actors else 'N/A'
        story.append(Paragraph(clean_text(actors_text), body_style))
        
        # Policy Intervention
        story.append(Paragraph("<b>Policy Intervention Recommendations:</b>", heading3_style))
        story.append(Paragraph(clean_text(event.policy_intervention), body_style))
        
        # Sources
        story.append(Paragraph("<b>Sources:</b>", heading3_style))
        if event.source:
            # Handle both list and string formats
            sources = event.source if isinstance(event.source, list) else event.source.split(', ')
            for source in sources:
                story.append(Paragraph(f"• {clean_text(source)}", bullet_style))
        
        # Informal Insights (if available)
        if hasattr(event, 'informal_insights') and event.informal_insights:
            story.append(Paragraph("<b>Informal Insights:</b>", heading3_style))
            story.append(Paragraph(clean_text(event.informal_insights), body_style))
        
        if idx < len(research_response.events):
            story.append(PageBreak())
    
    # Action Items
    if research_response.action_items:
        story.append(PageBreak())
        story.append(Paragraph("Recommended Action Items", heading1_style))
        story.append(Spacer(1, 0.1*inch))
        
        for idx, action in enumerate(research_response.action_items, 1):
            story.append(Paragraph(f"{idx}. {clean_text(action)}", bullet_style))
    
    # Repeated Events (if any)
    if hasattr(research_response, 'repeated_events') and research_response.repeated_events:
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Repeated Events", heading2_style))
        for event in research_response.repeated_events:
            story.append(Paragraph(f"• {clean_text(event)}", bullet_style))
    
    # Build PDF
    try:
        doc.build(story)
        file_size = Path(filename).stat().st_size / 1024
        print(f"\n✅ PDF saved: {filename}")
        print(f"📄 File size: {file_size:.2f} KB")
        return filename
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        return None


# -----------------------------
# Pydantic Models for Structured Output
# -----------------------------
class Event(BaseModel):
    event: str
    description: str = Field(
        description="Detailed 150+ word context with international comparisons/case studies."
    )
    date: List[str] = Field(
        description="⚠️ Each URL in 'source' must have a matching date in DD/MM/YYYY format. Example: if 3 URLs → 3 dates. Extract from article content or metadata; no generic years."
    )
    actors: List[str]
    location: Optional[str]
    category: str  # Policy / Systemic risk / Public sentiment

    impact: str = Field(
        description="150+ word CPF impact analysis by stakeholder group:"
                    "- Young (20–35): OA & housing"
                    "- Mid-career (35–50): SA/MA balance"
                    "- Pre-retirees (50–65): adequacy & withdrawals"
                    "- Retirees (65+): CPF LIFE & Medisave"
                    "- Gig workers / Low-wage / PMET: irregularity & risk gaps"
                    "Include $ figures, affected population, and precedents."
    )

    scenario: str = Field(
        min_length=200,
        description="Predictive scenario (200+ words) with 4 cases. Use inline [URL#] citations: "
                    "1️⃣ **Base** – Most likely outcome (estimate probability as a % range) "
                    "2️⃣ **Optimistic** – Best realistic outcome (estimate probability as a % range) "
                    "3️⃣ **Pessimistic** – Worst realistic outcome (estimate probability as a % range) "
                    "4️⃣ **Black Swan** – Tail risk (estimate probability as a % range) "
                    "Each case: year, trigger, quantified CPF impact, and affected groups. "
                    "You MUST estimate scenario probabilities based on the unique evidence, uncertainty, and context for each event. Do NOT use default or template probabilities—tailor the numbers to the specifics of the event. Briefly justify each probability in 1–2 sentences. Use 2026–2035 timeframe."
    )

    source: str = Field(
        description="""
        For Established issues
        List all full article URLs (comma-separated) from *different* sources. Example: 'https://a.com/x, https://b.com/y, https://c.com/z'.
        """

    )

    relevance: str = Field(description="High / Medium / Low — with one-line justification.")

    confidence: str = Field(
        description="Confidence (High/Med/Low) + evidence-based breakdown:"
                    "- Source quality (gov/academic/news)"
                    "- Trend consistency"
                    "- Geographic precedent"
                    "- Expert consensus"
                    "- Quantitative support"
                    "Rate each 1–10 (e.g. 'Source: 8/10 - 2 govt + 1 academic')."
    )

    policy_intervention: str = Field(
        description="200+ word decision-support note:"
                    "- Option A & B: describe intervention, pros/cons, precedent"
                    "- 3–5 monitoring indicators"
                    "- 3 reflective questions for policymakers (data gaps, stakeholders, feasibility)."
    )
    signal_strength: str = Field(description="Tag as 'Established', 'Emerging', or 'Weak Signal' with a brief rationale.")
    informal_insights: Optional[str] = Field(default=None, description="For established/mainstream events, summarize the latest new developments, sentiment, or weak signals from informal channels (e.g., forums, social media, community blogs). Only populate if signal_strength is 'Established'.")

# One research response---(contains)---> multiple events--> one event covers all the fields listed above
class ResearchResponse(BaseModel):
    topic: str
    summary: str = Field(description="EXECUTIVE SUMMARY for policymakers (150-200 words):"
                        "Brief overview of findings designed for senior decision-makers."
                        "Format: [X] emerging issues identified, prioritized by [criteria]. "
                        "Most urgent: [issue], requiring attention by [timeframe]. "
                        "Key uncertainties: [what we don't know]. "
                        "Recommended next actions: [immediate steps for validation/planning].")
    source: List[str] = Field(description="🚨 CRITICAL: Cite ALL FULL URLs used as evidence. Source QUALITY > quantity. A single government report is better than 3 opinion blogs. Justify the quality of sources in the 'confidence' field.")
    tools_used: List[str]
    events: List[Event]
    action_items: List[str] = Field(default_factory=list, description="Optional: 3-5 immediate next steps for policymakers (e.g., 'Request MOM data on caregiving workforce exits', 'Consult with eldercare sector on cost projections')")
    repeated_events: Optional[List[str]] = Field(default=None, description="List of event names that are repeated from previous reports for explicit highlighting.")

# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",           # Or "gpt-4o-mini" for faster runs
    temperature=0,             # Deterministic output for structured data
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Print to see what instructions are generated
print("="*80)
print("FORMAT INSTRUCTIONS BEING SENT TO LLM:")
print("="*80)
print(parser.get_format_instructions())
print("="*80)

# -----------------------------
# System Prompt
# -----------------------------
stage1_system_prompt_template = """
You are an elite research assistant specializing in CPF policy analysis and ground sensing of emerging issues in Singapore.

Current date: {current_date_time}

Instructions:
- **Do not summarize, analyze, or generate structured outputs. Only return raw search results and metadata.**
- Focus strictly on information from 2024-2025. Reject and ignore any articles or data from before 2024.
- Perform search using *tavily tool* to gain better ground sensing and collate a knowledge list


"""

stage1_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", stage1_system_prompt_template),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(
    current_date_time=current_datetime_str  # Add this line
)


# -----------------------------
# Tools Setup
# -----------------------------
tools = [tavily_tool, save_tool]

# Add fallback search if available
if search_tool is not None:
    tools.append(search_tool)

# -----------------------------
# Agent Setup
# -----------------------------
agent = create_tool_calling_agent(
    llm=llm,
    prompt=stage1_prompt,  # Use the new, simple prompt
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -----------------------------
# STAGE 1: Knowledge Building
# -----------------------------
print("" + "="*80)
print("🔍 STAGE 1: BUILDING KNOWLEDGE BASE")
print("="*80)

# "Singapore social media CPF discussions 2024 2025"
# "Singapore social security changes 2024 2025"


knowledge_queries = [
    # ============================================
    # TIER 1: GLOBAL MACRO TRENDS & SYSTEMIC RISKS
    # ============================================
    {"query": "What are the biggest global economic, demographic, or geopolitical risks that could impact retirement systems and pension funds worldwide 2024-2025?", "label": "Global Macro: Systemic Risks to Retirement", "sentiment": "horizon"},
    {"query": "What are international organizations (IMF, World Bank, OECD, BIS) warning about regarding pension sustainability and retirement adequacy 2024-2025?", "label": "Global Macro: International Warnings", "sentiment": "horizon"},
    {"query": "What are the most significant pension crises, reforms, or failures happening globally 2024-2025? Include Europe, Asia, Americas, and emerging markets.", "label": "Global Macro: Pension Crises Worldwide", "sentiment": "horizon"},
    
    # ============================================
    # TIER 2: CROSS-BORDER TRENDS & PRECEDENTS
    # ============================================
    {"query": "What innovative or experimental pension reforms are being tested in Nordic countries, UK, Australia, Canada, Japan 2024-2025?", "label": "International: Advanced Economy Experiments", "sentiment": "horizon"},
    {"query": "What retirement and social security challenges are Asian countries (Japan, South Korea, Taiwan, Hong Kong, Malaysia) facing 2024-2025? Regional comparisons.", "label": "International: Asian Retirement Challenges", "sentiment": "horizon"},
    {"query": "What lessons from international pension failures or controversies could apply to Singapore 2024-2025? Include UK, US, European cases.", "label": "International: Cautionary Tales & Failures", "sentiment": "horizon"},
    {"query": "What are global think tanks and research institutions publishing about future-of-retirement and pension sustainability 2024-2025? Include Brookings, CSIS, Peterson Institute.", "label": "International: Think Tank Research", "sentiment": "horizon"},
    
    # ============================================
    # TIER 3: TECHNOLOGY & DISRUPTION
    # ============================================
    {"query": "How are AI, automation, and gig economy disrupting traditional employment and retirement savings globally 2024-2025? Future of work implications.", "label": "Tech Disruption: AI & Future of Work", "sentiment": "horizon"},
    {"query": "What are fintech, crypto, and web3 innovations in retirement planning and pension management 2024-2025? Include DeFi, tokenization, digital assets.", "label": "Tech Disruption: Fintech & Web3 Pensions", "sentiment": "horizon"},
    {"query": "What are the cybersecurity risks, data breaches, or tech failures affecting pension funds and retirement systems 2024-2025?", "label": "Tech Disruption: Cyber Risks to Pensions", "sentiment": "horizon"},
    {"query": "How are longevity breakthroughs, healthtech, and aging science changing retirement planning assumptions 2024-2025? Impact of living to 100+.", "label": "Tech Disruption: Longevity & Healthtech", "sentiment": "horizon"},
    
    # ============================================
    # TIER 4: WEAK SIGNALS & FRINGE SOURCES
    # ============================================
    {"query": "What are the most surprising, unconventional, or contrarian views on retirement and pensions from blogs, podcasts, and alternative media 2024-2025?", "label": "Weak Signals: Alternative Media & Contrarians", "sentiment": "horizon"},
    {"query": "What are early warning signals, emerging risks, or 'canary in the coal mine' indicators for retirement systems from forums, Reddit, Twitter/X 2024-2025?", "label": "Weak Signals: Social Media Early Warnings", "sentiment": "horizon"},
    {"query": "What speculative scenarios, black swan events, or 'what if' analyses exist for pension and retirement systems 2024-2025? Include scenario planning.", "label": "Weak Signals: Black Swan Scenarios", "sentiment": "horizon"},
    {"query": "What are fringe communities, subcultures, or movements saying about retirement (FIRE movement, anti-work, digital nomads) 2024-2025?", "label": "Weak Signals: Fringe Movements & Subcultures", "sentiment": "horizon"},
    
    # ============================================
    # TIER 5: INTERDISCIPLINARY & ADJACENT DOMAINS
    # ============================================
    {"query": "How are climate change, environmental risks, and ESG factors affecting pension fund strategies and retirement security 2024-2025?", "label": "Adjacent: Climate & ESG Impact", "sentiment": "horizon"},
    {"query": "What are behavioral economics and psychology insights on retirement savings behavior and pension engagement 2024-2025? Nudge theory applications.", "label": "Adjacent: Behavioral Economics", "sentiment": "horizon"},
    {"query": "How are housing affordability crisis, real estate bubbles, and homeownership affecting retirement adequacy globally 2024-2025?", "label": "Adjacent: Housing & Retirement", "sentiment": "horizon"},
    {"query": "What are healthcare cost inflation, long-term care crises, and medical bankruptcy implications for retirement planning 2024-2025?", "label": "Adjacent: Healthcare Costs & Retirement", "sentiment": "horizon"},
    
    # ============================================
    # TIER 6: SINGAPORE-SPECIFIC (Enhanced Scope)
    # ============================================
    {"query": "What are the most surprising or under-discussed CPF and retirement issues in Singapore 2024-2025? Include forums, social media, Reddit r/singapore.", "label": "Singapore: Non-Obvious Issues & Ground Sensing", "sentiment": "horizon"},
    {"query": "What are Singapore policymakers, ministers, and MPs saying about CPF reforms and retirement challenges 2024-2025? Parliamentary debates.", "label": "Singapore: Policy Signals & Debates", "sentiment": "horizon"},
    {"query": "What are Singaporean researchers, universities, and think tanks (LKYSPP, IPS, RSIS) publishing on CPF and retirement 2024-2025?", "label": "Singapore: Academic & Research", "sentiment": "horizon"},
    {"query": "How do Singapore's retirement challenges compare to regional neighbors and advanced economies 2024-2025? Benchmarking and gap analysis.", "label": "Singapore: Comparative Analysis", "sentiment": "horizon"},
    
    # ============================================
    # TIER 7: EXPERT OPINIONS & THOUGHT LEADERSHIP
    # ============================================
    {"query": "What are leading economists, pension experts, and thought leaders predicting about retirement systems 2024-2025? Include Nobel laureates, IMF economists.", "label": "Expert Opinions: Leading Economists", "sentiment": "horizon"},
    {"query": "What are investment managers, asset allocators, and sovereign wealth funds saying about pension fund strategies 2024-2025? BlackRock, Vanguard, GIC insights.", "label": "Expert Opinions: Investment Perspectives", "sentiment": "horizon"},
    {"query": "What are demographic experts and population researchers warning about aging societies and pension sustainability 2024-2025?", "label": "Expert Opinions: Demographics & Aging", "sentiment": "horizon"},
]



# Gather knowledge from different time periods
all_knowledge = []
all_sources = []
found_urls = []  # ADD THIS: Track URLs

# -----------------------------
# Analyze past two research reports for repeated topics
# -----------------------------
import glob
from collections import Counter
def get_all_research_files(max_n=20):
    files = glob.glob("research_output_2025*.json")
    files += glob.glob("research_output_2024*.json")
    files = sorted(files, reverse=True)
    return files[:max_n]

def extract_event_names_list(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Support both 'events' at top-level or inside 'ResearchResponse'
        events = data.get('events')
        if events is None and 'ResearchResponse' in data:
            events = data['ResearchResponse'].get('events', [])
        if not events:
            return []
        return [e.get('event') for e in events if 'event' in e]
    except Exception as e:
        print(f"[PAST REPORTS] Error reading {filepath}: {e}")
        return []

def extract_event_details(filepath):
    """Return a dict of event name to description for a given report file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        events = data.get('events')
        if events is None and 'ResearchResponse' in data:
            events = data['ResearchResponse'].get('events', [])
        if not events:
            return {}
        return {e.get('event'): e.get('description', '') for e in events if 'event' in e}
    except Exception as e:
        print(f"[PAST REPORTS] Error reading {filepath}: {e}")
        return {}


# Get previous 2 weeks' event names (for filtering new vs repeated)
recent_files = get_all_research_files(max_n=2)
previous_events = set()
for f in recent_files:
    previous_events |= set(extract_event_names_list(f))

# Define all_files and event_counter for repeated event evolution tracking
all_files = get_all_research_files(max_n=2)
all_event_names = []
for f in all_files:
    all_event_names.extend(extract_event_names_list(f))
event_counter = Counter(all_event_names)

# Gather knowledge from different time periods
all_knowledge = []
all_sources = []
found_urls = []  # Track URLs
url_metadata_dates = {}  # NEW: Track published dates from Tavily metadata

for idx, kq in enumerate(knowledge_queries, 1):
    sentiment_icon = {"positive": "✅", "negative": "⚠️", "neutral": "ℹ️", "sentiment": "💭"}.get(kq.get('sentiment', 'neutral'), "ℹ️")
    print(f"📚 [{idx}/{len(knowledge_queries)}] {sentiment_icon} {kq['label']}")
    print(f"    Query: {kq['query']}")
    
    try:
        knowledge_response = agent_executor.invoke({"query": kq['query']})
        output = knowledge_response.get("output", "")
        
        if output:
            # Extract URLs from output
            urls_in_output = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', output)
            found_urls.extend(urls_in_output)
            
            # NEW: Extract published dates from Tavily metadata
            output_metadata = extract_dates_from_search_output(output)
            url_metadata_dates.update(output_metadata)
            
            # FILTER URLs: Only keep those from 2024 onwards (using metadata when available)
            urls_before_filter = len(urls_in_output)
            filtered_urls, date_info = filter_urls_by_date(
                urls_in_output, 
                min_year=2024, 
                verbose=False,
                url_metadata=output_metadata
            )
            urls_after_filter = len(filtered_urls)
            
            found_urls.extend(filtered_urls)
            
            # Add sentiment label to knowledge
            sentiment_label = kq.get('sentiment', 'neutral').upper()
            all_knowledge.append(f"## {kq['label']} [SENTIMENT: {sentiment_label}]{output}")
            all_sources.append(output)
            print(f"✅ Collected {len(output)} characters, {len(urls_in_output)} URLs")
            
            # ADD THIS: Show found URLs
            if urls_in_output:
                for url in urls_in_output[:2]:  # Show first 2
                    print(f"       📎 {url}")
            # Show filtering results
            filtered_count = urls_before_filter - urls_after_filter
            metadata_count = sum(1 for url in filtered_urls if url in output_metadata)
            
            if filtered_count > 0:
                print(f"✅ Collected {len(output)} characters, {urls_after_filter} URLs (🗑️ filtered {filtered_count} pre-2024, 📅 {metadata_count} with metadata)")
            else:
                print(f"✅ Collected {len(output)} characters, {urls_after_filter} URLs (📅 {metadata_count} with metadata)")
            
            # Show found URLs
            if filtered_urls:
                for url in filtered_urls[:2]:  # Show first 2
                    date_source = date_info.get(url, 'unknown')
                    print(f"       📎 {url} [{date_source}]")
        else:
            print(f"⚠️ No output received")
    except Exception as e:
        print(f"❌ Error: {e}")
        continue

# ADD THIS: Track URL frequency (hot topics = URLs appearing in multiple searches)
from collections import Counter
url_frequency = Counter(found_urls)
unique_urls = list(dict.fromkeys(found_urls))

# Identify "hot topics" - URLs cited by multiple knowledge queries
hot_topic_urls = {url: count for url, count in url_frequency.items() if count >= 2}

print(f"📊 Total unique URLs collected: {len(unique_urls)}")
print(f"🔥 Hot topic URLs (cited ≥2 times): {len(hot_topic_urls)}")

# ADD THIS: Detailed date filtering report
print("" + "="*80)
print("📅 URL DATE FILTERING REPORT")
print("="*80)
urls_with_metadata = 0
urls_with_url_dates = 0
urls_without_dates = 0
year_distribution = Counter()

for url in unique_urls:
    # Check if we have metadata date
    if url in url_metadata_dates:
        urls_with_metadata += 1
        year = url_metadata_dates[url].year
        year_distribution[year] += 1
    else:
        # Check URL pattern
        year = extract_year_from_url(url)
        if year:
            urls_with_url_dates += 1
            year_distribution[year] += 1
        else:
            urls_without_dates += 1

total_with_dates = urls_with_metadata + urls_with_url_dates

print(f"📊 Date Detection Summary:")
print(f"   ✅ URLs with metadata dates (from Tavily): {urls_with_metadata}")
print(f"   ✅ URLs with dates in URL pattern: {urls_with_url_dates}")
print(f"   ⚠️ URLs without detectable dates: {urls_without_dates}")
print(f"   📈 Total with dates: {total_with_dates}/{len(unique_urls)} ({100*total_with_dates/len(unique_urls):.1f}%)")

print(f"\n📊 Year Distribution:")
for year in sorted(year_distribution.keys(), reverse=True):
    bar = "█" * min(50, year_distribution[year])
    print(f"   {year}: {bar} ({year_distribution[year]} URLs)")

if urls_without_dates > 0:
    print(f"\n⚠️ WARNING: {urls_without_dates} URLs have no detectable date")
    print(f"   These are INCLUDED (benefit of doubt) but should be manually verified:")
    
    # Show sample URLs without dates
    no_date_urls = [url for url in unique_urls[:50] 
                    if url not in url_metadata_dates and extract_year_from_url(url) is None]
    for url in no_date_urls[:5]:
        print(f"   - {url}")
    if len(no_date_urls) > 5:
        print(f"   ... and {len(no_date_urls) - 5} more")

# Combine all knowledge
combined_knowledge = "" + "="*80 + "".join(all_knowledge)

print(f"✅ Knowledge building complete. Total knowledge: {len(combined_knowledge)} characters")

# Show collected URLs
if unique_urls:
    print("" + "="*80)
    print("📎 COLLECTED URLS FOR STAGE 2")
    print("="*80)
    for i, url in enumerate(unique_urls[:20], 1):
        print(f"{i}. {url}")
    if len(unique_urls) > 20:
        print(f"... and {len(unique_urls) - 20} more")

# -----------------------------
# ENHANCED URL TRACKING & VALIDATION
# -----------------------------
print("" + "="*80)
print("🔍 DEBUGGING: URL-TO-CONTENT MAPPING")
print("="*80)

# Create a mapping of URLs to their content
url_to_content = {}
for idx, kq in enumerate(knowledge_queries[:9], 1):
    output = all_sources[idx-1] if idx-1 < len(all_sources) else ""
    urls_in_section = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', output)
    
    for url in urls_in_section:
        if url not in url_to_content:
            # Extract ~200 chars of context around the URL
            url_pos = output.find(url)
            context_start = max(0, url_pos - 100)
            context_end = min(len(output), url_pos + len(url) + 100)
            context = output[context_start:context_end]
            
            url_to_content[url] = {
                'topic': kq['label'],
                'context': context,
                'full_section': output
            }

print(f"📊 Mapped {len(url_to_content)} unique URLs to content")

# Show sample mappings
for i, (url, data) in enumerate(list(url_to_content.items())[:3], 1):
    print(f"{i}. {url[:60]}...")
    print(f"   Topic: {data['topic']}")
    print(f"   Context: {data['context'][:100]}...")

# Display hot topics (URLs appearing in multiple searches)
if hot_topic_urls:
    print("" + "="*80)
    print("🔥 HOT TOPICS (URLs appearing in multiple searches - PRIORITY SIGNALS)")
    print("="*80)
    for url, count in sorted(hot_topic_urls.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   [{count}x] {url[:70]}...")
        if url in url_to_content:
            print(f"        Topic: {url_to_content[url]['topic']}")

# Define the prediction query with URLs (Improved for direct action)
prediction_query = f"""
**ANALYZE & FORECAST**: Identify 3-5 high-priority, non-obvious emerging issues impacting CPF AND/OR its CPF members (2026-2035).
**ANALYZE & FORECAST**: Identify 10-15 high-priority, non-obvious emerging issues impacting CPF AND/OR its CPF members (2026-2035).
When searching for information on mainstream CPF issues, prioritize and include results from informal channels (e.g., forums, social media, community blogs, public comments) in addition to mainstream news sources. Highlight early warning signals, sentiment, and public concerns from these informal sources.


Order the final output by Urgency × Impact × Novelty score.

### GATHERED KNOWLEDGE:
{combined_knowledge}

### AVAILABLE SOURCE URLS WITH CONTEXT (USE THESE EXACT FULL URLs):
{chr(10).join([f"- {url} (Topic: {url_to_content[url]['topic']})" for url in unique_urls[:50] if url in url_to_content])}

"""


# -----------------------------
# ------------------------------------------
# STAGE 2: Trend Analysis & Prediction
# -----------------------------
print("" + "="*80)
print("🔮 STAGE 2: TREND ANALYSIS & PREDICTIVE SYNTHESIS")
print("="*80)

prediction_system_prompt = f"""
You are a strategic foresight analyst for Singapore’s CPF system, advising senior policymakers and monitoring issues that could affect up to 4.4 million CPF members.

TASK: Use the provided data to anticipate **CPF-related risks, opportunities, and structural shifts** for 2024–2025, with special attention to both mainstream (established) and emerging issues.
TASK: Use the provided data to anticipate **CPF-related risks, opportunities, and structural shifts** for {one_year_ago_int} to {current_datetime_str}, with special attention to both mainstream (established) and emerging issues.

**MANDATORY CHECK FOR ESTABLISHED ISSUES:**
- Always check for and include established (mainstream) issues (e.g., cost of living, healthcare, CPF policy changes) in your analysis, even if they are recurring or well-known.
- For each established issue, surface and highlight the latest new developments, sentiment shifts, or weak signals from informal channels (e.g., Reddit, forums, social media, community blogs, public comments).
- When citing evidence for established issues, prioritize and include informal sources (such as Reddit articles or forum posts) if available, and cite them directly in the event's source field.
- Clearly distinguish between established issues and new/emerging issues in your output.

**PRIORITIZE:**
- For mainstream problems, focus your analysis on evidence, sentiment, and early warning signals from informal channels, not just official or mainstream news.
- Highlight how informal perspectives may reveal emerging risks, gaps, or public concerns that are not yet fully addressed by policy.
- Also include non-mainstream, weak-signal friction, and established issues only if they show novel urgency or character.

**Definition:** An emerging issue is a new, weak-signal, or rapidly developing trend with limited but credible evidence, not yet widely reported or discussed.

Current date: {current_datetime_str}
Intelligence horizon: STRICTLY 2024-2025 ONLY (ignore pre-2024 articles)

----------

### CRITICAL OUTPUT INSTRUCTIONS
1. DO NOT generate any  text, conversation, apologies, or markdown code blocks.
2. RETURN ONLY the raw JSON object that precisely conforms to the ResearchResponse schema below.
3. For each event, create a detailed, well-supported Event object.
4. Your analysis must be evidence-based and fully leverage the Pydantic Field Descriptions (minimum length, formatting, required content).
5. For the 'scenario' field, introductory estimate scenario probabilities based on unique evidence for each event. Do NOT use default/template probabilities—tailor numbers to the event and briefly justify each probability.
6. If an event has only one credible source URL:
   - Flag as "Emerging" or "Weak Signal".
   - Set confidence to "Medium" or "Low" (never "High").
   - Add a short justification for why only one source was found.
   - Recommend specific further monitoring actions (e.g., "Monitor for additional reports", "Seek field feedback", "Track social media/forums").


## ResearchResponse SCHEMA (SUMMARY)
- topic: str
- summary: str (150-200 words, executive summary)
- source: List[str] (ALL full URLs used)
- tools_used: List[str]
- events: List[Event]
- action_items: List[str] (optional)

## Event FIELDS (for each event)
- event: str
- description: str (≥150 words, with data)
- date: List[str] (DD/MM/YYYY, one per URL)
- actors: List[str]
- location: Optional[str]
- category: str
- impact: str (≥150 words, stakeholder segmentation, quantified)
- scenario: str (≥500 chars, 4 scenarios: Base, Optimistic, Pessimistic, Black Swan, with [**{"INSERT RELEVANT URL"}] inline citations)
- source: str (≥3 full URLs, comma-separated, from different topics)
- relevance: str (High/Medium/Low + justification)
- confidence: str (High/Medium/Low + evidence-based justification)
- policy_intervention: str (≥200 words, multiple options, pros/cons, precedents, questions, monitoring indicators)
- signal_strength: str (Tag as 'Established', 'Emerging', or 'Weak Signal' with rationale)

----------

## SOURCE & DATE RULES
- Each event should cite ≥2 full URLs from different topic areas where possible.
- For each URL, extract publication date (DD/MM/YYYY) from article context.
- The number of dates MUST match the number of URLs.
- NO generic domains or years.

---

## EVENT CHECKLIST (for each event)
- [ ] Has NOT been reported in last 2 weeks
- [ ] 2+ full URLs from different topics OR flag event as emerging, low confidence.
- [ ] Dates match URLs, all in DD/MM/YYYY
- [ ] description ≥150 words
- [ ] impact ≥150 words, stakeholder segmentation
- [ ] scenario ≥500 chars, 4 scenarios, [URL#] citations
- [ ] Quantified data (numbers, percentages, $)
- [ ] Policy options: multiple, with pros/cons, precedents
- [ ] Validation questions and monitoring metrics

---

## EXAMPLE (ABBREVIATED)
Event: "Quiet Quitting in Sandwich Generation"
Timeframe: 2026-2030 | Strength: 7-8
Description: [150+ words, with data and citations]
Impact: [150+ words, with segmentation and numbers]
Scenario: [500+ chars, 4 scenarios, [URL#] inline]
Policy Intervention: [200+ words, options, pros/cons, precedents, questions, metrics]
Source: "https://employment-url-1, https://cpf-policy-url-2, https://regional-pension-url-3"
Date: ["15/03/2024", "22/11/2024", "08/01/2025"]

---

## CRITICAL: Return ONLY the raw JSON object matching the schema. NO markdown, NO extra text.

{{format_instructions}}

"""

# Around line 200-260, your Stage 2 prompt should be:
# Use LLM directly for final synthesis (not agent, to avoid more searches)
prediction_prompt_template = ChatPromptTemplate.from_messages([
    ("system", prediction_system_prompt),
    ("human", "{query}")
]).partial(format_instructions=parser.get_format_instructions())

print(f"🧠 Analyzing trends and generating predictions...")
print(f"   Input size: {len(combined_knowledge)} characters")

try:


    # Format the prediction prompt
    formatted_messages = prediction_prompt_template.format_messages(query=prediction_query)

    # DEBUG: Print the full formatted prompt being sent to the LLM
    print("\n" + "="*80)
    print("📝 DEBUG: FULL FORMATTED PROMPT TO LLM (Stage 2)")
    print("="*80)
    for msg in formatted_messages:
        print(f"[{msg.type.upper()}] {msg.content}\n")
    print("="*80 + "\n")

    # Get prediction from LLM
    prediction_output = llm.invoke(formatted_messages)
    output_text = prediction_output.content

    # DEBUG: Print the raw LLM output before parsing
    print("\n" + "="*80)
    print("📝 DEBUG: RAW LLM OUTPUT (Stage 2)")
    print("="*80)
    print(output_text[:2000])  # Print up to 2000 chars for readability
    print("\n" + "="*80)

    print(f"✅ Prediction generated: {len(output_text)} characters")

    # Parse the response
    print("" + "="*80)
    print("📊 PARSING FINAL PREDICTION")
    print("="*80)

    # Extract JSON from markdown code block if present
    if "```json" in output_text:
        print("   Found markdown JSON block, extracting...")
        start = output_text.find("```json") + 7
        end = output_text.find("```", start)
        json_text = output_text[start:end].strip()
    else:
        print("   Using raw output as JSON")
        json_text = output_text

    # Handle case where LLM wraps response in {"ResearchResponse": {...}}
    import json
    parsed_json = json.loads(json_text)

    print(f"   ✅ JSON parsed successfully")
    print(f"   Top-level keys: {list(parsed_json.keys())}")

    # If wrapped, unwrap it
    if "ResearchResponse" in parsed_json and isinstance(parsed_json, dict):
        print("   ⚠️ Unwrapping nested ResearchResponse")
        json_text = json.dumps(parsed_json["ResearchResponse"])

    structured_response = parser.parse(json_text)


    # --- Filter and separate repeated vs new events using partial/fuzzy matching ---
    import difflib
    all_events = structured_response.events
    def is_repeated_event(event_name, previous_event_names, threshold=0.7):
        # Use difflib to find close matches
        for prev in previous_event_names:
            ratio = difflib.SequenceMatcher(None, event_name.lower(), prev.lower()).ratio()
            if ratio >= threshold:
                return True
        return False

    repeated_events = [e for e in all_events if is_repeated_event(e.event, previous_events)]
    new_events = [e for e in all_events if not is_repeated_event(e.event, previous_events)]

    print("\n==============================")
    if repeated_events:
        print(f"🔁 Repeated topics from previous reports (not included in main output):")
        for e in repeated_events:
            print(f"  - {e.event}")
    else:
        print("✅ All topics are new compared to the last two reports.")
    print("==============================\n")

    # For all events, if signal_strength is 'Established', extract and highlight new developments from informal channels
    for event in all_events:
        if hasattr(event, 'signal_strength') and event.signal_strength and 'established' in event.signal_strength.lower():
            # Try to extract new developments from informal sources in the description/impact fields
            informal_texts = []
            for field in [event.description, event.impact]:
                # Look for sentences mentioning Reddit, forum, social media, blog, or similar
                matches = re.findall(r'([^.]*?(Reddit|forum|social media|blog|community|Telegram|Facebook|WhatsApp|WeChat|Discord|X/Twitter)[^.]*\.)', field, re.IGNORECASE)
                informal_texts.extend([m[0].strip() for m in matches])
            if informal_texts:
                event.informal_insights = ' '.join(informal_texts)
            else:
                event.informal_insights = None

            # --- Ensure Reddit URLs are cited in the source field if referenced ---
            # Find all Reddit URLs in unique_urls
            reddit_urls = [url for url in unique_urls if 'reddit.com' in url]
            # If any Reddit URL is referenced in the event's informal_insights or description, add to source if not present
            if reddit_urls:
                # Get current sources as a set
                current_sources = set([s.strip() for s in event.source.split(',')]) if event.source else set()
                # Check if any Reddit URL is referenced in the event's text
                event_text = (event.informal_insights or '') + ' ' + (event.description or '')
                for rurl in reddit_urls:
                    if rurl in event_text and rurl not in current_sources:
                        current_sources.add(rurl)
                # Update event.source with all sources, comma-separated
                event.source = ', '.join(current_sources)

    # Only include new events in the main output
    structured_response.events = new_events

    # Add repeated events to a dedicated field for explicit highlighting in the report, with recurrence count and evolution summary
    if repeated_events:
        # Gather evolution history for each repeated event
        evolution_summaries = []
        for e in repeated_events:
            event_name = e.event
            # Collect descriptions from all previous reports (most recent first)
            desc_history = []
            for f in all_files:
                details = extract_event_details(f)
                if event_name in details:
                    desc_history.append(details[event_name])
            # Only keep up to 3 most recent descriptions for brevity
            desc_history = desc_history[:3]
            summary = f"{event_name} (seen {event_counter[event_name]} times)\n"
            for i, desc in enumerate(desc_history, 1):
                summary += f"  [Prev #{i}] {desc[:200].replace('\n',' ')}{'...' if len(desc)>200 else ''}\n"
            evolution_summaries.append(summary.strip())
        structured_response.repeated_events = evolution_summaries
    else:
        structured_response.repeated_events = []

    # Validate URLs in sources
    print("" + "="*80)
    print("🔗 VALIDATING SOURCE URLS - 3+ URLs REQUIRED PER EVENT")
    print("="*80)

    validation_failed = False
    for idx, event in enumerate(structured_response.events, 1):
        print(f"📌 Event {idx}: {event.event}")
        
        if event.source:
            sources = [s.strip() for s in event.source.split(',')]
            url_count = len(sources)
            date_count = len(event.date)
            
            # Count URLs that meet criteria
            valid_url_count = 0
            for source in sources:
                # Check if it's a full URL
                if 'http' in source:
                    path_count = source.count('/')
                    if path_count > 3:  # Has article path
                        valid_url_count += 1
                        print(f"   ✅ Full URL: {source[:80]}...")
                    else:
                        print(f"   ⚠️ Generic domain (no article path): {source}")
                else:
                    print(f"   ❌ Not a URL: {source}")
            
            # Check if event meets 3+ URL requirement
            if valid_url_count < 3:
                print(f"   🚨🚨🚨 VALIDATION FAILED: Only {valid_url_count} valid URLs (need 3+) 🚨🚨🚨")
                validation_failed = True
            else:
                print(f"   ✅ PASSED: {valid_url_count} valid URLs")
            
            # Validate dates match URLs
            print(f"   📅 Date Validation:")
            print(f"      URLs: {url_count}, Dates: {date_count}")
            
            if date_count != url_count:
                print(f"      🚨🚨🚨 VALIDATION FAILED: Date count ({date_count}) doesn't match URL count ({url_count}) 🚨🚨🚨")
                validation_failed = True
            else:
                print(f"      ✅ PASSED: Date count matches URL count")
                
            # Check date format
            for i, date_str in enumerate(event.date, 1):
                if '/' in date_str and len(date_str) >= 8:  # Proper date format like DD/MM/YYYY
                    print(f"      ✅ Date {i}: {date_str}")
                else:
                    print(f"      ⚠️ Date {i}: {date_str} (Generic - should be DD/MM/YYYY)")
                    
        else:
            print(f"   ❌ No source provided")
            print(f"   🚨🚨🚨 VALIDATION FAILED: No URLs provided 🚨🚨🚨")
            validation_failed = True
    
    if validation_failed:
        print("" + "="*80)
        print("❌ OVERALL VALIDATION: FAILED - Some events have insufficient URLs")
        print("="*80)
    else:
        print("" + "="*80)
        print("✅ OVERALL VALIDATION: PASSED - All events have 3+ URLs")
        print("="*80)

    # DETAILED URL USAGE ANALYSIS
    print("" + "="*80)
    print("🔬 DETAILED URL USAGE ANALYSIS")
    print("="*80)

    for idx, event in enumerate(structured_response.events, 1):
        print(f"📌 Event {idx}: {event.event}")
        print(f"   Category: {event.category}")
        
        if event.source:
            sources = [s.strip() for s in event.source.split(',')]
            print(f"   Total URLs cited: {len(sources)}")
            
            for i, source in enumerate(sources, 1):
                # Check if it's a full URL
                if 'http' in source:
                    path_count = source.count('/')
                    
                    # Check if URL was in our collected list
                    is_from_search = source in unique_urls
                    
                    if path_count > 3:  # Has article path
                        status = "✅ Valid" if is_from_search else "⚠️ Valid but not from search"
                        print(f"   {i}. {status}: {source[:70]}...")
                        
                        # Show which topic this URL came from
                        if source in url_to_content:
                            print(f"      📚 From: {url_to_content[source]['topic']}")
                    else:
                        print(f"   {i}. ⚠️ Generic domain: {source}")
                else:
                    print(f"   {i}. ❌ Not a URL: {source}")
            
            # Analyze if event needs more sources
            event_word_count = len(event.description.split()) + len(event.impact.split())
            recommended_sources = max(2, min(5, event_word_count // 150))
            
            if len(sources) < recommended_sources:
                print(f"   ⚠️ RECOMMENDATION: Add {recommended_sources - len(sources)} more sources")
                print(f"      (Event has {event_word_count} words, recommending {recommended_sources} sources)")
        else:
            print(f"   ❌ NO SOURCES PROVIDED")
    

    print("" + "="*80)
    print("✅ FINAL PREDICTIVE INTELLIGENCE REPORT")
    print("="*80)
    # Highlight repeated topics if present
    if structured_response.repeated_events and len(structured_response.repeated_events) > 0:
        print("\n==============================")
        print("🔁 HIGHLIGHTED REPEATED TOPICS (with evolution summary):")
        for summary in structured_response.repeated_events:
            print(summary)
        print("==============================\n")
    # Output JSON with repeated_events as a top-level field
    output_json = structured_response.model_dump()
    # Ensure repeated_events is always present in the output JSON
    if not output_json.get('repeated_events'):
        output_json['repeated_events'] = []
    import json as _json
    print(_json.dumps(output_json, indent=2, ensure_ascii=False))

    # Save to file (with repeated events highlighted at the top of the file)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"research_output_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(_json.dumps(output_json, indent=2, ensure_ascii=False))
    print(f"💾 Saved to: {filename}")
        print("" + "="*80)
        print("✅ FINAL PREDICTIVE INTELLIGENCE REPORT")
        print("="*80)
        # Highlight repeated topics if present
        if structured_response.repeated_events and len(structured_response.repeated_events) > 0:
            print("\n==============================")
            print("🔁 HIGHLIGHTED REPEATED TOPICS (with evolution summary):")
            for summary in structured_response.repeated_events:
                print(summary)
            print("==============================\n")
        # Output JSON with repeated_events as a top-level field
        output_json = structured_response.model_dump()
        # Ensure repeated_events is always present in the output JSON
        if not output_json.get('repeated_events'):
            output_json['repeated_events'] = []
        import json as _json
        print(_json.dumps(output_json, indent=2, ensure_ascii=False))

        # Save to file (with repeated events highlighted at the top of the file)
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        json_filename = f"research_output_{timestamp}.json"
        pdf_filename = f"research_output_{timestamp}.pdf"
    
        # Save JSON
        with open(json_filename, 'w', encoding='utf-8') as f:
            f.write(_json.dumps(output_json, indent=2, ensure_ascii=False))
        print(f"💾 JSON saved to: {json_filename}")
    
        # Save PDF
        print(f"\n📄 Generating PDF report...")
        pdf_path = save_research_output_as_pdf(structured_response, pdf_filename)
    
        if pdf_path:
            print(f"\n✅ Research complete! Files saved:")
            print(f"   📄 JSON: {json_filename}")
            print(f"   📄 PDF: {pdf_path}")
        else:
            print(f"\n⚠️ PDF generation failed, but JSON saved: {json_filename}")
    
        # Update status file for Streamlit dashboard
        from pathlib import Path
        status_file = Path('research_status.json')
        if status_file.exists():
            try:
                import json as _status_json
                status_data = _status_json.loads(status_file.read_text())
                status_data['status'] = 'completed'
                status_data['end_time'] = datetime.now().isoformat()
                status_data['output_file'] = json_filename
                status_file.write_text(_status_json.dumps(status_data))
            except:
                pass
    
except json.JSONDecodeError as e:
    print(f"❌ JSON Decode Error: {e}")
    print(f"📄 Problematic text (first 1000 chars):{json_text[:1000]}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    print(f"📄 Output text (first 1000 chars):{output_text[:1000]}")