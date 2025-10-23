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

# Import API keys from sample.env file
load_dotenv(dotenv_path="sample.env")
 
#Find Current Date for LLM context
now = datetime.datetime.now()
current_datetime_str = now.strftime("%Y-%m-%d, %A. Time: %H:%M:%S. Current timezone is UTC+8.")

# -----------------------------
# Pydantic Models for Structured Output
# -----------------------------
class Event(BaseModel):
    event: str
    description: str = Field(description="Detailed context (150+ words) including international comparisons and case studies")
    date: List[str] = Field(description="🚨 CRITICAL: Publication dates of EACH source URL in DD/MM/YYYY format. MUST match the number of URLs in 'source' field. If source has 3 URLs, date must have 3 dates. If source has 4 URLs, date must have 4 dates. Extract dates from article URLs or content. Example: ['03/09/2024', '15/01/2025', '22/11/2024']. NO generic years like '2024' or '2025' alone - use specific dates.")
    actors: List[str]
    location: Optional[str]
    category: str  # Policy / Systemic risk / Public sentiment
    impact: str = Field(description="CPF-specific impact analysis (150+ words) with STAKEHOLDER SEGMENTATION:\n"
                        "- Young Workers (20-35): Impact on OA accumulation, housing affordability\n"
                        "- Mid-Career (35-50): Impact on SA/MA balance, sandwich generation\n"
                        "- Pre-Retirees (50-65): Impact on retirement adequacy, withdrawal timing\n"
                        "- Retirees (65+): Impact on CPF LIFE payouts, Medisave sustainability\n"
                        "- Gig Workers: Impact on contribution irregularity\n"
                        "- Low-wage vs PMET: Different vulnerability levels\n"
                        "Quantify impacts with dollar amounts, affected population sizes, and comparable precedents.")
    scenario: str = Field(min_length=400, description="PREDICTIVE SCENARIO PLANNING (500+ words):\n\n"
                          "Structure your prediction clearly with 4 distinct future scenarios. **ADD INLINE CITATIONS [URL#] after each claim using URLs from the 'source' field.**\n\n"
                          "**BASE CASE (50-60% probability) - Most Likely Outcome:**\n"
                          "By [YEAR]: [Specific prediction with numbers] [URL1]\n"
                          "Assumptions: [What conditions lead to this] [URL2]\n"
                          "CPF Impact: [Quantified effect on contributions/withdrawals/accounts] [URL1]\n"
                          "Affected: [Number of members, demographic groups] [URL3]\n\n"
                          "**OPTIMISTIC CASE (20-30% probability) - Best Realistic Outcome:**\n"
                          "By [YEAR]: [What happens if intervention succeeds] [URL#]\n"
                          "Trigger: [What policy/action enables this] [URL#]\n"
                          "CPF Impact: [Quantified positive effect] [URL#]\n\n"
                          "**PESSIMISTIC CASE (15-20% probability) - Worst Realistic Outcome:**\n"
                          "By [YEAR]: [What happens if situation deteriorates] [URL#]\n"
                          "Trigger: [What failure/crisis causes this] [URL#]\n"
                          "CPF Impact: [Quantified negative effect] [URL#]\n\n"
                          "**BLACK SWAN (1-5% probability) - Tail Risk:**\n"
                          "By [YEAR]: [Extreme unexpected event] [URL#]\n"
                          "Trigger: [Systemic shock or unprecedented event] [URL#]\n"
                          "CPF Impact: [Potential system-wide consequences] [URL#]\n\n"
                          "**CITATION FORMAT:** Use [URL1], [URL2], [URL3] etc. to reference the URLs in your 'source' field. Match claims to their supporting sources.\n\n"
                          "Justify each probability with evidence. Use specific years (2026-2035) and quantified impacts.")
    source: str = Field(description="RECOMMENDED: Provide 3 or more FULL article URLs from DIFFERENT topic areas, comma-separated. Example: 'https://site1.com/article-2024, https://site2.com/another-article-2025, https://site3.com/third-article-2024'. Count the commas - you need AT LEAST 2 commas (= 3 URLs). NO generic domains like 'https://domain.com'. If only 1 strong, highly relevant URL is available, it is acceptable, but 3+ is preferred.")    
    relevance: str =Field(description="High/medium/low with a justification") # High / Medium / Low
    confidence: str = Field(
        description="Confidence level in prediction (High/Medium/Low) with EVIDENCE-BASED justification:\n\n"
                    "**ASSESSMENT CRITERIA:**\n"
                    "1. SOURCE QUALITY: Government data > Academic research > News analysis > Opinion pieces\n"
                    "2. TEMPORAL CONSISTENCY: Is trend accelerating, stable, or decelerating?\n"
                    "3. GEOGRAPHIC PRECEDENTS: Has this happened elsewhere? How applicable to Singapore?\n"
                    "4. EXPERT CONSENSUS: Do multiple independent sources agree?\n"
                    "5. QUANTITATIVE EVIDENCE: Are there hard numbers or just qualitative claims?\n\n"
                    "**CONFIDENCE LEVELS:**\n"
                    "High (7-10/10): ✓ 3+ government/academic sources ✓ Accelerating trend ✓ Regional precedent ✓ Expert consensus ✓ Quantified data\n"
                    "Medium (4-6/10): ✓ 3+ credible sources ✓ Stable/emerging trend ✓ Some precedents ✓ Mixed expert views ✓ Partial data\n"
                    "Low (1-3/10): Limited sources, weak signals, speculative, no precedents, qualitative only\n\n"
                    "**MUST EXPLICITLY RATE EACH CRITERION** (e.g., 'Source Quality: 8/10 - Two govt reports + one academic paper')"
    )
    policy_intervention: str = Field(description="DECISION SUPPORT for policymakers (200+ words) - NOT prescriptive recommendations:\n\n"
                                      "**POLICY OPTIONS TO CONSIDER:**\n"
                                      "Option A: [Describe intervention approach 1] - Link to specific CPF account (OA/SA/MA/RA)\n"
                                      "  Pros: [Benefits and strengths]\n"
                                      "  Cons: [Risks and implementation challenges]\n"
                                      "  Precedent: [Which country/region tried this? What happened?]\n\n"
                                      "Option B: [Alternative intervention approach] - Link to CPF scheme/mechanism\n"
                                      "  Pros: [...]\n"
                                      "  Cons: [...]\n"
                                      "  Precedent: [...]\n\n"
                                      "**QUESTIONS FOR POLICYMAKERS:**\n"
                                      "- What additional data would help validate this prediction?\n"
                                      "- Which stakeholder groups should be consulted first?\n"
                                      "- What are the budget implications and political feasibility?\n\n"
                                      "**MONITORING INDICATORS:**\n"
                                      "Suggest 3-5 early warning metrics policymakers should track to detect if this issue is emerging.")

# One research response---(contains)---> multiple events--> one event covers all the fields listed above
class ResearchResponse(BaseModel):
    topic: str
    summary: str = Field(description="EXECUTIVE SUMMARY for policymakers (150-200 words):\n"
                        "Brief overview of findings designed for senior decision-makers.\n"
                        "Format: [X] emerging issues identified, prioritized by [criteria]. "
                        "Most urgent: [issue], requiring attention by [timeframe]. "
                        "Key uncertainties: [what we don't know]. "
                        "Recommended next actions: [immediate steps for validation/planning].")
    source: List[str] = Field(description="🚨 CRITICAL: Cite ALL FULL URLs used as evidence. Source QUALITY > quantity. A single government report is better than 3 opinion blogs. Justify the quality of sources in the 'confidence' field.")
    tools_used: List[str]
    events: List[Event]
    action_items: List[str] = Field(default_factory=list, description="Optional: 3-5 immediate next steps for policymakers (e.g., 'Request MOM data on caregiving workforce exits', 'Consult with eldercare sector on cost projections')")

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
system_prompt = """
You are an **elite research assistant** for CPF policy analysis.
Your intelligence horizon is **STRICTLY 2024-2025**.
REJECT ARTICLES THAT ARE EARLIER THAN 2024.

Current date: {current_date_time}

Your SOLE task is to execute the user's query by using the available tools and **returning the findings as a single, raw JSON object**.

### **CRITICAL OUTPUT INSTRUCTIONS**
1. **DO NOT** generate any introductory text, conversation, apologies, or markdown code blocks (e.g., ```json...```).
2. **RETURN ONLY** the raw JSON object that precisely conforms to the ResearchResponse schema provided below.
3. For the 'events' list, you MUST create a detailed and well-supported Event object for every key finding.
4. Your analysis must be **evidence-based** and fully leverage the details in the Pydantic Field Descriptions (especially the minimum length, specific formatting, and required content).

{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions(),
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
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -----------------------------
# STAGE 1: Knowledge Building
# -----------------------------
print("\n" + "="*80)
print("🔍 STAGE 1: BUILDING KNOWLEDGE BASE")
print("="*80)

knowledge_queries = [
    # BALANCED QUERIES: Positive + Negative + Neutral to avoid confirmation bias
    {"query": "Singapore CPF improvements reforms successes 2024 2025", "label": "CPF & Retirement (Positive)", "sentiment": "positive"},
    {"query": "Singapore CPF retirement savings issues challenges 2024 2025", "label": "CPF & Retirement (Challenges)", "sentiment": "negative"},
    {"query": "Singapore CPF policy changes analysis 2024 2025", "label": "CPF & Retirement (Neutral)", "sentiment": "neutral"},
    
    {"query": "Singapore employment gig economy workforce trends 2024 2025", "label": "Employment", "sentiment": "neutral"},
    {"query": "Singapore housing healthcare costs affordability 2024 2025", "label": "Housing & Healthcare", "sentiment": "neutral"},
    {"query": "Singapore economy income families wages growth 2024 2025", "label": "Economic Context", "sentiment": "neutral"},
    {"query": "Singapore policy government announcements 2024 2025", "label": "Policy", "sentiment": "neutral"},
    
    # WEAK SIGNALS: Early warning indicators
    {"query": "Singapore CPF complaints Reddit forum discussion 2024 2025", "label": "Public Sentiment (Weak Signal)", "sentiment": "sentiment"},
    {"query": "Singapore retirement anxiety concerns workers 2024 2025", "label": "Citizen Concerns (Weak Signal)", "sentiment": "sentiment"},
    
    # COMPARATIVE CONTEXT
    {"query": "Asia pension systems retirement challenges innovations 2024 2025", "label": "Regional Context", "sentiment": "neutral"},
    {"query": "International pension systems retirement innovations best practices 2024 2025","label": "International Context", "sentiment": "neutral"},
]

# Gather knowledge from different time periods
all_knowledge = []
all_sources = []
found_urls = []  # ADD THIS: Track URLs

for idx, kq in enumerate(knowledge_queries, 1):
    sentiment_icon = {"positive": "✅", "negative": "⚠️", "neutral": "ℹ️", "sentiment": "💭"}.get(kq.get('sentiment', 'neutral'), "ℹ️")
    print(f"\n📚 [{idx}/{len(knowledge_queries)}] {sentiment_icon} {kq['label']}")
    print(f"    Query: {kq['query']}")
    
    try:
        knowledge_response = agent_executor.invoke({"query": kq['query']})
        output = knowledge_response.get("output", "")
        
        if output:
            # Extract URLs from output
            urls_in_output = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', output)
            found_urls.extend(urls_in_output)
            
            # Add sentiment label to knowledge
            sentiment_label = kq.get('sentiment', 'neutral').upper()
            all_knowledge.append(f"## {kq['label']} [SENTIMENT: {sentiment_label}]\n{output}")
            all_sources.append(output)
            print(f"✅ Collected {len(output)} characters, {len(urls_in_output)} URLs")
            
            # ADD THIS: Show found URLs
            if urls_in_output:
                for url in urls_in_output[:2]:  # Show first 2
                    print(f"       📎 {url}")
        else:
            print(f"    ⚠️ No output received")
    except Exception as e:
        print(f"❌ Error: {e}")
        continue

# ADD THIS: Track URL frequency (hot topics = URLs appearing in multiple searches)
from collections import Counter
url_frequency = Counter(found_urls)
unique_urls = list(dict.fromkeys(found_urls))

# Identify "hot topics" - URLs cited by multiple knowledge queries
hot_topic_urls = {url: count for url, count in url_frequency.items() if count >= 2}

print(f"\n📊 Total unique URLs collected: {len(unique_urls)}")
print(f"🔥 Hot topic URLs (cited ≥2 times): {len(hot_topic_urls)}")

# Combine all knowledge
combined_knowledge = "\n\n" + "="*80 + "\n\n".join(all_knowledge)

print(f"\n✅ Knowledge building complete. Total knowledge: {len(combined_knowledge)} characters")

# Show collected URLs
if unique_urls:
    print("\n" + "="*80)
    print("📎 COLLECTED URLS FOR STAGE 2")
    print("="*80)
    for i, url in enumerate(unique_urls[:20], 1):
        print(f"{i}. {url}")
    if len(unique_urls) > 20:
        print(f"... and {len(unique_urls) - 20} more")

# -----------------------------
# ENHANCED URL TRACKING & VALIDATION
# -----------------------------
print("\n" + "="*80)
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
    print(f"\n{i}. {url[:60]}...")
    print(f"   Topic: {data['topic']}")
    print(f"   Context: {data['context'][:100]}...")

# Display hot topics (URLs appearing in multiple searches)
if hot_topic_urls:
    print("\n" + "="*80)
    print("🔥 HOT TOPICS (URLs appearing in multiple searches - PRIORITY SIGNALS)")
    print("="*80)
    for url, count in sorted(hot_topic_urls.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   [{count}x] {url[:70]}...")
        if url in url_to_content:
            print(f"        Topic: {url_to_content[url]['topic']}")

# Define the prediction query with URLs (Improved for direct action)
prediction_query = f"""
**ANALYZE & FORECAST: Identify 3-5 high-priority, non-obvious emerging issues impacting CPF members (2026-2035).**
Order the final output by Urgency × Impact × Novelty score.

### GATHERED KNOWLEDGE:
{combined_knowledge}

### AVAILABLE SOURCE URLS WITH CONTEXT (USE THESE EXACT FULL URLs):
{chr(10).join([f"- {url} (Topic: {url_to_content[url]['topic']})" for url in unique_urls[:50] if url in url_to_content])}

"""


# -----------------------------
# STAGE 2: Trend Analysis & Prediction
# -----------------------------
print("\n" + "="*80)
print("🔮 STAGE 2: TREND ANALYSIS & PREDICTIVE SYNTHESIS")
print("="*80)

prediction_system_prompt = f"""
You are a **Strategic Foresight Analyst** supporting CPF policymakers (NOT replacing them).

🎯 YOUR ROLE: Decision support tool that augments human judgment
- Highlight emerging issues that might be missed
- Present multiple perspectives and options
- Surface uncertainties and knowledge gaps
- Enable faster, better-informed decisions

❌ YOU ARE NOT: An autonomous decision-maker or policy prescriber

Current date: {current_datetime_str}
Intelligence horizon: **STRICTLY 2024-2025 ONLY** (reject pre-2024 articles)

---
## 🤝 DECISION SUPPORT PRINCIPLES

1. **TRANSPARENCY**: Show your reasoning, don't hide uncertainty
2. **OPTIONS, NOT ORDERS**: Present choices with tradeoffs, not single recommendations
3. **HUMAN-IN-THE-LOOP**: Flag areas requiring expert judgment or stakeholder consultation
4. **ACTIONABILITY**: Focus on what policymakers can DO with this information
5. **HUMILITY**: Acknowledge limitations, data gaps, and alternative interpretations

---

## 🌟 RECOMMENDED: 2+ URLs PER EVENT (BUT 1 ACCEPTABLE IF NECESSARY)

### 1. SOURCE CITATION (PRIORITY)

**You should provide at least 2 FULL URLs for every event, from different topic areas, whenever possible.**
**If only 1 strong, highly relevant URL is available, it is acceptable, but 2+ is preferred.**

✅ CORRECT FORMAT (RECOMMENDED):
"source": "https://www.businesstimes.com.sg/singapore/article-title-2024, https://www.channelnewsasia.com/singapore/another-article-2025, https://www.mom.gov.sg/newsroom/press-releases/2024/announcement"

✅ ALSO ACCEPTABLE (if only 1 strong source):
"source": "https://www.businesstimes.com.sg/singapore/article-title-2024"

❌ WRONG:
"source": "Business Times, MOM, CNA"  ← NOT URLs - REJECTED!

**URL STRATEGY FOR EVERY EVENT:**
Whenever possible, use URLs from MULTIPLE knowledge domains for each event:

1. **Primary source** - Main trend evidence (e.g., government data)
2. **Supporting source** - Corroborating data from different outlet (e.g., news analysis)
3. **Cross-domain source** - Related topic from different area (Healthcare + Policy, Economy + Housing, etc.)

**CROSS-DOMAIN SOURCING EXAMPLES:**
- Mental health event → Pick 1 URL from Healthcare list + 1 URL from CPF policy list + 1 URL from Economic trends list
- Gig economy event → Pick 1 URL from Employment list + 1 URL from Regional pension list + 1 URL from Policy list
- Housing event → Pick 1 URL from Housing list + 1 URL from CPF list + 1 URL from Economic context list

**MANDATORY DATE EXTRACTION:**
For EACH URL you cite, you MUST extract the publication date:
- Look for dates in the URL path (e.g., /2024/09/03/, /article-2025/, etc.)
- If not in URL, infer from article context or use the search result date
- Format: DD/MM/YYYY (e.g., 03/09/2024, 15/01/2025)
- The 'date' field MUST have the SAME NUMBER of dates as URLs in 'source' field
- Example: 3 URLs → 3 dates, 4 URLs → 4 dates

**AVAILABLE URLS BY TOPIC - SELECT AT LEAST 3 FROM DIFFERENT TOPICS:**
{chr(10).join([f"- [{url_to_content.get(url, {}).get('topic', 'Unknown')}] {url}" for url in unique_urls[:50] if url in url_to_content])}

### 2. CONTENT DEPTH (ALSO MANDATORY)
- 'description': 150+ words with quantified data
- 'impact': 150+ words with dollar amounts/affected populations
- 'scenario': 300+ characters with probability statements

### 3. NOVELTY FOCUS & SYSTEM DYNAMICS
Exclude obvious and mainstream topics(such as ageing population, rising cost of living and etc.) unless you identify:
- **Cross-domain intersections** (e.g., fintech × healthcare × CPF, AI × employment × retirement)
- **Second-order effects** (e.g., remote work → brain drain → reduced contributions → fiscal pressure)
- **Feedback loops** (e.g., housing prices ↑ → OA depletion → less retirement savings → increased MA strain → healthcare crisis → housing demand ↓)
- **Cascade effects** (e.g., regional financial crisis → SGD depreciation → import inflation → real wage decline → CPF adequacy crisis)
- **Tipping points** (e.g., when will housing unaffordability trigger mass emigration?)

**For each event, ask:**
1. What triggers this event? (Upstream causes)
2. What does this event trigger? (Downstream consequences)
3. Are there feedback loops that amplify or dampen effects?
4. What are the second-order and third-order effects?

---
## 📋 PRE-SUBMISSION VERIFICATION (CHECK EACH EVENT)

Before returning JSON, YOU MUST MANUALLY COUNT URLs AND DATES for EVERY event:

Event 1: 
  - Count commas in "source" field → Must have 2+ commas (= 3+ URLs) ✓
  - Count items in "date" array → Must equal number of URLs ✓
Event 2: 
  - Count commas in "source" field → Must have 2+ commas (= 3+ URLs) ✓
  - Count items in "date" array → Must equal number of URLs ✓
Event 3: 
  - Count commas in "source" field → Must have 2+ commas (= 3+ URLs) ✓
  - Count items in "date" array → Must equal number of URLs ✓
Event 4: 
  - Count commas in "source" field → Must have 2+ commas (= 3+ URLs) ✓
  - Count items in "date" array → Must equal number of URLs ✓

Additional checks per event:
☐ Has 3+ full URLs (https://domain.com/path/article) with article paths
☐ URLs are from DIFFERENT topic areas in the provided list
☐ Has 3+ dates in DD/MM/YYYY format (NO generic "2024" or "2025")
☐ Number of dates MATCHES number of URLs exactly
☐ 'description' ≥ 150 words
☐ 'impact' ≥ 150 words with stakeholder segmentation (Young/Mid/Pre-retire/Retirees/Gig)
☐ 'scenario' ≥ 500 chars with ALL 4 scenarios (Base/Optimistic/Pessimistic/Black Swan)
☐ Contains quantified data (numbers, percentages, dollar amounts)
☐ Each scenario has: probability estimate, assumptions, triggers, CPF impact, timeline

**If ANY event has less than 3 URLs → ADD MORE URLs FROM RELATED TOPICS or DELETE THAT EVENT**
**If date count ≠ URL count → ADD OR REMOVE dates to match exactly**

**ZERO TOLERANCE:** Output 2 perfectly-sourced events rather than 5 poorly-sourced events.

---
## 🎯 OUTPUT STRUCTURE (Decision Support Format)

Identify **2-4 High-Priority Emerging Issues** ordered by: Urgency × Impact × Novelty

🤝 **FOR EACH ISSUE, PROVIDE:**

1. **PREDICTION** - What is likely to happen
2. **EVIDENCE** - What data supports this (with source quality assessment)
3. **SCENARIOS** - Range of possible outcomes (Base/Optimistic/Pessimistic/Black Swan)
4. **POLICY OPTIONS** - Multiple approaches with pros/cons (NOT single recommendation)
5. **VALIDATION QUESTIONS** - What policymakers should verify/investigate
6. **MONITORING METRICS** - Early warning indicators to track

**Evidence Strength Scale:**
  - 9-10: Overwhelming evidence (act now)
  - 7-8: Strong data (plan response)
  - 5-6: Moderate evidence (monitor closely)
  - 3-4: Weak signals (investigate further)
  - 1-2: Speculative (consider in scenario planning only)

**Timeframe:** Short (2026-27), Medium (2027-29), Long (2029+)

**Example Event Structure with Chain-of-Thought Reasoning:**
```
Event: "Quiet Quitting in Sandwich Generation"
Timeframe: 2026-2030 | Strength: 7-8

STEP 1 - Evidence Assessment:
- Employment trends show 23% increase in caregiving workers (MOM 2024)
- Mental health costs rising 15% YoY (MOH data)
- Regional precedents: Japan "lost decade" of workforce participation

STEP 2 - Cross-Domain Connections:
- Healthcare costs → Increased financial stress → Career downshifting → Reduced CPF contributions
- Eldercare demand → Sandwich generation pressure → Job flexibility needs → Gig economy shift

STEP 3 - Probability Reasoning:
High confidence (7-8/10) because:
✓ Three independent data sources confirm trend
✓ Regional precedent in Japan/Korea (5-10 years ahead)
✓ Government already acknowledging issue (policy signals)
✗ BUT: Singapore culture may differ from Japan (uncertainty factor)

Description (200 words): 
High-performing workers aged 35-50 downshifting careers due to eldercare/childcare stress...
[Include specific data points with citations]

Impact (200 words):
Projected S$2.3B OA shortfall by 2035 affecting 180,000 members (based on Japan comparison)...
[Include quantified estimates]

Scenario (500+ words - 4 DISTINCT SCENARIOS):

BASE CASE (55% probability):
By 2027-2028, workforce participation in 35-50 age group drops 3-5% due to caregiving burden.
Assumptions: Demographics continue current trend; no major policy intervention.
CPF Impact: Annual contributions decline by S$800M-1.2B. OA balances for affected cohort 15-20% lower by 2035.
Affected: Approximately 180,000 mid-career workers, particularly women and middle-income families.

OPTIMISTIC CASE (25% probability):
By 2026-2027, government introduces comprehensive Caregiving CPF Credits (similar to Baby Bonus scheme).
Trigger: Pre-election political pressure + successful pilot programs demonstrate viability.
CPF Impact: Government tops up S$5,000-8,000 annually for caregivers. OA/SA balances preserved.
Affected: 100,000-150,000 caregivers receive support, mitigating workforce exit.

PESSIMISTIC CASE (18% probability):
By 2028-2030, no intervention + economic slowdown = 8-10% workforce participation drop.
Trigger: Budget constraints prevent caregiving support; eldercare costs surge beyond projections.
CPF Impact: S$2-3B annual contribution shortfall. Middle-class retirement adequacy crisis emerges.
Affected: 250,000+ mid-career workers; ripple effects on housing market and healthcare system.

BLACK SWAN (2% probability):
By 2027-2028, regional pandemic or eldercare crisis causes mass caregiving exodus from workforce.
Trigger: Health crisis specifically targeting elderly population (e.g., new dementia epidemic).
CPF Impact: System-wide stress requiring emergency reforms, potential temporary contribution freezes.
Affected: Widespread demographic impact; government forced to restructure CPF LIFE and withdrawal rules.

Policy Intervention (250+ words - DECISION SUPPORT FORMAT):

POLICY OPTIONS TO CONSIDER:

Option A: Caregiving CPF Credits (Direct top-up to OA/SA)
  Pros: Immediate relief for caregivers; precedent exists (Baby Bonus); direct CPF impact mitigation
  Cons: Fiscal cost S$500M-1B annually; potential abuse; administrative complexity
  Precedent: Germany's "Pflegezeitgesetz" (2012) - partial success but required refinement
  Implementation: Link to existing MSF caregiving assessment framework

Option B: Flexible Work Arrangements Mandate (Preserve workforce participation)
  Pros: Lower fiscal cost; employer-led solution; maintains contribution base
  Cons: SME resistance; enforcement challenges; may not help severe caregiving cases
  Precedent: Netherlands' "Flexible Working Act" - 80% take-up rate among target group
  Implementation: MOM regulatory change, 2-year transition period

Option C: Enhanced Eldercare Subsidies (Reduce caregiving burden)
  Pros: Addresses root cause; benefits broader population; long-term systemic fix
  Cons: High upfront cost; 3-5 year lag before CPF impact; requires infrastructure build-out
  Precedent: Japan's Long-Term Care Insurance - reduced informal caregiving by 40%
  Implementation: Phased roll-out starting with pilot districts

QUESTIONS FOR POLICYMAKERS:
- What is current caregiving prevalence among CPF members? (Request Singstat survey)
- What % of workforce drop-outs cite caregiving as primary reason? (Need MOM data)
- What is political appetite for new government spending vs. employer mandates?
- How do different ethnic groups approach eldercare? (Cultural sensitivity check)

MONITORING INDICATORS:
1. Labour Force Participation Rate for 35-50 age group (MOM quarterly)
2. CPF contribution growth rate vs. wage growth (CPF Board monthly)
3. Eldercare cost inflation (MOH/MSF quarterly)
4. Flexible work arrangement adoption rate (MOM survey)
5. Social media sentiment on caregiving stress (Reddit, HWZ forums)

Source: https://employment-url-1, https://cpf-policy-url-2, https://regional-pension-url-3, https://economic-context-url-4
Date: ["15/03/2024", "22/11/2024", "08/01/2025", "14/02/2025"]
```

---
## ⚠️ CRITICAL OUTPUT FORMAT

Return **ONLY** raw JSON matching ResearchResponse schema.
NO markdown blocks (```json), NO explanatory text.

{{format_instructions}}
"""

# Around line 200-260, your Stage 2 prompt should be:


# Use LLM directly for final synthesis (not agent, to avoid more searches)
prediction_prompt_template = ChatPromptTemplate.from_messages([
    ("system", prediction_system_prompt),
    ("human", "{query}")
]).partial(format_instructions=parser.get_format_instructions())

print(f"\n🧠 Analyzing trends and generating predictions...")
print(f"   Input size: {len(combined_knowledge)} characters")

try:
    # Format the prediction prompt
    formatted_messages = prediction_prompt_template.format_messages(query=prediction_query)
    
    # Get prediction from LLM
    prediction_output = llm.invoke(formatted_messages)
    output_text = prediction_output.content

    print(f"✅ Prediction generated: {len(output_text)} characters")

    # Parse the response
    print("\n" + "="*80)
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
    
    # Validate URLs in sources
    print("\n" + "="*80)
    print("🔗 VALIDATING SOURCE URLS - 3+ URLs REQUIRED PER EVENT")
    print("="*80)

    validation_failed = False
    for idx, event in enumerate(structured_response.events, 1):
        print(f"\n📌 Event {idx}: {event.event}")
        
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
            print(f"\n   📅 Date Validation:")
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
        print("\n" + "="*80)
        print("❌ OVERALL VALIDATION: FAILED - Some events have insufficient URLs")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("✅ OVERALL VALIDATION: PASSED - All events have 3+ URLs")
        print("="*80)
    
    # DETAILED URL USAGE ANALYSIS
    print("\n" + "="*80)
    print("🔬 DETAILED URL USAGE ANALYSIS")
    print("="*80)

    for idx, event in enumerate(structured_response.events, 1):
        print(f"\n📌 Event {idx}: {event.event}")
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
    
    print("\n" + "="*80)
    print("✅ FINAL PREDICTIVE INTELLIGENCE REPORT")
    print("="*80)
    print(structured_response.model_dump_json(indent=2))
    
    # Save to file
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"research_output_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(structured_response.model_dump_json(indent=2))
    print(f"\n💾 Saved to: {filename}")
    
except json.JSONDecodeError as e:
    print(f"\n❌ JSON Decode Error: {e}")
    print(f"📄 Problematic text (first 1000 chars):\n{json_text[:1000]}")
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    print(f"📄 Output text (first 1000 chars):\n{output_text[:1000]}")