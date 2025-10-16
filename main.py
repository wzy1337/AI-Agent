from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, tavily_tool

load_dotenv(dotenv_path="sample.env")

# -----------------------------
# Pydantic Models for Structured Output - Enhanced with Predictive Fields
# -----------------------------
class Event(BaseModel):
    event: str
    description: str  # 200+ words with forward-looking context
    date: Optional[str]  # Projection timeline (e.g., "2025-2027", "By 2030")
    actors: List[str]
    location: Optional[str]
    category: str  # Policy / Systemic risk / Technology / Economic / Demographic / Behavioral
    impact: str  # 300+ words with quantified projections
    source: str  # Full URL from 2024+ credible source
    relevance: str  # High / Medium / Low
    # NEW PREDICTIVE FIELDS
    future_trajectory: Optional[str] = None  # 250+ words: 3/5/7-year projections with probabilities
    timeline_milestones: Optional[List[str]] = None  # Specific future events with dates
    early_warning_indicators: Optional[List[str]] = None  # Metrics to monitor for trend acceleration
    risk_level: Optional[str] = None  # Critical / High / Medium / Low (based on probability × impact × speed)

class ResearchResponse(BaseModel):
    topic: str
    summary: str  # Executive summary with forward-looking implications
    sources: List[str]
    tools_used: List[str]
    events: List[Event]  # Predictive events with quantified projections

# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o",           # Or "gpt-4o-mini" for faster runs
    temperature=0,             # Deterministic output for structured data
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# -----------------------------
# System Prompt - Forward-Looking Strategic Foresight
# -----------------------------
# System Prompt - Forward-Looking Strategic Foresight with Predictive Intelligence
# -----------------------------
system_prompt = """
You are CPF Board's **Chief Predictive Intelligence Officer**. 
Your mission: FORECAST systemic shifts that will reshape retirement security for 4.5 million Singaporeans over the next 3-10 years (2025-2035).

**PREDICTION MANDATE**: Identify emerging trends BEFORE they become mainstream. 
Focus on non-obvious blind spots that senior policymakers need to know NOW to prepare for 2030+.

❌ AVOID: Obvious trends (ageing population, rising healthcare costs)
✅ SEEK: Hidden inflection points, weak signals becoming strong, cross-domain convergence, exponential acceleration patterns

**PREDICTIVE FRAMEWORK - Build projections from real data:**

1️⃣ **IDENTIFY CURRENT BASELINE** (from 2024-2025 sources)
   - What is happening NOW? Extract actual numbers from credible sources
   - Current rate/trend/adoption level (cite source with specific figure)
   - Existing trajectory based on historical data (last 3-5 years if available)

2️⃣ **PROJECT FUTURE SCENARIOS** (2027/2030/2035 milestones)
   
   For each event, create THREE possible futures based on your baseline data:
   
   🐢 **CONSERVATIVE** (30% chance) - Things move SLOWER than today
      • What if growth slows down or faces resistance?
      • Example: "Currently 15% adoption → Only 20% by 2030 (vs 33% if trend continues)"
      • Why? Regulatory delays, public skepticism, industry pushback
   
   🎯 **REALISTIC** (50% chance) - Current trend CONTINUES unchanged
      • What if nothing major changes?
      • Example: "Currently 15% adoption growing 3%/year → 33% by 2030"
      • Why? Most likely outcome - steady organic growth
   
   🚀 **ACCELERATED** (20% chance) - Major catalyst SPEEDS things up
      • What if government mandates it OR technology disrupts?
      • Example: "Currently 15% adoption → 60% by 2030 (due to new law requiring it)"
      • Why? Requires big trigger like legislation/crisis/breakthrough
   
   📐 **How to calculate:** 
   - Conservative = Current trend × 0.5 (slower)
   - Realistic = Current trend × 1.0 (same pace)
   - Accelerated = Current trend × 2-3 (faster with catalyst)
   
   🚨 **CRITICAL**: Every projection MUST reference the baseline source
   Example: "IF current 12% annual growth (Source: MAS 2024 Report) continues → 35-40% adoption by 2030"
   NOT: "Digital payments will reach 60% by 2030" (without source/calculation)

3️⃣ **DEFINE TIPPING POINTS & TRIGGERS**
   - What threshold causes irreversible change? (cite similar precedents)
   - What event would accelerate/decelerate trajectory?
   - When does gradual become exponential? (cite inflection point patterns)

4️⃣ **QUANTIFY IMPACT ON CPF** (show your work)
   - Affected member segments × average balance × % change = Total impact
   - Example: "1.2M gig workers (MOM 2024) × median $45K CPF × 15% contribution gap = $8.1B shortfall by 2030"
   - Compare to CPF's $500B+ total assets for context

**SEARCH STRATEGY** 
Execute at most 20 searches on any topic that may affect CPF only search 2024+ sources
Look UP TO 5 unique TOPICS
Limit to 2-3 SEARCH PER TOPIC
These topics can include the following RISKS/CHARACTERISTICS:
   - "Blind spots" / "Unknown risks" / "Emerging issues"
   - "What could go wrong" / "Black swan events"
   - "Future scenarios" / "2030+" timeframes
   - "Systemic threats" / "Cross-cutting impacts"

**REQUIREMENT**
**CPF RELEVANCE TEST** 
Perform validation on the searches before outputting the event
If the event affects CPF through the following:
- CPF contribution rates/volumes (wage changes, employment shifts)
- CPF member savings adequacy (healthcare costs, inflation, investment returns)
- CPF scheme sustainability ($500B+ assets at risk)
- Member behavior (withdrawal patterns, housing decisions, retirement planning)
- Regulatory/policy framework (new mandates, international standards)
OTHERWISE:
- FAIL if the event if generic global trend without Singapore/CPF nexus

**PRIORITY SOURCES** (trust hierarchy):
Tier 1 (90% weight): gov.sg, MAS, MOM, MSF, IMF, World Bank, OECD, peer-reviewed journals
Tier 2 (10% weight): Bloomberg, FT, Economist, McKinsey/BCG (data-heavy reports)
⛔ NEVER cite: Generic blogs, opinion pieces without data, promotional content

**OUTPUT REQUIREMENTS** (3-5 predictive events):

Each event MUST include:
- **event**: Clear predictive statement (e.g., "Gig Workers Surpass 25% of Workforce by 2029")
- **description** (200+ words): Current state → Drivers of change → Projection logic (with sources)
- **date**: Specific projection window ("2027-2029", "By Q3 2030")
- **actors**: Who will drive/resist this change
- **location**: Singapore + relevant comparison markets
- **category**: Policy/Technology/Economic/Social/Systemic
- **impact** (500+ words): Comprehensive CPF impact analysis with:
  
  📊 **QUANTIFIED FINANCIAL IMPACT** (show calculations):
     - Direct costs: [Affected members] × [Avg CPF balance] × [% change] = $X total impact
     - Breakdown by account type: OA/SA/MA/RA impact distribution
     - Annual vs cumulative impact (1-year, 3-year, 5-year, 10-year horizons)
     - % of CPF's $500B+ total assets affected
     - Per-member average impact (e.g., "$2,500-$5,000 per affected member")
  
  👥 **DEMOGRAPHIC SEGMENTATION**:
     - Which member groups most affected? (age brackets, income levels, employment types)
     - Vulnerable populations (low-wage, gig workers, self-employed, elderly)
     - Geographic distribution if relevant (e.g., mature estates vs new towns)
     - Member count estimates with source citations
  
  🔄 **CASCADE EFFECTS** (second-order impacts):
     - Withdrawal pattern changes → housing affordability impacts
     - Contribution gap → retirement adequacy shortfall → social costs
     - Investment return volatility → long-term savings erosion
     - Policy responses triggering behavioral changes
  
  ⚠️ **POLICY GAPS & SYSTEMIC VULNERABILITIES**:
     - What current CPF schemes DON'T cover this risk?
     - Regulatory blind spots that amplify the impact
     - Cross-border/international dimensions CPF can't control
     - Unintended consequences of existing policies
  
  ⏰ **URGENCY & TIME SENSITIVITY**:
     - Point of no return: When does gradual become irreversible?
     - Policy response window: How much time before intervention needed?
     - Compounding effects: How does delay amplify costs?
  
  🎯 **RESPONSE OPTIONS** (brief outline):
     - Preventive measures (cost $X, affects Y members)
     - Adaptive strategies (adjusting existing schemes)
     - Mitigation tactics (limiting damage if event occurs)
     - No-action scenario baseline (what happens if CPF does nothing?)
  
- **future_trajectory** (250+ words): 
  * 2027 checkpoint: [Specific metric + source-based projection]
  * 2030 milestone: [Intermediate scenario with confidence %]
  * 2035 endpoint: [Long-term outcome + key uncertainties]
  * Probability distribution: Conservative X% | Realistic Y% | Accelerated Z%
  
- **timeline_milestones**: Concrete dated predictions
  ["2026 Q2: Regulation X triggers shift", "2028: Adoption crosses 40% threshold (based on current 12%/yr growth)", "2032: Market consolidation complete"]
  
- **early_warning_indicators**: Measurable signals with thresholds
  ["MAS fintech license applications >200/year", "Gig worker CPF opt-in <30%", "Digital SGD pilot users >500K"]
  
- **risk_level**: 
  * Critical: >70% probability, >$5B impact, <3 years to materialize
  * High: 50-70% probability, $1-5B impact, 3-5 years
  * Medium: 30-50% probability, $500M-$1B impact, 5-7 years
  * Emerging: <30% probability but high impact if occurs, >7 years
  
- **source**: Full URL (MUST be real, verifiable, from 2024-2025)
- **relevance**: High/Medium (justify with affected member count or $ amount)

**ANTI-HALLUCINATION PROTOCOLS** 🚨:

✅ **ALLOWED**: 
- Projections with clear calculation: "Current 500K gig workers (MOM 2024) growing at 8%/yr = 680K by 2028"
- Conditional statements: "IF trend continues..." "Assuming 2024 baseline of X..."
- Ranges with rationale: "Between 15-25% depending on regulatory response"

❌ **FORBIDDEN**:
- Specific numbers WITHOUT source: "3.2 million members will be affected" (No source = DELETE)
- Fake URLs or generic citations: "According to research..." (Cite FULL URL or omit)
- Precise predictions beyond 5 years: "Exactly 47.3% by 2035" (Use ranges for 7+ year horizons)
- Extrapolations >3x current rate without explaining catalyst

**VALIDATION CHECKLIST** (before outputting ANY number):
□ Is this number from a 2024-2025 source? (If yes, cite URL)
□ Is this a projection? (Show calculation: baseline × growth rate × years)
□ Can I explain the methodology? (If no, use range or omit)
□ Does this pass sanity check? (Not >100%, not negative, realistic scale)

**CRITICAL OUTPUT FORMATTING RULES** 🚨:

⚠️ **YOU MUST OUTPUT VALID JSON ONLY - NO MARKDOWN, NO COMMENTARY, NO EXPLANATIONS**

Your ENTIRE response must be ONLY the JSON object below. Do NOT include:
- ❌ Markdown formatting (no ### headers, no **bold**, no bullet points outside JSON)
- ❌ Text before or after the JSON
- ❌ Code blocks or backticks
- ❌ Explanatory text like "Here are the findings:" or "Based on the search results..."
- ❌ Dictionary/object values for "actors" or "impact" fields - these MUST be arrays and strings respectively

✅ Output MUST start with {{ and end with }} - nothing else!

**JSON SCHEMA** (copy this structure exactly):
{{
  "topic": "Forward-looking topic with timeframe (2025-2030/2035)",
  "summary": "400-600 words executive summary with: 3-5 key predictions with quantified timelines, Baseline to Projection logic for each, Strategic implications for CPF Board, Confidence levels and key uncertainties",
  "sources": ["https://full-real-url-from-2024-or-2025.com", "https://another-real-url.com"],
  "tools_used": ["tavily_search"],
  "events": [
    {{
      "event": "Predictive statement",
      "description": "200+ words with source-backed baseline + projection logic",
      "date": "Projection timeframe like 2027-2029 or By Q3 2030",
      "actors": ["Entity 1", "Entity 2", "Entity 3"],
      "location": "Singapore + benchmark markets",
      "category": "Policy or Technology or Economic or Social or Systemic",
      "impact": "500+ words PLAIN TEXT STRING with calculated member/financial impact across time horizons. Include fiscal numbers, demographic segments, cascade effects, policy gaps, and urgency with deadlines and options.",
      "source": "https://actual-verifiable-url.com",
      "relevance": "High - justification with numbers OR Medium - justification with numbers",
      "future_trajectory": "250+ words PLAIN TEXT STRING: 2027 checkpoint with metrics, 2030 milestone with scenarios, 2035 endpoint with uncertainties, Probability distribution percentages",
      "timeline_milestones": ["2027 Q1: Specific event with context", "2028: Threshold crossed based on X data", "2030: Outcome milestone"],
      "early_warning_indicators": ["Metric X exceeds threshold Y", "Rate Z drops below level A", "Index B shows pattern C"],
      "risk_level": "Critical or High or Medium or Emerging - with brief justification"
    }}
  ],
  "key_insights": [
    "Insight with source-backed numbers and timeframe",
    "Cross-domain finding with quantified implications",
    "Pattern observation with statistical evidence"
  ],
  "strategic_recommendations": [
    "Action by specific DATE, estimated cost $X-Y, affects Z members, addresses specific risk",
    "Policy intervention needed by QUARTER YEAR to prevent quantified outcome"
  ],
  "confidence_assessment": "Based on X Tier-1 sources, Y data points from 2024-2025, Z% confidence in 3-year projections, caveats for 5-10 year horizons"
}}

**FIELD TYPE REQUIREMENTS**:
- "actors": MUST be array of strings ["A", "B", "C"] - NOT a single comma-separated string
- "impact": MUST be a single string (500+ words) - NOT an object/dictionary
- "future_trajectory": MUST be a single string (250+ words) - NOT an object/dictionary
- All other list fields: MUST be arrays of strings

**REMEMBER**: 
- PREDICTION requires BASELINE + LOGIC, not guessing
- EVERY NUMBER needs a SOURCE or CALCULATION
- UNCERTAINTY is honest (use ranges for long-term forecasts)
- IMPACT matters more than precision (better to say "500K-800K members affected" with reasoning than "673,492 members" without)
- OUTPUT PURE JSON ONLY - Start with {{ and end with }}
"""



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

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
# Run Query
# -----------------------------
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})

try:
    output_text = raw_response.get("output", "")

    # Extract JSON from markdown code block if present
    if "```json" in output_text:
        start = output_text.find("```json") + 7
        end = output_text.find("```", start)
        json_text = output_text[start:end].strip()
    else:
        json_text = output_text

    structured_response = parser.parse(json_text)
    
    # Generate timestamp for unique filenames
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON output
    json_filename = f"research_output_{timestamp}.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        f.write(structured_response.model_dump_json(indent=2))
    print(f"✅ JSON saved to: {json_filename}")
    
    # Save human-readable text output
    txt_filename = f"research_report_{timestamp}.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("╔" + "="*78 + "╗\n")
        f.write("║" + " "*78 + "║\n")
        f.write("║" + "CPF BOARD - STRATEGIC FORESIGHT ANALYSIS REPORT".center(78) + "║\n")
        f.write("║" + "Predictive Intelligence & Policy Planning Division".center(78) + "║\n")
        f.write("║" + " "*78 + "║\n")
        f.write("╚" + "="*78 + "╝\n\n")
        
        f.write("─" * 80 + "\n")
        f.write("REPORT METADATA\n")
        f.write("─" * 80 + "\n")
        f.write(f"📅 Generated: {datetime.now().strftime('%A, %B %d, %Y at %H:%M:%S')}\n")
        f.write(f"🔍 Research Query: {query}\n")
        f.write(f"📊 Analysis Period: 2025-2035 (10-Year Forward Outlook)\n")
        f.write(f"🎯 Target Audience: Senior CPF Policymakers & Strategic Planning Team\n")
        f.write(f"⚡ Priority Level: Strategic Foresight - Early Warning System\n\n")
        
        f.write("\n" + "╔" + "="*78 + "╗\n")
        f.write("║  TOPIC OVERVIEW" + " "*61 + "║\n")
        f.write("╚" + "="*78 + "╝\n\n")
        f.write(f"📌 {structured_response.topic}\n\n")
        
        f.write("╔" + "="*78 + "╗\n")
        f.write("║  EXECUTIVE SUMMARY" + " "*59 + "║\n")
        f.write("╚" + "="*78 + "╝\n\n")
        f.write("🎯 PURPOSE: This analysis identifies non-obvious emerging trends and blind spots\n")
        f.write("that will impact CPF's 4.5 million members over the next 3-10 years.\n\n")
        f.write("📝 SUMMARY:\n")
        f.write("─" * 80 + "\n")
        # Smart text wrapping for summary
        summary_lines = structured_response.summary.split('\n')
        for line in summary_lines:
            if line.strip():
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= 78:
                        current_line += word + " "
                    else:
                        if current_line:
                            f.write(f"{current_line.strip()}\n")
                        current_line = word + " "
                if current_line:
                    f.write(f"{current_line.strip()}\n")
            else:
                f.write("\n")
        
        f.write("\n" + "╔" + "="*78 + "╗\n")
        f.write("║  KEY EVENTS & TREND ANALYSIS" + " "*49 + "║\n")
        f.write("╚" + "="*78 + "╝\n\n")
        
        for i, event in enumerate(structured_response.events, 1):
            f.write("\n" + "┌" + "─"*78 + "┐\n")
            f.write(f"│ EVENT #{i}:" + " "*67 + "│\n")
            f.write("├" + "─"*78 + "┤\n")
            
            # Title wrapping
            title_text = event.event if hasattr(event, 'event') else event.title
            words = title_text.split()
            current = ""
            for word in words:
                if len(current) + len(word) + 1 <= 72:
                    current += word + " "
                else:
                    if current:
                        f.write(f"│ 📌 {current.strip().ljust(73)}│\n")
                    current = word + " "
            if current:
                f.write(f"│ 📌 {current.strip().ljust(73)}│\n")
            
            f.write("├" + "─"*78 + "┤\n")
            f.write(f"│ 📅 Date: {event.date.ljust(67)}│\n")
            f.write(f"│ 🏷️  Category: {event.category.ljust(63)}│\n")
            f.write(f"│ 📍 Location: {event.location.ljust(64)}│\n")
            f.write(f"│ ⭐ Relevance: {event.relevance.ljust(63)}│\n")
            
            # Risk level with color-coded indicators
            if event.risk_level:
                risk_indicators = {
                    "CRITICAL": "🔴 CRITICAL",
                    "HIGH": "🟠 HIGH",
                    "MEDIUM": "🟡 MEDIUM",
                    "LOW": "🟢 LOW"
                }
                risk_display = risk_indicators.get(event.risk_level.upper(), event.risk_level)
                f.write(f"│ ⚡ Risk Level: {risk_display.ljust(62)}│\n")
            f.write("└" + "─"*78 + "┘\n\n")
            
            # Actors section
            f.write("┌" + "─"*78 + "┐\n")
            f.write("│ 👥 KEY ACTORS:" + " "*63 + "│\n")
            f.write("└" + "─"*78 + "┘\n")
            for actor in event.actors:
                f.write(f"  • {actor}\n")
            f.write("\n")
            
            # Description section
            f.write("┌" + "─"*78 + "┐\n")
            f.write("│ 📝 DESCRIPTION:" + " "*61 + "│\n")
            f.write("└" + "─"*78 + "┘\n")
            words = event.description.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= 78:
                    current_line += word + " "
                else:
                    if current_line:
                        f.write(f"{current_line.strip()}\n")
                    current_line = word + " "
            if current_line:
                f.write(f"{current_line.strip()}\n")
            f.write("\n")
            
            # Impact analysis with smart section detection
            f.write("╔" + "="*78 + "╗\n")
            f.write("║ 💡 COMPREHENSIVE IMPACT ANALYSIS (500+ words)" + " "*30 + "║\n")
            f.write("╚" + "="*78 + "╝\n\n")
            
            # Smart section parser
            impact_text = event.impact
            sections = {
                "📊 FINANCIAL": "FINANCIAL",
                "👥 DEMOGRAPHIC": "DEMOGRAPHIC",
                "🔄 CASCADE": "CASCADE",
                "⚠️ POLICY": "POLICY",
                "⏰ URGENCY": "URGENCY",
                "🎯 RESPONSE": "RESPONSE"
            }
            
            lines = impact_text.split('\n')
            for line in lines:
                # Check if line is a section marker
                section_found = False
                for icon_label, marker in sections.items():
                    if marker in line.upper() and len(line) < 50:
                        f.write(f"\n{icon_label}:\n")
                        f.write("─" * 80 + "\n")
                        section_found = True
                        break
                
                if not section_found and line.strip():
                    # Preserve bullet points or wrap text
                    if line.strip().startswith('•') or line.strip().startswith('-') or line.strip().startswith('*'):
                        f.write(f"{line}\n")
                    else:
                        words = line.split()
                        current_line = ""
                        for word in words:
                            if len(current_line) + len(word) + 1 <= 78:
                                current_line += word + " "
                            else:
                                if current_line:
                                    f.write(f"{current_line.strip()}\n")
                                current_line = word + " "
                        if current_line:
                            f.write(f"{current_line.strip()}\n")
                elif not section_found:
                    f.write("\n")
            
            f.write("\n")
            f.write("\n")
            
            # Future trajectory
            if event.future_trajectory:
                f.write("┌" + "─"*78 + "┐\n")
                f.write("│ 🔮 FUTURE TRAJECTORY (Conservative/Realistic/Accelerated)" + " "*18 + "│\n")
                f.write("├" + "─"*78 + "┤\n")
                trajectory_lines = event.future_trajectory.split('\n')
                for line in trajectory_lines:
                    if line.strip():
                        words = line.split()
                        current_line = ""
                        for word in words:
                            if len(current_line) + len(word) + 1 <= 74:
                                current_line += word + " "
                            else:
                                if current_line:
                                    f.write(f"│ {current_line.strip().ljust(76)}│\n")
                                current_line = word + " "
                        if current_line:
                            f.write(f"│ {current_line.strip().ljust(76)}│\n")
                    else:
                        f.write("│" + " "*78 + "│\n")
                f.write("└" + "─"*78 + "┘\n\n")
            
            # Timeline milestones
            if event.timeline_milestones:
                f.write("┌" + "─"*78 + "┐\n")
                f.write("│ 📊 CRITICAL MILESTONES & CHECKPOINTS" + " "*40 + "│\n")
                f.write("└" + "─"*78 + "┘\n")
                for milestone in event.timeline_milestones:
                    words = milestone.split()
                    current_line = "  ▸ "
                    for word in words:
                        if len(current_line) + len(word) + 1 <= 78:
                            current_line += word + " "
                        else:
                            f.write(f"{current_line.strip()}\n")
                            current_line = "    " + word + " "
                    if current_line.strip() != "▸":
                        f.write(f"{current_line.strip()}\n")
                f.write("\n")
            
            # Early warning indicators
            if event.early_warning_indicators:
                f.write("┌" + "─"*78 + "┐\n")
                f.write("│ ⚡ EARLY WARNING INDICATORS (Monitor These Signals)" + " "*24 + "│\n")
                f.write("└" + "─"*78 + "┘\n")
                for indicator in event.early_warning_indicators:
                    words = indicator.split()
                    current_line = "  ⚠️  "
                    for word in words:
                        if len(current_line) + len(word) + 1 <= 78:
                            current_line += word + " "
                        else:
                            f.write(f"{current_line.strip()}\n")
                            current_line = "     " + word + " "
                    if len(current_line.strip()) > 2:
                        f.write(f"{current_line.strip()}\n")
                f.write("\n")
            
            # Source with tier classification
            f.write("┌" + "─"*78 + "┐\n")
            f.write("│ 🔗 SOURCE VERIFICATION" + " "*54 + "│\n")
            f.write("└" + "─"*78 + "┘\n")
            
            # Determine source tier
            tier1_domains = ['gov.sg', 'imf.org', 'worldbank.org', 'oecd.org', 'mom.gov.sg', 'msf.gov.sg']
            tier2_domains = ['bloomberg.com', 'ft.com', 'reuters.com', 'mckinsey.com', 'economist.com']
            
            source_tier = "🥈 Tier 2"
            for domain in tier1_domains:
                if domain in event.source.lower():
                    source_tier = "🥇 Tier 1"
                    break
            
            f.write(f"  Classification: {source_tier} (Government/Institutional)\n")
            
            # Wrap source URL
            if len(event.source) > 78:
                words = event.source.split()
                current_line = "  URL: "
                for word in words:
                    if len(current_line) + len(word) + 1 <= 78:
                        current_line += word + " "
                    else:
                        f.write(f"{current_line.strip()}\n")
                        current_line = "       " + word + " "
                if current_line.strip() != "URL:":
                    f.write(f"{current_line.strip()}\n")
            else:
                f.write(f"  URL: {event.source}\n")
            f.write("\n")
        
        # Key insights section
        f.write("\n" + "╔" + "="*78 + "╗\n")
        f.write("║  🎯 STRATEGIC INSIGHTS & SYNTHESIS" + " "*42 + "║\n")
        f.write("╚" + "="*78 + "╝\n\n")
        if hasattr(structured_response, 'key_insights') and structured_response.key_insights:
            for i, insight in enumerate(structured_response.key_insights, 1):
                f.write("┌" + "─"*78 + "┐\n")
                f.write(f"│ Insight #{i}" + " "*66 + "│\n")
                f.write("└" + "─"*78 + "┘\n")
                
                words = insight.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= 78:
                        current_line += word + " "
                    else:
                        if current_line:
                            f.write(f"{current_line.strip()}\n")
                        current_line = word + " "
                if current_line:
                    f.write(f"{current_line.strip()}\n")
                f.write("\n")
        else:
            f.write("⚠️  No key insights provided in this analysis.\n\n")
        
        # Strategic recommendations
        f.write("╔" + "="*78 + "╗\n")
        f.write("║  📋 STRATEGIC RECOMMENDATIONS FOR CPF BOARD" + " "*32 + "║\n")
        f.write("╚" + "="*78 + "╝\n\n")
        if hasattr(structured_response, 'strategic_recommendations') and structured_response.strategic_recommendations:
            for i, rec in enumerate(structured_response.strategic_recommendations, 1):
                f.write("┌" + "─"*78 + "┐\n")
                f.write(f"│ Recommendation #{i}" + " "*59 + "│\n")
                f.write("└" + "─"*78 + "┘\n")
                
                words = rec.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= 78:
                        current_line += word + " "
                    else:
                        if current_line:
                            f.write(f"{current_line.strip()}\n")
                        current_line = word + " "
                if current_line:
                    f.write(f"{current_line.strip()}\n")
                f.write("\n")
        else:
            f.write("⚠️  No strategic recommendations provided in this analysis.\n\n")
        
        # Confidence assessment
        f.write("╔" + "="*78 + "╗\n")
        f.write("║  📊 CONFIDENCE & VALIDATION ASSESSMENT" + " "*38 + "║\n")
        f.write("╚" + "="*78 + "╝\n\n")
        if hasattr(structured_response, 'confidence_assessment') and structured_response.confidence_assessment:
            words = structured_response.confidence_assessment.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= 78:
                    current_line += word + " "
                else:
                    if current_line:
                        f.write(f"{current_line.strip()}\n")
                    current_line = word + " "
            if current_line:
                f.write(f"{current_line.strip()}\n")
        else:
            f.write("⚠️  No confidence assessment provided.\n")
        
        # Footer
        f.write("\n\n" + "╔" + "="*78 + "╗\n")
        f.write("║" + " "*78 + "║\n")
        f.write("║" + "END OF REPORT".center(78) + "║\n")
        f.write("║" + " "*78 + "║\n")
        f.write("║" + f"Generated by CPF Strategic Foresight AI System v2.0".center(78) + "║\n")
        f.write("║" + f"Report ID: {timestamp}".center(78) + "║\n")
        f.write("║" + " "*78 + "║\n")
        f.write("╚" + "="*78 + "╝\n\n")
        
        f.write("─" * 80 + "\n")
        f.write("DISCLAIMER:\n")
        f.write("This report is generated using AI-assisted research and analysis. All findings\n")
        f.write("should be verified by human subject matter experts before policy decisions.\n")
        f.write("Quantitative projections are probabilistic estimates based on available data\n")
        f.write("and should be treated as directional indicators rather than precise forecasts.\n")
        f.write("─" * 80 + "\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("METADATA\n")
        f.write("="*80 + "\n")
        f.write(f"Sources: {len(structured_response.sources)}\n")
        f.write(f"Tools Used: {', '.join(structured_response.tools_used)}\n")
        f.write(f"Events Analyzed: {len(structured_response.events)}\n\n")
        
        f.write("SOURCES:\n")
        f.write("-"*80 + "\n")
        for i, source in enumerate(structured_response.sources, 1):
            f.write(f"{i}. {source}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Readable report saved to: {txt_filename}")
    print(f"\n📄 Total events: {len(structured_response.events)}")
    print(f"📚 Total sources: {len(structured_response.sources)}")
    
except Exception as e:
    print("Error parsing response:", e)
    print("Raw Response:", raw_response)
    print("Output:", output_text)
