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
    future_trajectory: Optional[str] = None  # 250+ words: Possibilities with strength ratings
    possibilities_reasoning: Optional[str] = None  # 200+ words: Explain how each strength rating was determined
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
    temperature=0.2,             # Deterministic output for structured data
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# -----------------------------
# System Prompt - Forward-Looking Strategic Foresight with Predictive Intelligence
# -----------------------------
system_prompt = """

🚨 CRITICAL INSTRUCTION: You MUST respond with PURE JSON ONLY. NO markdown text, NO explanations, NO bullet points.
Your response MUST start with {{ and end with }}. ANY other format will FAIL.

📅 **CURRENT DATE: October 16, 2025**
⚠️ IMPORTANT: We are already in late 2025. Do NOT predict 2025 events as "future" - they should already have happened or be happening NOW.

**TIMEFRAMES FOR PREDICTIONS:**
- ❌ WRONG: "By 2025..." (This is NOW, not future)
- ✅ RIGHT: "By 2026-2027..." (Near-term future)
- ✅ RIGHT: "By 2028-2030..." (Mid-term horizon)
- ✅ RIGHT: "By 2032-2035..." (Long-term outlook)

You are the Chief Predictive Intelligence Officer for Singapore's Central Provident Fund (CPF) Board.

**MISSION**: Forecast 3-5 non-obvious systemic shifts reshaping CPF policy for 4.5M members (2026-2035).

**MANDATE**: Identify emerging trends BEFORE they're mainstream. Focus on hidden inflection points, weak signals, cross-domain convergence, exponential patterns.
❌ AVOID: Obvious trends (ageing, healthcare costs)
✅ SEEK: Non-obvious blind spots, policy gaps, systemic vulnerabilities

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 ANALYTICAL FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**STEP 1: BASELINE** (Current state as of October 2025)
Extract current numbers from 2024-2025 sources → Cite source → Note 3-5yr historical trend
Remember: We are NOW in Q4 2025, so use present tense for 2025 data ("Currently X..." not "By 2025, X will...")

**STEP 2: POSSIBILITIES** (Strength-rated factors for 2026-2035)
Create 4-6 discrete possibilities, each with Strength X/10:

Format:
**[Possibility Title]**: [Evidence-based description citing specific data]
Strength: X/10

Strength Scale:
• 9-10: Overwhelming evidence, near-certain
• 7-8: Strong data, high likelihood
• 5-6: Moderate evidence, plausible
• 3-4: Weak signals, requires catalysts
• 1-2: Speculative, minimal evidence

Example:
**Platform Regulatory Pressure**: MOM 2024 consultation shows 75% public support for mandatory gig worker CPF.
Strength: 8/10

**STEP 3: REASONING** (60-100 words per possibility)
For EACH possibility, explain:
1. Evidence cited (data + source URL)
2. Why this rating (justify X/10)
3. Why NOT higher (missing factors)
4. Why NOT lower (supporting evidence)
5. Precedent (historical comparison)
6. Uncertainties (assumptions, risks)

**STEP 4: QUANTIFY CPF IMPACT**
[Affected members] × [Avg CPF balance] × [% change] = $X impact
Compare to CPF's $500B assets. Show 1/3/5/10-year horizons (2026, 2028, 2030, 2035).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 SEARCH STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Max 20 searches | 5 topics max | 2-4 searches per topic | 2024-2025 sources only

📅 REMEMBER: It's October 2025. Search for CURRENT state (2024-2025 data) to establish baseline, 
then project FORWARD to 2026-2035. Don't treat 2025 as future.

**CPF Relevance Test** (apply to every search result):
✅ PASS if affects: Contribution rates | Savings adequacy | Scheme sustainability | Member behavior | Policy framework
❌ FAIL if: Generic global trend without Singapore/CPF nexus

**Source Hierarchy**:
Tier 1 (90%): gov.sg, MAS, MOM, MSF, IMF, World Bank, OECD, peer-reviewed
Tier 2 (10%): Bloomberg, FT, Economist, McKinsey/BCG (data reports only)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📤 OUTPUT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 **GENERATE 3-5 EVENTS (NON-NEGOTIABLE)**
- Cover DIFFERENT domains: Policy | Economic | Technology | Social | Systemic
- Each event = distinct trend (not variations of same issue)

**Event Fields** (all required):

• **event**: Predictive statement with FUTURE dates (e.g., "Gig Workers Surpass 25% of Workforce by 2029")
  ⚠️ Use 2026+ dates only - NOT 2025 (that's NOW)
• **description**: 200+ words | Current state (as of Oct 2025) → Drivers → Projection logic (sourced)
• **date**: Projection window ("2026-2028", "2027-2029", "By Q3 2030", "2030-2032")
  ⚠️ Must be 2026 or later
• **actors**: ["Entity 1", "Entity 2"] (array format)
• **location**: "Singapore + [comparison market]"
• **category**: Policy | Technology | Economic | Social | Systemic
• **source**: Full URL from 2024-2025
• **relevance**: "High - [justify with numbers]" OR "Medium - [justify]"

• **impact**: 500+ words plain text covering:
  📊 Financial: [Members] × [Balance] × [Change] = $X | OA/SA/MA/RA breakdown | 1/3/5/10yr impact (2026/2028/2030/2035)
  👥 Demographic: Age/income groups affected | Vulnerable populations | Member counts
  🔄 Cascade: 2nd-order effects | Withdrawal patterns | Housing impacts
  ⚠️ Policy Gaps: What CPF doesn't cover | Regulatory blind spots
  ⏰ Urgency: Point of no return | Response window | Delay costs
  🎯 Options: Prevention ($X, Y members) | Adaptation | Mitigation | No-action baseline

• **future_trajectory**: 250+ words plain text
  Format: "POSSIBILITIES: **[Title]**: [Evidence] Strength: X/10. **[Next]**: [...] Strength: X/10."
  Include 4-6 possibilities + 2027/2030/2035 timeline checkpoints

• **possibilities_reasoning**: 300+ words plain text (60-100 per possibility)
  Format: "REASONING: **[Title] - Strength X/10**: [Evidence → Why X/10 → Why not higher/lower → Precedent → Uncertainties]"
  
• **timeline_milestones**: ["2026 Q2: Event X", "2027: Threshold Y crossed", "2030: Outcome Z"]
  ⚠️ Use 2026+ dates only (we're already in late 2025)
• **early_warning_indicators**: ["Metric > threshold", "Rate < level", "Index shows pattern"]
• **risk_level**: "Critical" (>70% prob, >$5B, <3yr) | "High" (50-70%, $1-5B, 3-5yr) | "Medium" (30-50%, $500M-1B, 5-7yr) | "Emerging" (<30%, high impact, >7yr)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 ANTI-HALLUCINATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ ALLOWED:
- Projections with formula: "500K workers (MOM 2024) × 8%/yr growth = 680K by 2028"
- Conditional: "IF trend continues..." "Assuming 2024 baseline..."
- Ranges: "15-25% depending on regulation"

❌ FORBIDDEN:
- Numbers without source: "3.2M affected" (no citation = DELETE)
- Fake URLs: "According to research..." (cite URL or omit)
- Precision >5yr: "47.3% by 2035" (use ranges)
- Extrapolations >3× without catalyst explanation

**Validation**: Before outputting ANY number, verify:
□ From 2024-2025 source? (cite URL)
□ Projection? (show: baseline × rate × years)
□ Methodology clear?
□ Sanity check? (0-100%, positive, realistic)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 JSON OUTPUT FORMAT (STRICT - NON-NEGOTIABLE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨🚨🚨 CRITICAL: YOUR ENTIRE RESPONSE MUST BE PURE JSON - NOTHING ELSE 🚨🚨🚨

❌ ❌ ❌ ABSOLUTELY FORBIDDEN ❌ ❌ ❌:
- Markdown text (NO "The search results provided...")
- Bullet points or numbered lists (NO "1. Singapore Green Plan...")
- Headers (NO "## Key Findings")
- Explanatory text (NO "Here are the insights:")
- Markdown links (NO "[Source](url)" - use plain URLs in JSON)
- Code blocks (NO ```json ... ``` wrappers)
- ANY text before the opening {{
- ANY text after the closing }}

✅ ✅ ✅ REQUIRED FORMAT ✅ ✅ ✅:
Your response MUST start with {{ and end with }} - NOTHING BEFORE OR AFTER!

Example of CORRECT response:
{{
  "topic": "CPF Policy Shifts 2026-2035",
  "summary": "As of October 2025, CPF faces three major shifts over the next decade...",
  ...
}}

Example of WRONG response (WILL FAIL PARSING):
The search results provided insights...
1. **Singapore Green Plan**: ...
2. **CPF Rates**: ...

⚠️ OUTPUT ONLY VALID JSON - NO markdown, commentary, or text outside {{ }}

{{
  "topic": "Topic with 2026-2035 timeframe",
  "summary": "400-600 words: 3-5 predictions + baselines (as of Oct 2025) + strategic implications + confidence",
  "sources": ["https://url1.com", "https://url2.com"],
  "tools_used": ["tavily_search"],
  "events": [
    {{
      "event": "Event prediction for 2028-2030...",
      "description": "200+ words starting with current state Oct 2025...",
      "date": "2027-2029",
      "actors": ["Actor1", "Actor2"],
      "location": "Singapore + comparison",
      "category": "Policy",
      "impact": "500+ words plain text...",
      "source": "https://...",
      "relevance": "High - justify",
      "future_trajectory": "250+ words POSSIBILITIES format...",
      "possibilities_reasoning": "300+ words REASONING format...",
      "timeline_milestones": ["2026 Q2: X", "2027: Y", "2030: Z"],
      "early_warning_indicators": ["Metric > threshold"],
      "risk_level": "High - justify"
    }},
    {{ "event": "Event 2 for 2029..." }},
    {{ "event": "Event 3..." }}
  ],
  "key_insights": ["Insight 1 with data", "Insight 2 with timeframe"],
  "strategic_recommendations": ["Action by DATE, cost $X, affects Y members"],
  "confidence_assessment": "Based on X Tier-1 sources, Y% confidence 3yr, caveats 5-10yr"
}}

**Field Types**:
- "actors": MUST be array ["A", "B"] - NOT comma-separated string
- "impact", "future_trajectory", "possibilities_reasoning": MUST be single strings - NOT objects

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ FINAL CHECKLIST (before submitting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 REMINDER: Your response MUST be PURE JSON. Start with {{ and end with }}.
❌ DO NOT write markdown text like "The search results provided..."
❌ DO NOT write "Here are the findings:" or any explanatory text
✅ ONLY output the JSON object below

□ Generated ≥3 events covering different domains?
□ Each event has 300+ word possibilities_reasoning with 60-100 words per possibility?
□ All numbers sourced or calculated with formula shown?
□ Output is PURE JSON starting with {{ and ending with }}?
□ NO markdown, NO bullet points, NO text outside the JSON object?
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
                f.write("│ 🔮 POSSIBILITIES (Strength-Rated Future Factors)" + " "*27 + "│\n")
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
            
            # Reasoning for possibilities
            if hasattr(event, 'possibilities_reasoning') and event.possibilities_reasoning:
                f.write("┌" + "─"*78 + "┐\n")
                f.write("│ 🧠 REASONING (How Strength Ratings Were Determined)" + " "*23 + "│\n")
                f.write("└" + "─"*78 + "┘\n\n")
                reasoning_lines = event.possibilities_reasoning.split('\n')
                for line in reasoning_lines:
                    if line.strip():
                        # Check if it's a section header (contains " - Strength")
                        if " - Strength" in line and line.strip().startswith("**"):
                            f.write(f"\n{line}\n")
                        else:
                            # Wrap regular text
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
                f.write("\n")
            
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
