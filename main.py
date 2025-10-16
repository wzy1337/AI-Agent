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
system_prompt = """
You are CPF Board's **Strategic Foresight Analyst**. Identify EMERGING TRENDS that will impact CPF members over 3-7 years (2025-2032).
Do not provide responses that are obvious trends like Ageing population. Provide insights into events that may be blind spots to senior CPF board policymaker.

**Mission**: Predict future challenges BEFORE they become crises. Focus on:
- Weak signals → Major disruptions
- Second-order effects (ripple impacts)
- Scenario planning (optimistic/realistic/pessimistic with %)
- Early warning metrics
- Proactive policy options

**Search Strategy** (2-3 queries per category, 2024+ sources):
A) **Tech/Digital**: Fintech disruption, AI automation, digital identity, crypto/DeFi
B) **Economic**: Gig economy growth, future of work, platform workers, restructuring
C) **Social**: Longevity risk, retirement patterns, housing pressure, cross-border workers  
D) **Systemic**: Climate finance, geopolitical risks, pension sustainability, inflation

**Priority Sources**: gov.sg > IMF/WorldBank > Academic > Bloomberg/FT > Consulting firms

**Output Requirements** (3-5 events, 200+ word descriptions, 300+ word impacts):
- **event, description(CPF senior policymakers would want to read this so provide a convicing and detailed 100 word writeup ), date, actors, location, category, impact, source, relevance**
- **future_trajectory** (250+ words): 3/5/7-year projections with probabilities (Opt 30%/Real 50%/Pess 20%)
- **timeline_milestones**: ["2027: X happens", "2029: Y threshold", "2032: Z outcome"]
- **early_warning_indicators**: ["Metric >X", "Rate exceeds Y%", "Index <Z"]
  Examples: "Gig worker growth >15% annually triggers mandatory CPF review"
            "Fintech adoption >30% of members requires digital security audit"
            "CPF fund returns <3% for 2 years demands investment strategy overhaul"
- **risk_level**: Critical (>75% prob, >$1B, <2yr) / High (50-75%, $500M-$1B, 2-3yr) / Medium (25-50%, $100M-$500M, 3-5yr) / Low

**Analysis Standards**:
✓ Quantify all claims: "300K-450K members (10-15%)", "$500M-$750M by 2028"
✓ Show calculations: "20% × 2.5M workforce × $500/mo = $3B annual gap"
✓ Compare scenarios: "Best: 5% | Realistic: 15% | Worst: 30%"
✓ Identify tipping points: "Irreversible after Q2 2026 without intervention"
✓ Benchmark: "Australia faced this in 2018, impact was X%"

**JSON Output** (valid JSON only, no markdown):
{{
  "topic": "Brief topic",
  "summary": "300-500 words with quantified forward projections",
  "sources": ["https://full-url1.com", "https://full-url2.com"],
  "tools_used": ["tavily_search"],
  "events": [...all fields above...],
  "key_insights": ["Insight with % or $", "Finding with timeline"],
  "strategic_recommendations": ["Action by DATE, cost $X-Y, impact: Z members"],
  "confidence_assessment": "X sources, Y% tier-1, Z% confidence"
}}
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
        f.write("="*80 + "\n")
        f.write("CPF STRATEGIC FORESIGHT ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Query: {query}\n\n")
        
        f.write("="*80 + "\n")
        f.write("TOPIC\n")
        f.write("="*80 + "\n")
        f.write(f"{structured_response.topic}\n\n")
        
        f.write("="*80 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"{structured_response.summary}\n\n")
        
        f.write("="*80 + "\n")
        f.write("EVENTS ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        for i, event in enumerate(structured_response.events, 1):
            f.write(f"\n{'─'*80}\n")
            f.write(f"EVENT {i}: {event.event}\n")
            f.write(f"{'─'*80}\n\n")
            
            f.write(f"📅 Date: {event.date}\n")
            f.write(f"🏷️  Category: {event.category}\n")
            f.write(f"📍 Location: {event.location}\n")
            f.write(f"⭐ Relevance: {event.relevance}\n")
            if event.risk_level:
                f.write(f"⚠️  Risk Level: {event.risk_level}\n")
            f.write(f"\n👥 Key Actors:\n")
            for actor in event.actors:
                f.write(f"   • {actor}\n")
            
            f.write(f"\n📝 DESCRIPTION:\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{event.description}\n\n")
            
            f.write(f"💡 IMPACT ANALYSIS:\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{event.impact}\n\n")
            
            if event.future_trajectory:
                f.write(f"🔮 FUTURE TRAJECTORY (3/5/7-Year Projections):\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{event.future_trajectory}\n\n")
            
            if event.timeline_milestones:
                f.write(f"📊 KEY MILESTONES:\n")
                f.write(f"{'-'*80}\n")
                for milestone in event.timeline_milestones:
                    f.write(f"   ▸ {milestone}\n")
                f.write("\n")
            
            if event.early_warning_indicators:
                f.write(f"⚡ EARLY WARNING INDICATORS:\n")
                f.write(f"{'-'*80}\n")
                for indicator in event.early_warning_indicators:
                    f.write(f"   ⚠️  {indicator}\n")
                f.write("\n")
            
            f.write(f"🔗 SOURCE:\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{event.source}\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*80 + "\n")
        if hasattr(structured_response, 'key_insights') and structured_response.key_insights:
            for i, insight in enumerate(structured_response.key_insights, 1):
                f.write(f"{i}. {insight}\n")
        else:
            f.write("No key insights provided.\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("STRATEGIC RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        if hasattr(structured_response, 'strategic_recommendations') and structured_response.strategic_recommendations:
            for i, rec in enumerate(structured_response.strategic_recommendations, 1):
                f.write(f"{i}. {rec}\n")
        else:
            f.write("No strategic recommendations provided.\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("CONFIDENCE ASSESSMENT\n")
        f.write("="*80 + "\n")
        if hasattr(structured_response, 'confidence_assessment') and structured_response.confidence_assessment:
            f.write(f"{structured_response.confidence_assessment}\n")
        else:
            f.write("No confidence assessment provided.\n")
        
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
