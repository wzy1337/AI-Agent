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
Do not provide responses that are obvious trends like Ageing population that are obvious. Provide insights into events that may be blind spots to senior CPF board policymaker.

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
- **event, description, date, actors, location, category, impact, source, relevance**
- **future_trajectory** (250+ words): 3/5/7-year projections with probabilities (Opt 30%/Real 50%/Pess 20%)
- **timeline_milestones**: ["2027: X happens", "2029: Y threshold", "2032: Z outcome"]
- **early_warning_indicators**: ["Metric >X", "Rate exceeds Y%", "Index <Z"]
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
    print(structured_response.json(indent=2))  # Nicely formatted JSON
except Exception as e:
    print("Error parsing response:", e)
    print("Raw Response:", raw_response)
    print("Output:", output_text)
