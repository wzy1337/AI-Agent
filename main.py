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


load_dotenv(dotenv_path="sample.env")

now = datetime.datetime.now()
current_datetime_str = now.strftime("%Y-%m-%d, %A. Time: %H:%M:%S. Current timezone is UTC+8.")

# -----------------------------
# Pydantic Models for Structured Output
# -----------------------------
class Event(BaseModel):
    event: str
    description: str = Field(description="Detailed context (150+ words) including international comparisons and case studies")
    date: Optional[str]
    actors: List[str]
    location: Optional[str]
    category: str  # Policy / Systemic risk / Public sentiment
    impact: str = Field(description="CPF-specific impact analysis (150+ words) with quantified estimates and comparable precedents")
    scenairo: str = Field(description="Possible scenairos that could occur based on events analysed and prediction (e.g : Event 1 with probabilty )")
    source: str
    relevance: str  # High / Medium / Low

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: List[str]
    tools_used: List[str]
    events: List[Event]

# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o",           # Or "gpt-4o-mini" for faster runs
    temperature=0.5,             # Deterministic output for structured data
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
You are a senior intelligence analyst for Central Provident Fund Board (CPFB).
**Mission:** Identify emerging issues affecting CPF members (2026-2030). Ignore obvious trends like aging or inflation.

**Current date:** {current_date_time}

** You will undergo two stages: First perform an analysis of the current issues then NEXT, perform trend analysis and prediction
**You can analyse current local new articles, forums, international news for understanding of current issues**

## Event Extraction Requirements
**Requirements:**
- Focus on non-obvious, emerging issues
- Provide specific data: dollar amounts, percentages, affected populations
- Include credible sources with URLs
- Minimum 150 words per description and impact field
- **MANDATORY: Include international comparisons** - Show how similar issues played out in other countries
  Example: "Similar to Australia's superannuation early access scheme in 2020, which saw 3.5M withdrawals totaling $38B AUD"

### Output Requirements
- Include specific dates, not just year
- Cite comparable international cases with outcomes
- Show what worked/failed in other countries
- Only include events from the past 12 months
- Order events by relevance score (High → Medium → Low)

### **CRITICAL OUTPUT FORMAT**
Return ONLY a raw JSON object that directly matches the ResearchResponse schema below.
Do NOT wrap it in markdown code blocks.

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

# Calculate dates for temporal queries

knowledge_queries = [
    # Broad societal trends
    {"query": "Singapore emerging social trends 2024 2025", "label": "Singapore Social Trends"},
    {"query": "Southeast Asia economic political developments 2025", "label": "Regional Developments"},
    {"query": "global technology disruption employment workforce 2025", "label": "Tech Disruption"},
    {"query": "financial system changes digital currency fintech Asia 2025", "label": "Financial Innovation"},
    
    # Open-ended issue discovery
    {"query": "Singapore controversies debates public concern 2025", "label": "Singapore Public Concerns"},
    {"query": "Asia retirement savings challenges problems 2025", "label": "Asia Retirement Issues"},
    {"query": "unexpected economic risks financial stability 2024 2025", "label": "Economic Risks"},
    {"query": "behavioral changes lifestyle trends millennials Gen Z Asia", "label": "Generational Shifts"},
    
    # Weak signal detection
    {"query": "emerging technologies societal impact future of work", "label": "Emerging Tech Impact"},
    {"query": "climate change economic consequences Asia infrastructure", "label": "Climate Economic Impact"},
    {"query": "geopolitical tensions trade regional stability Southeast Asia", "label": "Geopolitical Stability"},
    {"query": "health trends aging longevity healthcare costs Asia", "label": "Health & Longevity"},
    
    # Wild cards
    {"query": "black swan events tail risks financial markets 2024", "label": "Black Swan Events"},
    {"query": "social unrest protests inequality Asia 2024", "label": "Social Instability"},
    {"query": "regulatory changes government policy shifts Singapore 2024", "label": "Regulatory Changes"},
]

# Gather knowledge from different time periods
all_knowledge = []
all_sources = []

for idx, kq in enumerate(knowledge_queries[:6], 1):  # Limit to 6 queries to save time
    print(f"\n📚 [{idx}/6] Gathering: {kq['label']}")
    print(f"    Query: {kq['query']}")
    
    try:
        knowledge_response = agent_executor.invoke({"query": kq['query']})
        output = knowledge_response.get("output", "")
        
        if output:
            all_knowledge.append(f"## {kq['label']}\n{output}")
            all_sources.append(output)
            print(f"    ✅ Collected {len(output)} characters")
        else:
            print(f"    ⚠️ No output received")
    except Exception as e:
        print(f"    ❌ Error: {e}")
        continue

# Combine all knowledge
combined_knowledge = "\n\n" + "="*80 + "\n\n".join(all_knowledge)

print(f"\n✅ Knowledge building complete. Total knowledge: {len(combined_knowledge)} characters")

# Define the prediction query
prediction_query = """
Analyze the gathered knowledge and identify 3-5 emerging issues that will significantly 
impact CPF members from 2026-2030. Focus on non-obvious trends, weak signals, and 
cross-domain connections. Exclude mainstream topics like AI automation, aging, or climate change 
unless you can show a novel intersection or accelerating trend.
"""


# -----------------------------
# STAGE 2: Trend Analysis & Prediction
# -----------------------------
print("\n" + "="*80)
print("🔮 STAGE 2: TREND ANALYSIS & PREDICTIVE SYNTHESIS")
print("="*80)

# Create prediction-focused prompt
prediction_system_prompt = """
You are a strategic foresight analyst for CPFB.

**Task:** Analyze the provided knowledge and identify 3-5 emerging issues affecting CPF members (2026-2030).

**Analysis Framework:**

1. **Trend Identification:** Which issues are accelerating? Which are weak signals?
2. **Cross-domain Synthesis:** How do issues interact? What second-order effects emerge?
3. **Temporal Forecasting:** Short-term (2026-27), Medium-term (2027-29), Long-term (2029+)
4. **Impact Quantification:** Provide specific numbers:
   - Dollar impact: "$X billion affects Y members"
   - Populations: "180,000 workers aged 30-45"
   - Timelines: "Q2 2026: First signs, Q4 2027: Full impact"
5. **Scenairo planning based on criteria**
Strength Scale:
-• 9-10: Overwhelming evidence, near-certain
-• 7-8: Strong data, high likelihood
-• 5-6: Moderate evidence, plausible
-• 3-4: Weak signals, requires catalysts
-• 1-2: Speculative, minimal evidence

**What to Include:**
✅ Non-obvious issues not yet on policymakers' radar
✅ Issues with delayed impacts
✅ Cross-domain connections (e.g., tech + housing → CPF impact)
✅ Specific statistics and credible sources with URLs
✅ Minimum 150 words per impact analysis

**What to Exclude:**
❌ Obvious trends (aging, cost of living)
❌ Issues already well-known
❌ Vague predictions without data

### **Output Requirements**
- Identify 3-5 HIGH-PRIORITY emerging issues
- Provide DETAILED impact analysis (minimum 150 words per issue), YOU ARE PRESENTING TO SENIOR POLICYMAKER BE CLEAR AND CONVINCING

- Include evidence from multiple time periods showing trend evolution
- Include the URLS used
- 
- Prioritize Singapore/Southeast Asia context
- Order by: Urgency × Impact × Novelty score ###

### **CRITICAL OUTPUT FORMAT**
Return ONLY a raw JSON object matching the ResearchResponse schema.
Do NOT wrap in markdown code blocks.
Do NOT add labels before the JSON.

{format_instructions}
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

    print(f"   ✅ Prediction generated: {len(output_text)} characters")

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