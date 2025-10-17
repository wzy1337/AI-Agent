from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, tavily_tool
import datetime

load_dotenv(dotenv_path="sample.env")

now = datetime.datetime.now()
current_datetime_str = now.strftime("%Y-%m-%d, %A. Time: %H:%M:%S. Current timezone is UTC+8.")

# -----------------------------
# Pydantic Models for Structured Output
# -----------------------------
class Event(BaseModel):
    event: str
    description: str
    date: Optional[str]
    actors: List[str]
    location: Optional[str]
    category: str  # Policy / Systemic risk / Public sentiment
    impact: str
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
### **Core Directive**
You are senior intelligence bot for Central Provident Fund Board (CPFB). 
Your task is to predict emerging issues that are likely to significantly affect CPF and/or its members in the future. 
**IMPORTANT TO IGNORE OBVIOUS DEMOGRAPHIC TRENDS(AGEING POPULATION, RISING COST OF LIVING) AS THEY ARE ALREADY OBVIOUS**

# Current Temporal Context (MANDATORY)
The current date and time is: **{current_date_time}**
Use this information to correctly resolve any relative time references (e.g., 'today', 'tomorrow', 'next week').

**You can analyse current local new articles, forums, international news for understanding of current issues**
**From there you are to predictions on issues which will emerge as significant problems**

** Search up to 5 unique topics that are currently underlying and are not known to CPFB senior policy maker but is significant. Widen your search as the issues may NOT OBVIOUS**

### Event Extraction Requirements

For each identified issue, provide DETAILED analysis:

**Impact Field (minimum 150 words):**
- Quantify financial implications where possible (e.g., "could affect $X billion in CPF savings")
- Identify specific member segments affected (e.g., "primarily impacts members aged 40-55 in gig economy")
- Provide timeline (short-term: 0-2 years, medium-term: 2-5 years, long-term: 5+ years)
- Suggest preliminary mitigation strategies
- Include both direct and indirect effects

**Relevance Field (minimum 100 words):**
- Explain specific operational impact on CPF systems
- Map out causal chain from issue to member impact
- Justify urgency rating (High/Medium/Low) with specific criteria
- Compare to similar historical precedents if applicable

**Description Field (minimum 100 words):**
- Provide full context and background
- Explain why this is emerging NOW
- Include supporting statistics or data points
- Mention conflicting viewpoints if relevant

### Output Requirements
- Prioritize Singapore and regional (Southeast Asia) sources
- Include specific dates, not just year
- Merge duplicate events across multiple sources (show merge count)
- Only include events from the past 12 months
- Order events by relevance score (High → Medium → Low)

### **CRITICAL OUTPUT FORMAT**
Return ONLY a raw JSON object that directly matches the ResearchResponse schema below.
Do NOT wrap it in markdown code blocks.
Do NOT add any labels like "ResearchResponse:" before the JSON.
Do NOT include any explanatory text before or after the JSON.

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
from dateutil.relativedelta import relativedelta

knowledge_queries = []
for months_ago in [0, 3, 6, 9]:  # Current, 3, 6, 9 months ago
    date = now - relativedelta(months=months_ago)
    month_year = date.strftime("%B %Y")
    
    knowledge_queries.append({
        "query": f"Singapore CPF retirement pension emerging issues trends {month_year}",
        "label": f"CPF Issues - {month_year}"
    })
    knowledge_queries.append({
        "query": f"Southeast Asia retirement pension system changes {month_year}",
        "label": f"Regional Trends - {month_year}"
    })

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

# -----------------------------
# STAGE 2: Trend Analysis & Prediction
# -----------------------------
print("\n" + "="*80)
print("🔮 STAGE 2: TREND ANALYSIS & PREDICTIVE SYNTHESIS")
print("="*80)

# Create prediction-focused prompt
prediction_system_prompt = """
### **Core Directive**
You are a senior strategic foresight analyst for Central Provident Fund Board (CPFB).

You have been provided with knowledge gathered over the past 12 months about retirement, pension, and CPF-related issues.
Your task is to perform PREDICTIVE ANALYSIS to identify emerging issues that will significantly impact CPF members in 2026-2030.

### **Analysis Framework**

**1. TREND IDENTIFICATION**
- Which issues appear repeatedly across time periods? (Growing trends)
- Which issues appeared once but are gaining momentum? (Weak signals)
- Which issues are declining in importance? (Ignore these)

**2. CROSS-DOMAIN SYNTHESIS**
- How do different issues interact and compound each other?
- What second-order and third-order effects might emerge?
- Which combinations create systemic risks?

**3. TEMPORAL FORECASTING**
- Short-term (2026-2027): What will materialize soon?
- Medium-term (2027-2029): What's building momentum?
- Long-term (2029-2030): What delayed impacts will hit?

**4. WEAK SIGNAL DETECTION**
Prioritize issues that are:
- Currently SMALL but GROWING exponentially
- Have DELAYED impacts (won't be obvious until later)
- Result from MULTIPLE converging factors
- NOT yet on policymakers' radar

**5. STRATEGIC SURPRISES**
- What low-probability, high-impact events could emerge?
- What assumptions might be wrong?
- What are the "unknown unknowns"?

### **Exclusion Criteria**
❌ IGNORE obvious demographic trends (aging population, rising cost of living)
❌ IGNORE issues already well-known to policymakers
❌ IGNORE short-term fluctuations without long-term implications

### **Output Requirements**
- Identify 3-5 HIGH-PRIORITY emerging issues
- Provide DETAILED impact analysis (minimum 150 words per issue)
- Include evidence from multiple time periods showing trend evolution
- Quantify impacts where possible
- Prioritize Singapore/Southeast Asia context
- Order by: Urgency × Impact × Novelty score

### **CRITICAL OUTPUT FORMAT**
Return ONLY a raw JSON object matching the ResearchResponse schema.
Do NOT wrap in markdown code blocks.
Do NOT add labels before the JSON.

{format_instructions}
"""

# Create prediction prompt with gathered knowledge
prediction_query = f"""
Based on the following knowledge gathered over multiple time periods:

{combined_knowledge}

---

Now perform your PREDICTIVE ANALYSIS to identify the TOP 3-5 emerging issues that will significantly impact CPF and its members in 2026-2030.

Focus on:
- Issues that are GROWING across the time periods analyzed
- WEAK SIGNALS that appear small now but could explode
- CONVERGENT RISKS from multiple factors combining
- Issues with DELAYED IMPACTS that aren't obvious yet

Provide detailed, evidence-based predictions with specific timelines and quantified impacts.
"""

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