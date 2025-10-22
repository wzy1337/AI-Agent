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
    date: List[str] = Field(description="Dates of the various sources in Day/Month/Year format, seperated by ',' ")
    actors: List[str]
    location: Optional[str]
    category: str  # Policy / Systemic risk / Public sentiment
    impact: str = Field(description="CPF-specific impact analysis (150+ words) with quantified estimates and comparable precedents")
    scenario: str = Field(min_length=300,description="Possible scenairos that could occur based on events analysed and prediction (EXAMPLE: By {year}, {event} is likely to occur with probabilty of 9/10 AND By {year},{event} is likely to occur with probabilty of 3/10. Justify")
    source: str = Field(description="MANDATORY: Full article URLs used as evidence, separated by commas (e.g., https://full-url-1, https://full-url-2). NO generic domains.")    
    relevance: str =Field(description="High/medium/low with a justification") # High / Medium / Low
    policy_intervention: str = Field(description="Suggest possible way policymakers could intervene to capitalise or alleviate this issue(150+ words)." \
    "MUST link to a specific CPF account (OA, SA, MA) or scheme")

# One research response---(contains)---> multiple events--> one event covers all the fields listed above
class ResearchResponse(BaseModel):
    topic: str
    summary: str 
    sources: List[str] = Field(description="MANDATORY:List all article URLs used as evidence, separated by commas (e.g., https://full-url-1, https://full-url-2). NO generic domains.")  
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
You are an **elite research assistant** for CPF policy analysis.
Your intelligence horizon is **STRICTLY 2024-2025**.
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
    {"query": "Singapore CPF retirement savings issues 2024 2025", "label": "CPF & Retirement"},
    {"query": "Singapore employment gig economy workforce 2024 2025", "label": "Employment"},
    {"query": "Singapore housing healthcare costs 2024 2025", "label": "Housing & Healthcare"},
    {"query": "Singapore economy income families finances 2024 2025", "label": "Economic Context"},
    {"query": "Singapore policy government announcements 2024 2025", "label": "Policy"},
    {"query": "Asia pension systems retirement challenges 2024 2025", "label": "Regional Context"},
    {"query": "International pension systems retirement challenges 2024 2025","label": "International context"},
]

# Gather knowledge from different time periods
all_knowledge = []
all_sources = []
found_urls = []  # ADD THIS: Track URLs

for idx, kq in enumerate(knowledge_queries[:9], 1):
    print(f"\n📚 [{idx}/6] Gathering: {kq['label']}")
    print(f"    Query: {kq['query']}")
    
    try:
        knowledge_response = agent_executor.invoke({"query": kq['query']})
        output = knowledge_response.get("output", "")
        
        if output:
            # Extract URLs from output
            urls_in_output = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', output)
            found_urls.extend(urls_in_output)
            
            all_knowledge.append(f"## {kq['label']}\n{output}")
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

# ADD THIS: Deduplicate URLs
unique_urls = list(dict.fromkeys(found_urls))
print(f"\n📊 Total unique URLs collected: {len(unique_urls)}")

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

# Create prediction-focused prompt
prediction_system_prompt = f"""
You are a **Strategic Foresight Analyst** for the CPFB, mandated to provide **early warning** of specific, actionable threats to CPF members (up to 4.4 million) from 2026-2035.

**🚨 MANDATE: FIND WHAT OTHERS ARE MISSING.**
**Exclude** mainstream topics (e.g., general aging, AI disruption, climate change) unless you identify a 
**novel, accelerating intersection** or a **second-order effect** specific to CPF financial security.

---
## CORE TASK & CITATION REQUIREMENTS (MANDATORY)
**Task:** Analyze the provided knowledge to identify **3-5 High-Priority Emerging Issues** (2026-2035).

**Output Criteria:**
1.  **Novelty:** Focus on **weak signals** and **cross-domain connections** (e.g., fintech regulation + healthcare costs → CPF impact).
2.  **Depth:** Minimum **150 words** per issue analysis. Be clear, convincing, and present specific numbers/data (dollar impact, affected populations, timelines) where possible.

**Source Citation (ABSOLUTE REQUIREMENT):**
-   You **MUST** cite URLs from the list provided in the query.
-   **MINIMUM 2-3 URLs per event** to demonstrate thorough research
-   Use **FULL ARTICLE URLs** (e.g., https://www.channelnewsasia.com/singapore/cpf-withdrawal-changes-gig-workers-2024-10-15).
-   DO NOT use generic domains (e.g., "www.mom.gov.sg") or invent URLs.
-   **Format:** `"source": "https://full-url-1, https://full-url-2, https://full-url-3"` (Separate multiple URLs with commas).
-   **Cross-reference:** Use URLs from DIFFERENT topics to show cross-domain analysis
-   If NO matching URL exists in the list for a point, write: `"Source: [Publication Name] - URL not available in search results"`.

**Citation Strategy:**
- Primary claim → Cite main URL
- Supporting statistics → Cite additional URL
- Comparative precedent(WHERE POSSIBLE) → Cite international/regional URL
- Each major paragraph in description/impact should reference at least one URL
- For each major paragraph cite the relvevant URLS for substantiation 


## ANALYSIS & SCENARIO FRAMEWORK
**Trend Evolution & Forecasting:**
* **Time:** Show trend progression (Short-term: 2026-27, Medium-term: 2027-29, Long-term: 2029+).
* **Strength Scale (Likelihood):**
    * 9-10: Overwhelming evidence, near-certain
    * 7-8: Strong data, high likelihood
    * 5-6: Moderate evidence, plausible
    * 3-4: Weak signals, requires catalysts
    * 1-2: Speculative, minimal evidence


**Final Ordering:**
The issues must be ordered by the combined metric: **Urgency × Impact × Novelty score.**

### FEW-SHOT EXAMPLES: SCENARIO PLANNING
Use the structure and depth below to guide your analysis of the 3-5 emerging issues. Notice the connection between the **Strength Scale** and the **specific quantification** of the threat.

| Issue Focus | Timeframe | Strength (Likelihood) | Scenario Application (Required Output Depth) |
| :--- | :--- | :--- | :--- |
| **"Quiet Quitting" and CPF Contribution Gaps in the Sandwich Generation** | 2026-2030 | 7-8 (Strong data) | The trend of high-performing individuals (35-50 y.o.) downshifting careers or coasting to manage eldercare/childcare stress. This results in stagnant wages/bonuses, leading to a projected **S$X billion shortfall** in their Ordinary Account (OA) balances by 2035, specifically impacting their Minimum Sum eligibility and housing payment capacity. |
| **Rapid Adoption of Decentralized Autonomous Organizations (DAOs) and Enforcement Complexity** | 2028-2035 | 5-6 (Moderate evidence) | A small, yet accelerating, cohort of young, high-earning gig workers (tech/creative) receiving substantial income and tokens through global DAO treasuries, entirely bypassing traditional Singaporean payroll. This creates a regulatory gap, leading to unintentional **non-compliance** in mandatory CPF contributions for an estimated **Y thousand members** by 2030, reducing their Medisave balances. |
| **Exaggerated Longevity Claims Fueling Irrational Withdrawal at 55** | 2026-2028 | 3-4 (Weak signals) | Media hype around medical breakthroughs (e.g., cell rejuvenation therapies) and increased average lifespans leads a segment of members who have recently turned 55 to withdraw the maximum amount of their **Special Account (SA)** savings, based on the **irrational belief** that they have 10-15 more years to work. This prematurely depletes their guaranteed interest nest egg, leading to an earlier-than-expected reliance on government assistance for **Z members** post-2040. |

---
### **CRITICAL OUTPUT FORMAT**
Return **ONLY** a raw JSON object matching the ResearchResponse schema.
Do NOT wrap in markdown code blocks.
Do NOT add labels or explanatory text before or after the JSON.

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
    print("🔗 VALIDATING SOURCE URLS")
    print("="*80)

    for idx, event in enumerate(structured_response.events, 1):
        print(f"\n📌 Event {idx}: {event.event}")
        
        if event.source:
            sources = [s.strip() for s in event.source.split(',')]
            
            for source in sources:
                # Check if it's a full URL
                if 'http' in source:
                    path_count = source.count('/')
                    if path_count > 3:  # Has article path
                        print(f"   ✅ Full URL: {source[:80]}...")
                    else:
                        print(f"   ⚠️ Generic domain (no article path): {source}")
                else:
                    print(f"   ❌ Not a URL: {source}")
        else:
            print(f"   ❌ No source provided")
    
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