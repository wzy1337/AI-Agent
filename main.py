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
# STAGE 1: Autonomous Knowledge Discovery
# -----------------------------
print("" + "="*80)
print("🔍 STAGE 1: AUTONOMOUS HORIZON SCANNING")
print("="*80)

# -----------------------------
# Helper functions for tracking previous reports
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

# Get previous event names for deduplication
recent_files = get_all_research_files(max_n=2)
previous_events = set()
for f in recent_files:
    previous_events |= set(extract_event_names_list(f))

all_files = get_all_research_files(max_n=2)
all_event_names = []
for f in all_files:
    all_event_names.extend(extract_event_names_list(f))
event_counter = Counter(all_event_names)

# Autonomous discovery system - NO predetermined queries
autonomous_system_prompt = """
You are an autonomous horizon scanner. Your mission: DISCOVER non-obvious emerging issues that policymakers don't know to look for.

**YOUR AUTONOMOUS DISCOVERY PROCESS:**

1. **Start with a Random Intersection**: Pick an unusual combination of domains that might relate to retirement/social protection
   - Examples: "psychedelics + financial planning", "quantum computing + pension infrastructure", "longevity research + insurance markets"
   - Do NOT start with obvious topics like "AI jobs", "aging", "gig economy"

2. **Search and Analyze**: Execute the search and look for:
   - Anomalies (something that doesn't fit the pattern)
   - Contradictions (experts disagreeing in surprising ways)
   - Edge cases (extreme scenarios being discussed)
   - Cross-pollination (unrelated fields influencing each other)

3. **Follow Breadcrumbs**: Based on what SURPRISES you in the results:
   - Identify 2-3 unexpected threads or weak signals
   - Pick the MOST surprising/counter-intuitive thread
   - Formulate a new search to go deeper

4. **Chain Searches**: Each search should follow from surprises in the previous one
   - NOT: Search 1: "AI jobs" → Search 2: "automation employment" (TOO PREDICTABLE)
   - YES: Search 1: "longevity biotech 2024" → (find: CRISPR pricing) → Search 2: "gene therapy insurance models orphan drugs"

5. **Document Trail**: For each search, note:
   - What you searched for
   - What surprised you
   - What breadcrumb you're following next
   - Why this matters for retirement/social protection

**CRITICAL RULES:**
- You are AUTONOMOUS - decide your own search path
- Follow SURPRISES, not confirmations
- Go deep into rabbit holes, not broad
- Prefer niche/fringe sources over mainstream
- Each search should be MORE specific than the last
- Stop searching obvious policy topics

**CURRENT ITERATION GOAL:** {iteration_goal}

Return your search query, rationale, and what you hope to discover.
"""

# Initialize discovery tracking
discovery_log = []
all_knowledge = []
all_sources = []
found_urls = []

# Autonomous discovery loop - 10 iterations
num_iterations = 10

print(f"\n🤖 Starting {num_iterations} autonomous discovery iterations...")
print("The AI will self-direct its exploration based on surprises and weak signals.\n")

for iteration in range(1, num_iterations + 1):
    print(f"\n{'='*80}")
    print(f"🔄 ITERATION {iteration}/{num_iterations}")
    print(f"{'='*80}")
    
    # Define iteration goal based on stage
    if iteration == 1:
        iteration_goal = "Start with an unusual domain intersection that could affect retirement/social protection but isn't obvious. Avoid AI, aging, gig economy, healthcare costs."
    elif iteration <= 3:
        iteration_goal = f"Follow the most surprising thread from iteration {iteration-1}. Go deeper into the rabbit hole. What unexpected connection did you find?"
    elif iteration <= 6:
        iteration_goal = "You're mid-exploration. What anomaly or contradiction have you discovered? Follow that thread into more niche territory."
    elif iteration <= 8:
        iteration_goal = "You're deep in the rabbit hole now. What edge case or extreme scenario is being discussed in specialist communities?"
    else:
        iteration_goal = "Final iterations: What's the most counter-intuitive finding you can validate? Look for cross-domain collisions."
    
    # Build context from previous discoveries
    previous_context = ""
    if discovery_log:
        previous_context = "\n\n**PREVIOUS DISCOVERIES:**\n"
        for i, log in enumerate(discovery_log[-3:], 1):  # Last 3 for context
            previous_context += f"\nIteration {log['iteration']}:\n"
            previous_context += f"  Searched: {log['query']}\n"
            previous_context += f"  Surprise: {log['surprise']}\n"
            previous_context += f"  Next thread: {log['next_thread']}\n"
    
    # Ask AI to autonomously decide next search
    discovery_prompt = f"""
{autonomous_system_prompt}

{previous_context}

**YOUR TASK FOR THIS ITERATION:**
Decide your next search query. Explain:
1. What are you searching for?
2. Why this query (what breadcrumb from previous searches)?
3. What might surprise you if you find it?

Then execute the search using the tavily_tool.
"""
    
    try:
        # Get autonomous decision from AI
        discovery_decision = agent_executor.invoke({
            "query": discovery_prompt
        })
        
        output = discovery_decision.get("output", "")
        
        if output:
            # Extract URLs
            urls_in_output = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', output)
            found_urls.extend(urls_in_output)
            
            # Store in knowledge base
            all_knowledge.append(f"\n## ITERATION {iteration} - AUTONOMOUS DISCOVERY\n{output}")
            all_sources.append(output)
            
            print(f"\n📊 Iteration {iteration} Results:")
            print(f"   Content: {len(output)} characters")
            print(f"   URLs found: {len(urls_in_output)}")
            
            if urls_in_output:
                print(f"   Sample URLs:")
                for url in urls_in_output[:3]:
                    print(f"      📎 {url[:80]}...")
            
            # Extract search query and surprises for logging (simple heuristic)
            lines = output.split('\n')
            search_query = "autonomous discovery"
            surprise_note = "exploring weak signals"
            next_thread = "following breadcrumbs"
            
            # Try to extract actual search terms from output
            for line in lines[:20]:  # Check first 20 lines
                if '?' in line or 'search' in line.lower() or 'query' in line.lower():
                    search_query = line[:150]
                    break
            
            discovery_log.append({
                'iteration': iteration,
                'query': search_query,
                'surprise': surprise_note,
                'next_thread': next_thread,
                'url_count': len(urls_in_output),
                'content_length': len(output)
            })
            
            print(f"   ✅ Discovery logged")
            
        else:
            print(f"   ⚠️ No output from iteration {iteration}")
            
    except Exception as e:
        print(f"   ❌ Error in iteration {iteration}: {e}")
        continue

# Summary of autonomous discovery
print(f"\n{'='*80}")
print(f"📊 AUTONOMOUS DISCOVERY SUMMARY")
print(f"{'='*80}")
print(f"Total iterations: {len(discovery_log)}")
print(f"Total URLs discovered: {len(found_urls)}")
print(f"Total knowledge collected: {sum([len(k) for k in all_knowledge])} characters")

print(f"\n🔍 Discovery Trail:")
for log in discovery_log:
    print(f"\n  [{log['iteration']}] {log['query'][:100]}...")
    print(f"      → {log['url_count']} URLs, {log['content_length']} chars")

# Track URL frequency
url_frequency = Counter(found_urls)
unique_urls = list(dict.fromkeys(found_urls))
hot_topic_urls = {url: count for url, count in url_frequency.items() if count >= 2}

print(f"\n📊 URL Analysis:")
print(f"   Unique URLs: {len(unique_urls)}")
print(f"   Recurring URLs (≥2 mentions): {len(hot_topic_urls)}")

# Combine all knowledge
combined_knowledge = "=" * 80 + "\n" + "".join(all_knowledge)
print(f"\n✅ Autonomous discovery complete. Total knowledge: {len(combined_knowledge)} characters")

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

# Create a mapping of URLs to their content from discovery iterations
url_to_content = {}
for idx, (source, log) in enumerate(zip(all_sources, discovery_log), 1):
    urls_in_section = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', source)
    
    for url in urls_in_section:
        if url not in url_to_content:
            # Extract ~200 chars of context around the URL
            url_pos = source.find(url)
            context_start = max(0, url_pos - 100)
            context_end = min(len(source), url_pos + len(url) + 100)
            context = source[context_start:context_end]
            
            url_to_content[url] = {
                'topic': f"Discovery Iteration {log['iteration']}",
                'context': context,
                'full_section': source
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

# Define the prediction query with URLs (Improved for horizon scanning)
prediction_query = f"""
**HORIZON SCANNING: NON-OBVIOUS EMERGING ISSUES ONLY**

From the gathered intelligence, identify 3-5 emerging issues that meet ALL these criteria:
1. ✅ Would SURPRISE an experienced policymaker (not routine concerns)
2. ✅ NOT widely covered in policy circles yet
3. ✅ Challenges fundamental assumptions about retirement/social protection
4. ✅ Evidence from non-mainstream sources (niche research, fringe communities, cross-sector data)

🚫 **DO NOT INCLUDE** (these are TOO OBVIOUS):
- Gig economy growth
- AI/automation job displacement  
- Aging demographics
- Healthcare cost rises
- Housing affordability
- Fintech adoption
- Income inequality
- Climate impacts
- Geopolitical tensions
- Pension sustainability

✅ **LOOK FOR INSTEAD:**
- Second-order effects (e.g., AI-generated scams targeting retirees)
- Paradigm shifts (e.g., Web3 replacing traditional employment)
- Hidden vulnerabilities (e.g., infrastructure dependencies)
- Unexpected intersections (e.g., psychedelic therapy + retirement patterns)
- Fringe movements becoming mainstream (e.g., biohacking longevity)

**THINK:** What blind spots exist? What's happening at the edges that could cascade?

### GATHERED GLOBAL INTELLIGENCE:
{combined_knowledge}

### AVAILABLE SOURCE URLS:
{chr(10).join([f"- {url} (Topic: {url_to_content[url]['topic']})" for url in unique_urls[:50] if url in url_to_content])}

"""


# -----------------------------
# STAGE 2: Trend Analysis & Prediction
# -----------------------------
print("" + "="*80)
print("🔮 STAGE 2: TREND ANALYSIS & PREDICTIVE SYNTHESIS")
print("="*80)

prediction_system_prompt = f"""
You are an expert horizon scanner conducting strategic foresight for Singapore's Central Provident Fund Board (CPFB).
CRITICAL: Find NON-OBVIOUS, SURPRISING, UNDER-THE-RADAR issues CPF policymakers are NOT already tracking.

**NOVELTY FILTER (must pass 4/6):**
□ Would surprise experienced policymakers
□ NOT in standard policy briefs
□ Challenges core assumptions
□ Connects unrelated domains
□ Evidence from non-obvious sources
□ Invisible crisis brewing for 5–10 years

Examples of “non-obvious” signals:
- Early cultural or behavioural shifts from niche online communities.
- Financial experiments in Asia affecting social safety nets.
- Longevity, cognitive health, or microinsurance innovations.
- Digital tools changing savings behaviour or trust in institutions.

**PRIORITIZE:**
Highlight how informal perspectives reveal emerging risks, gaps, or public concerns that policy has not yet addressed.

**Definition:**
An emerging issue is a new, weak-signal, or rapidly developing trend with limited but credible evidence, not yet widely reported.

Current date: {current_datetime_str}
Intelligence horizon: STRICTLY 2024–2025 ONLY.

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

    # Save raw output for debugging
    debug_filename = f"debug_output_{now.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(debug_filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAW LLM OUTPUT:\n")
        f.write("="*80 + "\n")
        f.write(output_text)
        f.write("\n\n" + "="*80 + "\n")
        f.write("EXTRACTED JSON:\n")
        f.write("="*80 + "\n")
        f.write(json_text)
    print(f"   💾 Debug output saved to: {debug_filename}")

    # Handle case where LLM wraps response in {"ResearchResponse": {...}}
    import json
    try:
        parsed_json = json.loads(json_text)
        print(f"   ✅ JSON parsed successfully")
        print(f"   Top-level keys: {list(parsed_json.keys())}")

        # If wrapped, unwrap it
        if "ResearchResponse" in parsed_json and isinstance(parsed_json, dict):
            print("   ⚠️ Unwrapping nested ResearchResponse")
            json_text = json.dumps(parsed_json["ResearchResponse"])
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parsing failed: {e}")
        print(f"   First 500 chars of json_text: {json_text[:500]}")
        raise

    try:
        structured_response = parser.parse(json_text)
    except Exception as parse_error:
        print(f"   ❌ Pydantic parsing failed: {parse_error}")
        print(f"   Attempting manual validation...")
        
        # Try to identify what's wrong
        parsed_data = json.loads(json_text) if isinstance(json_text, str) else json_text
        print(f"   Required fields check:")
        print(f"      - topic: {'✅' if 'topic' in parsed_data else '❌'}")
        print(f"      - summary: {'✅' if 'summary' in parsed_data else '❌'}")
        print(f"      - source: {'✅' if 'source' in parsed_data else '❌'}")
        print(f"      - tools_used: {'✅' if 'tools_used' in parsed_data else '❌'}")
        print(f"      - events: {'✅' if 'events' in parsed_data else '❌'}")
        
        if 'events' in parsed_data and parsed_data['events']:
            print(f"   First event check:")
            first_event = parsed_data['events'][0]
            required_event_fields = ['event', 'description', 'date', 'actors', 'category', 
                                     'impact', 'scenario', 'source', 'relevance', 
                                     'confidence', 'policy_intervention', 'signal_strength']
            for field in required_event_fields:
                print(f"      - {field}: {'✅' if field in first_event else '❌'}")
        
        raise


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
    
except json.JSONDecodeError as e:
    print(f"❌ JSON Decode Error: {e}")
    print(f"📄 Problematic text (first 1000 chars):{json_text[:1000]}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    print(f"📄 Output text (first 1000 chars):{output_text[:1000]}")