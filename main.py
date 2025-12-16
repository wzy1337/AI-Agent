# -*- coding: utf-8 -*-
import sys
import io

# Force UTF-8 encoding for stdout/stderr to avoid Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from dotenv import load_dotenv
from typing import List, Optional, Dict
<<<<<<< HEAD
# CURRENT: Gemini implementation
from langchain_google_genai import ChatGoogleGenerativeAI
# FALLBACK: For OpenAI, replace with: from langchain_openai import ChatOpenAI
=======
from langchain_openai import ChatOpenAI
>>>>>>> parent of 7b55bc0 (change to gemini api)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, save_tool, tavily_tool, tavily_extract_tool
from datetime import datetime
import json as _json
import json
import re
import difflib
from collections import Counter

# Import from new modular structure
from config import (
    NOW, ONE_YEAR_AGO_INT, CURRENT_DATETIME_STR,
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    MIN_YEAR_FILTER, FUZZY_MATCH_THRESHOLD, MIN_URLS_PER_EVENT
)
from models import Event, ResearchResponse
from utils.date_utils import (
    extract_year_from_url,
    extract_date_from_tavily_metadata,
    extract_dates_from_search_output,
    is_url_from_2024_onwards,
    filter_urls_by_date
)
from utils.file_utils import (
    get_all_research_files,
    extract_event_names_list,
    extract_event_details,
    get_previous_events
)
from utils.pdf_export import save_research_output_as_pdf
from stage1_knowledge import execute_knowledge_queries

# Cloud storage for Streamlit Cloud deployment
try:
    from cloud_storage import save_to_cloud
except ImportError:
    save_to_cloud = None

# Import API keys from sample.env file
load_dotenv(dotenv_path="sample.env")
 
# Use config values
now = NOW
one_year_ago_int = ONE_YEAR_AGO_INT
current_datetime_str = CURRENT_DATETIME_STR

# -----------------------------
# LLM Setup
# -----------------------------
<<<<<<< HEAD
# CURRENT: Gemini
llm = ChatGoogleGenerativeAI(
=======
llm = ChatOpenAI(
>>>>>>> parent of 7b55bc0 (change to gemini api)
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)

# FALLBACK: For OpenAI, replace above with:
# llm = ChatOpenAI(
#     model=LLM_MODEL,
#     temperature=LLM_TEMPERATURE,
#     max_tokens=LLM_MAX_TOKENS,
# )

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

**Instructions:**
- **Do not summarize, analyze, or generate structured outputs. Only return raw search results and metadata.**
- Focus strictly on information from {one_year_ago} to {current_date_time}. REJECT and IGNORE any articles or data from before {one_year_ago}.
- Perform search using *tavily_tool* to find relevant articles with URLs
- Use *tavily_extract_tool* to verify article publication dates and extract full metadata from URLs
- The tavily_extract_tool will AUTOMATICALLY REJECT articles published before {min_year_filter}
- If an article is rejected by tavily_extract_tool due to old publication date, search for more recent sources
- Always verify article dates using tavily_extract_tool before including them in your knowledge base


"""

stage1_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", stage1_system_prompt_template),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(
    current_date_time=current_datetime_str, # Add this line
    one_year_ago=one_year_ago_int,  # ADD THIS LINE
    min_year_filter=MIN_YEAR_FILTER,  # ADD THIS LINE
    
)


# -----------------------------
# Tools Setup
# -----------------------------
tools = [tavily_tool, tavily_extract_tool, save_tool]

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

# Import knowledge queries from config
from config import KNOWLEDGE_QUERIES

# Get previous events using utility function (already imported from utils.file_utils)
previous_events, event_counter, all_files = get_previous_events(max_files=2)

# Gather knowledge from different time periods
all_knowledge = []
all_sources = []
found_urls = []  # Track URLs
url_metadata_dates = {}  # Track published dates from Tavily metadata

for idx, kq in enumerate(KNOWLEDGE_QUERIES, 1):
    sentiment_icon = {"positive": "✅", "negative": "⚠️", "neutral": "ℹ️", "sentiment": "💭"}.get(kq.get('sentiment', 'neutral'), "ℹ️")
    print(f"📚 [{idx}/{len(KNOWLEDGE_QUERIES)}] {sentiment_icon} {kq['label']}")
    print(f"    Query: {kq['query']}")
    
    try:
        knowledge_response = agent_executor.invoke({"query": kq['query']})
        output = knowledge_response.get("output", "")
        
        if output:
            # Extract URLs from output
            urls_in_output = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', output)
            
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
for idx, kq in enumerate(KNOWLEDGE_QUERIES[:9], 1):
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
**ANALYZE & FORECAST**: Identify 5-8 high-priority, non-obvious (CANNOT BE MAINSTREAM PROBLEMS) emerging issues impacting CPF AND/OR its CPF members (2026-2035).
Prioritize *emerging* issues based on Urgency × Impact × Novelty.
When searching for information on mainstream CPF issues, prioritize and include results from informal channels (e.g., forums, social media, community blogs, public comments) in addition to mainstream news sources. Highlight early warning signals, sentiment, and public concerns from these informal sources.

⚠️ **CRITICAL VALIDATION REQUIREMENT**:
For EACH event, the number of dates MUST EXACTLY EQUAL the number of source URLs.
- If you cite 3 URLs → you MUST provide 3 dates
- If you cite 5 URLs → you MUST provide 5 dates
- Date format: DD/MM/YYYY (e.g., "15/03/2024", NOT "2024")
- Count carefully before submitting!

Order the final output by Urgency × Impact × Novelty score.

### GATHERED KNOWLEDGE:
{combined_knowledge}

### AVAILABLE SOURCE URLS WITH CONTEXT (USE THESE EXACT FULL URLs):
{chr(10).join([f"- {url} (Topic: {url_to_content[url]['topic']})" for url in unique_urls[:50] if url in url_to_content])}

"""


# ------------------------------------------
# STAGE 2: Trend Analysis & Prediction
# -----------------------------
print("" + "="*80)
print("🔮 STAGE 2: TREND ANALYSIS & PREDICTIVE SYNTHESIS")
print("="*80)

prediction_system_prompt = f"""
You are a **strategic foresight analyst** advising senior policymakers on Singapore’s CPF system. 
Your role: monitor and anticipate **CPF-related risks, opportunities, and structural shifts** from **{one_year_ago_int} to {current_datetime_str}**, covering both **established** and **emerging** issues.

---

### 1️⃣ SCOPE & MANDATE
- **Always include established (mainstream) issues** (e.g., cost of living, healthcare, CPF policy changes).
- For each established issue, surface the **latest developments, sentiment shifts, or weak signals** — especially from **informal channels** (Reddit, forums, social media, blogs).
- Cite informal sources directly in the `"source"` field when relevant.
- **Clearly distinguish** between *established* and *emerging* issues in your output.

---

### 2️⃣ PRIORITIZATION RULES
- For mainstream issues: emphasize **informal signals** and **sentiment gaps** beyond official reports.
- Include non-mainstream or weak-signal issues **only** if they show **new urgency or novelty**.
- Highlight how informal perspectives reveal **unaddressed risks or opportunities**.

**Definition:**  
An *emerging issue* is a new, weak-signal, or fast-developing trend with limited but credible evidence, **not yet widely reported**.

**Time Horizon:** STRICTLY **2024–2025** (ignore pre-2024 material).  
**Current Date:** {current_datetime_str}

---

### 3️⃣ OUTPUT FORMAT (CRITICAL)
1. **Return ONLY** the raw JSON object conforming to the `ResearchResponse` schema.  
   - ❌ No markdown, no explanations, no extra text.  
2. Each `"event"` must be a full, evidence-based `Event` object.
3. Fully leverage the Pydantic Field Descriptions (respect min lengths, required fields, and formatting).

---

### 4️⃣ SCENARIO FIELD REQUIREMENTS
- Write a **predictive scenario (200+ words)** containing **four cases**:
  1️⃣ **Base** – Most likely outcome (include probability % range)  
  2️⃣ **Optimistic** – Best realistic outcome (include probability % range)  
  3️⃣ **Pessimistic** – Worst realistic outcome (include probability % range)  
  4️⃣ **Black Swan** – Low-probability, high-impact case (include probability % range)  

Each case must specify:
- **Year (2026–2035)**  
- **Trigger** or key event  
- **Quantified CPF impact**  
- **Affected groups**

**Important:**  
- Tailor all probabilities to the event’s **unique evidence and uncertainty** — *never use default/template values.*  
- Briefly justify each probability (1–2 sentences).

---

### 5️⃣ SOURCE & DATE VALIDATION (STRICT)
1. Each event must have **≥2 full URLs** from different topical domains where possible.  
2. For every URL in `"source"`, provide an exact matching `"date"`.  
   - ✅ 3 URLs → 3 Dates (exact count match required)  
3. Date format: **DD/MM/YYYY** (e.g., `"15/03/2024"`, NOT `"2024"` or `"March 2024"`).  
4. Extract dates from article metadata or URL; if unavailable, default to `"01/01/2024"`.  
5. No bare domains or year-only citations.

---

### 6️⃣ QUALITY & VALIDATION CHECKLIST
- [ ] Not reported in past 2 weeks  
- [ ] ≥2 full URLs (else mark as “Emerging” with low/medium confidence)  
- [ ] URLs ↔ Dates count **exactly match**  
- [ ] Dates formatted **DD/MM/YYYY**  
- [ ] `description` ≥150 words  
- [ ] `impact` ≥150 words, segmented by stakeholders  
- [ ] `scenario` ≥500 chars, 4 cases, citations(for example: According to [URL],)
- [ ] Includes quantitative data (% / $ / counts)  
- [ ] `policy_intervention` includes multiple options with pros/cons & precedents  
- [ ] Includes validation questions and monitoring metrics  

---

### 7️⃣ SPECIAL HANDLING
If only **one credible source** exists:
- Label event as **"Emerging"** or **"Weak Signal"**  
- Set confidence: `"Medium"` or `"Low"` (never `"High"`)  
- Add justification: why only one source  
- Recommend follow-up actions (e.g., `"Monitor forums"`, `"Track additional reports"`)

---

### FINAL OUTPUT
- Return ONLY the raw JSON object (no markdown).  
- Schema compliance is mandatory — failure to match counts, formats, or lengths will invalidate output.

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

# Retry mechanism for LLM parsing failures
MAX_RETRIES = 3
retry_count = 0
structured_response = None
last_error = None

while retry_count < MAX_RETRIES and structured_response is None:
    try:
        if retry_count > 0:
            print(f"\n🔄 Retry attempt {retry_count}/{MAX_RETRIES - 1}")
            print(f"   Previous error: {last_error}")

        # Format the prediction prompt
        formatted_messages = prediction_prompt_template.format_messages(query=prediction_query)

        # DEBUG: Print the full formatted prompt being sent to the LLM (only on first attempt)
        if retry_count == 0:
            print("\n" + "="*80)
            print("📝 DEBUG: FULL FORMATTED PROMPT TO LLM (Stage 2)")
            print("="*80)
            for msg in formatted_messages:
                print(f"[{msg.type.upper()}] {msg.content}\n")
            print("="*80 + "\n")

        # Get prediction from LLM
        prediction_output = llm.invoke(formatted_messages)
        output_text = prediction_output.content

        # DEBUG: Save and print the FULL raw LLM output
        debug_file = f"debug_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"📝 DEBUG: RAW LLM OUTPUT (Stage 2) - Attempt {retry_count + 1}\n")
            f.write("="*80 + "\n")
            f.write(output_text)
            f.write("\n" + "="*80 + "\n")
            f.write(f"✅ Prediction generated: {len(output_text)} characters\n")
            f.write("="*80 + "\n")
        
        print(f"\n💾 Full LLM output saved to: {debug_file}")
        print(f"✅ Prediction generated: {len(output_text)} characters")
        print(f"📝 Preview (first 500 chars):\n{output_text[:500]}...")
        print(f"📝 Preview (last 500 chars):\n...{output_text[-500:]}")

        # Parse the response
        print("" + "="*80)
        print(f"📊 PARSING FINAL PREDICTION - Attempt {retry_count + 1}")
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

        # Repair unterminated strings in JSON
        # Common Gemini issue: unclosed quotes
        import re
        import json
        
        # Fix common JSON issues from Gemini
        json_text = json_text.strip()
        # Fix unterminated strings (quotes not closed before end of line/object)
        json_text = re.sub(r'"([^"]*)(\n|,|})', r'"\1"\2', json_text)
        
        try:
            parsed_json = json.loads(json_text,strict=False)
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON decode error: {e}")
            print(f"   Attempting to fix malformed JSON...")
            # Remove trailing commas
            json_text = re.sub(r'[\x00-\x1F\x7F]', '', json_text)
            json_text = re.sub(r',\s*([\]}])', r'\1', json_text)
            json_text = re.sub(r'(?<!\\)\\n', r'\\\\n', json_text)
            json_text = re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', json_text)
            # Try parsing again
            try:
                parsed_json = json.loads(json_text)
            except json.JSONDecodeError as e2:
                print(f"   ❌ Still invalid JSON: {e2}")
                raise

        print(f"   ✅ JSON parsed successfully")
        print(f"   Top-level keys: {list(parsed_json.keys())}")

        # If wrapped, unwrap it
        if "ResearchResponse" in parsed_json and isinstance(parsed_json, dict):
            print("   ⚠️ Unwrapping nested ResearchResponse")
            json_text = json.dumps(parsed_json["ResearchResponse"])

        structured_response = parser.parse(json_text)
        print(f"✅ Pydantic validation successful on attempt {retry_count + 1}")
        
        # POST-PARSE VALIDATION: Check date-URL matching
        print("\n" + "="*80)
        print("🔍 POST-PARSE VALIDATION: Checking Date-URL Matching")
        print("="*80)
        
        validation_errors = []
        for idx, event in enumerate(structured_response.events, 1):
            event_name = event.event[:60]
            
            # Get URL count
            if isinstance(event.source, list):
                url_count = len(event.source)
            elif isinstance(event.source, str):
                # Handle comma-separated string
                url_count = len([s.strip() for s in event.source.split(',') if s.strip()])
            else:
                url_count = 0
            
            # Get date count
            date_count = len(event.date) if event.date else 0
            
            print(f"Event {idx}: {event_name}")
            print(f"   URLs: {url_count}, Dates: {date_count}")
            
            # Check if counts match
            if url_count != date_count:
                error_msg = f"Event {idx} '{event_name}': URL count ({url_count}) ≠ Date count ({date_count})"
                validation_errors.append(error_msg)
                print(f"   ❌ MISMATCH: {error_msg}")
            else:
                print(f"   ✅ Match: {url_count} URLs = {date_count} dates")
            
            # Check date format
            for i, date_str in enumerate(event.date, 1):
                if '/' in date_str and len(date_str.split('/')) == 3:
                    print(f"   ✅ Date {i}: {date_str} (correct format)")
                else:
                    error_msg = f"Event {idx} '{event_name}': Date {i} '{date_str}' not in DD/MM/YYYY format"
                    validation_errors.append(error_msg)
                    print(f"   ❌ WRONG FORMAT: Date {i}: {date_str}")
        
        # If validation errors found, decide whether to retry or continue
        if validation_errors:
            print("\n" + "="*80)
            print(f"⚠️ FOUND {len(validation_errors)} VALIDATION ERRORS:")
            print("="*80)
            for err in validation_errors:
                print(f"   - {err}")
            
            # If we have retries left, raise an error to trigger retry
            if retry_count < MAX_RETRIES - 1:
                raise ValueError(f"Date-URL validation failed with {len(validation_errors)} errors. Retrying with clearer instructions.")
            else:
                print("\n⚠️ WARNING: Proceeding with validation errors (max retries reached)")
        else:
            print("\n✅ All events passed date-URL validation!")
        
        print("="*80 + "\n")
        
    except Exception as e:
        retry_count += 1
        last_error = str(e)
        print(f"❌ Parsing failed on attempt {retry_count}: {e}")
        
        # Save the problematic output for debugging
        error_file = f"debug_parse_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"❌ PARSING ERROR - Attempt {retry_count}\n")
            f.write("="*80 + "\n")
            f.write(f"Error: {e}\n\n")
            f.write("="*80 + "\n")
            f.write("FULL OUTPUT TEXT:\n")
            f.write("="*80 + "\n")
            f.write(output_text if 'output_text' in locals() else "No output_text available")
            f.write("\n" + "="*80 + "\n")
            f.write("JSON TEXT BEING PARSED:\n")
            f.write("="*80 + "\n")
            f.write(json_text if 'json_text' in locals() else "No json_text available")
            f.write("\n" + "="*80 + "\n")
        print(f"💾 Error details saved to: {error_file}")
        
        if retry_count >= MAX_RETRIES:
            print(f"\n💥 FATAL: Failed to parse after {MAX_RETRIES} attempts")
            print(f"   Last error: {last_error}")
            print(f"   Check {error_file} for full details")
            raise
        else:
            print(f"   Retrying with fresh LLM call...")
            import time
            time.sleep(2)  # Brief pause before retry

# If we get here, parsing succeeded
if __name__ == '__main__':
    try:
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
                    # Get current sources as a set (handle both list and string)
                    if isinstance(event.source, list):
                        current_sources = set(event.source)
                    else:
                        current_sources = set([s.strip() for s in event.source.split(',')]) if event.source else set()
                
                    # Check if any Reddit URL is referenced in the event's text
                    event_text = (event.informal_insights or '') + ' ' + (event.description or '')
                    for rurl in reddit_urls:
                        if rurl in event_text and rurl not in current_sources:
                            current_sources.add(rurl)
                
                    # Update event.source - keep as list
                    event.source = list(current_sources)

        # Only include new events in the main output
        structured_response.events = new_events

        # Add repeated events to a dedicated field for explicit highlighting in the report, with recurrence count and evolution summary
        if repeated_events:
            # Gather evolution history for each repeated event
            evolution_summaries = []
            for e in repeated_events:
                event_name = e.event
                # Collect descriptions and dates from all previous reports (most recent first)
                # Use FUZZY matching to find similar events in previous reports
                desc_history = []
                for f in all_files:
                    details = extract_event_details(f)
                    # Check for exact match first
                    if event_name in details:
                        # Extract date from filename (format: research_output_YYYYMMDD_HHMMSS.json)
                        try:
                            date_str = f.replace('research_output_', '').replace('.json', '').split('_')[0]
                            date_obj = datetime.strptime(date_str, '%Y%m%d')
                            formatted_date = date_obj.strftime('%Y-%m-%d')
                        except:
                            formatted_date = 'Unknown date'
                        desc_history.append((formatted_date, details[event_name]))
                    else:
                        # If no exact match, use fuzzy matching (same threshold as is_repeated_event)
                        for prev_name, prev_desc in details.items():
                            ratio = difflib.SequenceMatcher(None, event_name.lower(), prev_name.lower()).ratio()
                            if ratio >= 0.7:
                                try:
                                    date_str = f.replace('research_output_', '').replace('.json', '').split('_')[0]
                                    date_obj = datetime.strptime(date_str, '%Y%m%d')
                                    formatted_date = date_obj.strftime('%Y-%m-%d')
                                except:
                                    formatted_date = 'Unknown date'
                                desc_history.append((formatted_date, prev_desc))
                                break  # Only take the first fuzzy match per file
                
                # Only keep up to 3 most recent descriptions for brevity
                desc_history = desc_history[:3]
                # Count should be at least 1 if it's flagged as repeated
                occurrence_count = max(1, len(desc_history), event_counter.get(event_name, 0))
                summary = f"{event_name} (seen {occurrence_count} time{'s' if occurrence_count != 1 else ''})\n"
                for i, (date, desc) in enumerate(desc_history, 1):
                    # Show full description instead of truncating
                    desc_clean = desc.replace('\n', ' ')
                    summary += f"  [Prev #{i} - {date}] {desc_clean}\n"
                evolution_summaries.append(summary.strip())
            structured_response.repeated_events = evolution_summaries
        else:
            structured_response.repeated_events = []

        # Validate URLs in sources
        print("" + "="*80)
        print("🔗 VALIDATING SOURCE URLS - 3+ URLs REQUIRED PER EVENT")
        print("="*80)

        validation_failed = False
        total_pre_2024_urls = 0
    
        for idx, event in enumerate(structured_response.events, 1):
            print(f"📌 Event {idx}: {event.event}")
        
            if event.source:
                # Handle both list and string formats
                sources = event.source if isinstance(event.source, list) else [s.strip() for s in event.source.split(',')]
                url_count = len(sources)
                date_count = len(event.date)
            
                # Count URLs that meet criteria
                valid_url_count = 0
                pre_2024_count = 0
            
                for source in sources:
                    # Check if it's a full URL
                    if 'http' in source:
                        path_count = source.count('/')
                    
                        # CHECK DATE: Is this URL from 2024 onwards? (use metadata if available)
                        metadata_date = url_metadata_dates.get(source)
                        if not is_url_from_2024_onwards(source, verbose=False, metadata_date=metadata_date):
                            pre_2024_count += 1
                            total_pre_2024_urls += 1
                            year = metadata_date.year if metadata_date else extract_year_from_url(source)
                            date_source = "metadata" if metadata_date else "URL"
                            print(f"   🚨 PRE-2024 URL DETECTED ({date_source}: {year}): {source[:80]}...")
                            validation_failed = True
                        elif path_count > 3:  # Has article path
                            valid_url_count += 1
                            # Show date source for valid URLs
                            if source in url_metadata_dates:
                                print(f"   ✅ Full URL [metadata: {url_metadata_dates[source].strftime('%Y-%m-%d')}]: {source[:80]}...")
                            else:
                                year = extract_year_from_url(source)
                                date_str = f"URL: {year}" if year else "no date"
                                print(f"   ✅ Full URL [{date_str}]: {source[:80]}...")
                        else:
                            print(f"   ⚠️ Generic domain (no article path): {source}")
                    else:
                        print(f"   ❌ Not a URL: {source}")
            
                # Show pre-2024 warning
                if pre_2024_count > 0:
                    print(f"   🚨🚨🚨 FOUND {pre_2024_count} PRE-2024 URLs - MUST BE REMOVED 🚨🚨🚨")
            
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
    
        # Summary of pre-2024 URLs
        if total_pre_2024_urls > 0:
            print("" + "="*80)
            print(f"🚨 CRITICAL: FOUND {total_pre_2024_urls} PRE-2024 URLs IN FINAL OUTPUT")
            print("="*80)
            print("⚠️ These URLs must be removed and replaced with 2024+ sources")
            print("⚠️ Consider re-running the analysis with stricter date filters")
    
        if validation_failed:
            print("" + "="*80)
            print("❌ OVERALL VALIDATION: FAILED - Some events have insufficient URLs or pre-2024 URLs")
            print("="*80)
        else:
            print("" + "="*80)
            print("✅ OVERALL VALIDATION: PASSED - All events have 3+ URLs from 2024+")
            print("="*80)

        # DETAILED URL USAGE ANALYSIS
        print("" + "="*80)
        print("🔬 DETAILED URL USAGE ANALYSIS")
        print("="*80)

        for idx, event in enumerate(structured_response.events, 1):
            print(f"📌 Event {idx}: {event.event}")
            print(f"   Category: {event.category}")
        
            if event.source:
                # Handle both list and string formats
                sources = event.source if isinstance(event.source, list) else [s.strip() for s in event.source.split(',')]
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
        
        # Update summary to reflect actual event counts after validation and deduplication
        total_new_events = len(structured_response.events)
        total_repeated_events = len(structured_response.repeated_events) if structured_response.repeated_events else 0
        total_issues = total_new_events + total_repeated_events
        
        # Update the summary with accurate counts
        old_summary = structured_response.summary
        # Replace any mention of event counts with accurate numbers
        # Pattern 1: "Identified X emerging issues" 
        # Pattern 2: "X emerging issues identified"
        import re
        summary_updated = re.sub(
            r'[Ii]dentified\s+(\d+)\s+(emerging\s+)?issues?',
            f'Identified {total_issues} emerging issues',
            old_summary
        )
        summary_updated = re.sub(
            r'(\d+)\s+(emerging\s+)?issues?\s+(have\s+been\s+)?identified',
            f'{total_issues} emerging issues have been identified',
            summary_updated,
            flags=re.IGNORECASE
        )
        
        # Add validation note if events were filtered
        if total_new_events < 5:  # If we have fewer than expected
            summary_updated += f" Note: {total_new_events} new events and {total_repeated_events} repeated events passed validation."
        
        structured_response.summary = summary_updated
        
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
        
        # Save to Google Sheets (cloud storage) if configured
        if save_to_cloud:
            try:
                report_id = timestamp
                save_to_cloud(output_json, report_id)
                print(f"☁️ Synced to Google Sheets")
            except Exception as cloud_err:
                print(f"⚠️ Cloud sync failed (local save OK): {cloud_err}")
    
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
        # Update status file on error
        from pathlib import Path
        status_file = Path('research_status.json')
        if status_file.exists():
            try:
                import json as _status_json
                status_data = _status_json.loads(status_file.read_text())
                status_data['status'] = 'error'
                status_data['error'] = f"JSON Decode Error: {str(e)}"
                status_data['end_time'] = datetime.now().isoformat()
                status_file.write_text(_status_json.dumps(status_data))
            except:
                pass
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        print(f"📄 Output text (first 1000 chars):{output_text[:1000]}")
        # Update status file on error
        from pathlib import Path
        status_file = Path('research_status.json')
        if status_file.exists():
            try:
                import json as _status_json
                status_data = _status_json.loads(status_file.read_text())
                status_data['status'] = 'error'
                status_data['error'] = f"{type(e).__name__}: {str(e)}"
                status_data['end_time'] = datetime.now().isoformat()
                status_file.write_text(_status_json.dumps(status_data))
            except:
                pass