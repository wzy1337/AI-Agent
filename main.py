from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, serpapi_tool

load_dotenv(dotenv_path="sample.env")

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

# -----------------------------
# System Prompt
# -----------------------------
system_prompt = """
### Core Directive
Act as a junior analyst providing a briefing to a senior policymaker at Singapore's CPF system. Identify and analyze key emerging issues that could impact CPF policy or public perception in the next 1–2 years.

### Agent Search & Sources
Perform **separate searches** for each task below; do not combine multiple questions in a single tool call. Synthesize results into structured events.

#### A. Public Sentiment & Social Media (Past 3–12 Months)
- Sources: Reddit (r/singapore, r/singaporefi), HardwareZone, X/Twitter, LinkedIn, Facebook Public Groups
- Tasks:
  1. SA Closure / Fund Transfer
  2. Retirement Sum Increase
  3. Gig Economy Contributions (Platform Workers Act)
  4. Emerging non-policy issues affecting CPF decision-making

#### B. Mainstream & Local Media (Past 6 Months)
- Sources: The Straits Times, CNA, TODAY Online, Business Times, CNA Finance
- Tasks:
  1. CPF Special Account and ERS coverage
  2. Platform Workers Act impacts
  3. Policy justifications, challenges, early outcomes

#### C. Official & Government Sources
- Sources: CPF Board circulars, MOF press releases, parliamentary speeches, committee reports, SingStat/MAS releases
- Tasks:
  1. Policy changes in CPF or retirement schemes
  2. Parliamentary discussions or committee reports
  3. Official actuarial reviews or systemic risk alerts

#### D. Regional & International Benchmarking
- Sources: Nikkei Asia, SCMP, OECD, World Bank, IMF, Japan/Korea pension policy reports, think tanks (RSIS, LKYSPP, IDSS, Brookings), consulting reports (McKinsey, PwC, WEF)
- Tasks:
  1. Macro-demographic or geopolitical risks affecting CPF reserves
  2. Pension & retirement policy innovations in comparable nations
  3. Future-of-work studies affecting retirement contributions/income security

#### E. Economic & Systemic Indicators
- Sources: MAS reports, NIRC performance data, sovereign fund reports
- Tasks:
  1. Risks to CPF solvency or net investment returns
  2. Macroeconomic or financial trends impacting CPF funding
  3. Global economic events with indirect CPF implications

### Event Extraction
For each identified issue, extract a structured event with fields:
- event, description, date, actors, location, category, impact, source, relevance
- Merge duplicate events across multiple sources(show a count of merged evemts)
- Only include events from the past 12 months unless historically significant
- Wrap final output in JSON following the ResearchResponse schema

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
tools = [search_tool, wiki_tool, save_tool]
if serpapi_tool is not None:
    tools.append(serpapi_tool)

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
