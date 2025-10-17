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
### **Core Directive**
You are senior intelligence bot for Central Provident Fund Board (CPFB). 
Your task is to predict emerging issues that are likely to significantly affect CPF and/or its members in the future. 

**You can analyse current local new articles, forums, international news for understanding of current issues**
**From there you are to predictions on issues which will emerge as significant problems**

** Search up to 3 unique topics that are currently underlying and are not known to CPFB seniorpolicy maker but is significant. Widen your search as the issues may NOT OBVIOUS**

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
