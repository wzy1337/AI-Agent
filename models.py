# -*- coding: utf-8 -*-
"""
Pydantic data models for research responses.
"""

# CURRENT: Native Pydantic v2
from pydantic import BaseModel, Field
# FALLBACK: For older LangChain (0.2.16), use:
# from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional


class Event(BaseModel):
    """Individual research event with full analysis."""
    
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

    source: List[str] = Field(
        description="List all full article URLs from *different* sources as an array. Example: ['https://a.com/x', 'https://b.com/y', 'https://c.com/z']."
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
    
    signal_strength: str = Field(
        description="Tag as 'Established', 'Emerging', or 'Weak Signal' with a brief rationale."
    )
    
    informal_insights: Optional[str] = Field(
        default=None,
        description="For established/mainstream events, summarize the latest new developments, sentiment, or weak signals from informal channels (e.g., forums, social media, community blogs). Only populate if signal_strength is 'Established'."
    )


class ResearchResponse(BaseModel):
    """Complete research response containing multiple events and analysis."""
    
    topic: str
    summary: str = Field(
        description="EXECUTIVE SUMMARY for policymakers (150-200 words):"
                    "Brief overview of findings designed for senior decision-makers."
                    "Format: [X] emerging issues identified, prioritized by [criteria]. "
                    "Most urgent: [issue], requiring attention by [timeframe]. "
                    "Key uncertainties: [what we don't know]. "
                    "Recommended next actions: [immediate steps for validation/planning]."
    )
    source: List[str] = Field(
        description="🚨 CRITICAL: Cite ALL FULL URLs used as evidence. Source QUALITY > quantity. A single government report is better than 3 opinion blogs. Justify the quality of sources in the 'confidence' field."
    )
    tools_used: List[str]
    events: List[Event]
    action_items: List[str] = Field(
        default_factory=list,
        description="Optional: 3-5 immediate next steps for policymakers (e.g., 'Request MOM data on caregiving workforce exits', 'Consult with eldercare sector on cost projections')"
    )
    repeated_events: Optional[List[str]] = Field(
        default=None,
        description="List of event names that are repeated from previous reports for explicit highlighting."
    )
