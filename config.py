# -*- coding: utf-8 -*-
"""
Configuration settings for the CPF Research Intelligence system.
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta

# ============================================
# Time Configuration
# ============================================
NOW = datetime.now()
ONE_YEAR_AGO_DATETIME = NOW - relativedelta(years=1)
ONE_YEAR_AGO_INT = ONE_YEAR_AGO_DATETIME.year
CURRENT_DATETIME_STR = NOW.strftime("%Y-%m-%d, %A. Time: %H:%M:%S. Current timezone is UTC+8.")

# ============================================
# LLM Configuration
# ============================================
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0  # Deterministic output
LLM_MAX_TOKENS = 16000  # Increased to allow 8-10 complete events (~2000 tokens each)
MAX_RETRIES = 3  # Retry attempts for parsing failures

# ============================================
# Research Configuration
# ============================================
MIN_YEAR_FILTER = NOW.year -1  # Dynamically uses current year (2025)
MAX_PAST_REPORTS = 2  # Number of past reports to compare for deduplication
FUZZY_MATCH_THRESHOLD = 0.7  # Similarity threshold for repeated event detection
MIN_URLS_PER_EVENT = 3  # Minimum number of source URLs required per event

# ============================================
# Knowledge Queries - Tier 1: Global Macro Trends
# ============================================
KNOWLEDGE_QUERIES_TIER= [
    {"query": "What are the biggest global economic, demographic, or geopolitical risks that could impact retirement systems and pension funds worldwide 2024-2025?", "label": "Global Macro: Systemic Risks to Retirement", "sentiment": "horizon"},
    {"query": "What are international organizations (IMF, World Bank, OECD, BIS) warning about regarding pension sustainability and retirement adequacy 2024-2025?", "label": "Global Macro: International Warnings", "sentiment": "horizon"},
    {"query": "What are the most significant pension crises, reforms, or failures happening globally 2024-2025? Include Europe, Asia, Americas, and emerging markets.", "label": "Global Macro: Pension Crises Worldwide", "sentiment": "horizon"},
    
    # ============================================
    # TIER 2: CROSS-BORDER TRENDS & PRECEDENTS
    # ============================================
    {"query": "What innovative or experimental pension reforms are being tested in Nordic countries, UK, Australia, Canada, Japan 2024-2025?", "label": "International: Advanced Economy Experiments", "sentiment": "horizon"},
    {"query": "What retirement and social security challenges are Asian countries (Japan, South Korea, Taiwan, Hong Kong, Malaysia) facing 2024-2025? Regional comparisons.", "label": "International: Asian Retirement Challenges", "sentiment": "horizon"},
    {"query": "What lessons from international pension failures or controversies could apply to Singapore 2024-2025? Include UK, US, European cases.", "label": "International: Cautionary Tales & Failures", "sentiment": "horizon"},
    {"query": "What are global think tanks and research institutions publishing about future-of-retirement and pension sustainability 2024-2025? Include Brookings, CSIS, Peterson Institute.", "label": "International: Think Tank Research", "sentiment": "horizon"},
    
    # ============================================
    # TIER 3: TECHNOLOGY & DISRUPTION
    # ============================================
    {"query": "How are AI, automation, and gig economy disrupting traditional employment and retirement savings globally 2024-2025? Future of work implications.", "label": "Tech Disruption: AI & Future of Work", "sentiment": "horizon"},
    {"query": "What are fintech, crypto, and web3 innovations in retirement planning and pension management 2024-2025? Include DeFi, tokenization, digital assets.", "label": "Tech Disruption: Fintech & Web3 Pensions", "sentiment": "horizon"},
    {"query": "What are the cybersecurity risks, data breaches, or tech failures affecting pension funds and retirement systems 2024-2025?", "label": "Tech Disruption: Cyber Risks to Pensions", "sentiment": "horizon"},
    {"query": "How are longevity breakthroughs, healthtech, and aging science changing retirement planning assumptions 2024-2025? Impact of living to 100+.", "label": "Tech Disruption: Longevity & Healthtech", "sentiment": "horizon"},
    
    # ============================================
    # TIER 4: WEAK SIGNALS & FRINGE SOURCES
    # ============================================
    {"query": "What are the most surprising, unconventional, or contrarian views on retirement and pensions from blogs, podcasts, and alternative media 2024-2025?", "label": "Weak Signals: Alternative Media & Contrarians", "sentiment": "horizon"},
    {"query": "What are early warning signals, emerging risks, or 'canary in the coal mine' indicators for retirement systems from forums, Reddit, Twitter/X 2024-2025?", "label": "Weak Signals: Social Media Early Warnings", "sentiment": "horizon"},
    {"query": "What speculative scenarios, black swan events, or 'what if' analyses exist for pension and retirement systems 2024-2025? Include scenario planning.", "label": "Weak Signals: Black Swan Scenarios", "sentiment": "horizon"},
    {"query": "What are fringe communities, subcultures, or movements saying about retirement (FIRE movement, anti-work, digital nomads) 2024-2025?", "label": "Weak Signals: Fringe Movements & Subcultures", "sentiment": "horizon"},
    
    # ============================================
    # TIER 5: INTERDISCIPLINARY & ADJACENT DOMAINS
    # ============================================
    {"query": "How are climate change, environmental risks, and ESG factors affecting pension fund strategies and retirement security 2024-2025?", "label": "Adjacent: Climate & ESG Impact", "sentiment": "horizon"},
    {"query": "What are behavioral economics and psychology insights on retirement savings behavior and pension engagement 2024-2025? Nudge theory applications.", "label": "Adjacent: Behavioral Economics", "sentiment": "horizon"},
    {"query": "How are housing affordability crisis, real estate bubbles, and homeownership affecting retirement adequacy globally 2024-2025?", "label": "Adjacent: Housing & Retirement", "sentiment": "horizon"},
    {"query": "What are healthcare cost inflation, long-term care crises, and medical bankruptcy implications for retirement planning 2024-2025?", "label": "Adjacent: Healthcare Costs & Retirement", "sentiment": "horizon"},
    
    # ============================================
    # TIER 6: SINGAPORE-SPECIFIC (Enhanced Scope)
    # ============================================
    {"query": "What are the most surprising or under-discussed CPF and retirement issues in Singapore 2024-2025? Include forums, social media, Reddit r/singapore.", "label": "Singapore: Non-Obvious Issues & Ground Sensing", "sentiment": "horizon"},
    {"query": "What are Singapore policymakers, ministers, and MPs saying about CPF reforms and retirement challenges 2024-2025? Parliamentary debates.", "label": "Singapore: Policy Signals & Debates", "sentiment": "horizon"},
    {"query": "What are Singaporean researchers, universities, and think tanks (LKYSPP, IPS, RSIS) publishing on CPF and retirement 2024-2025?", "label": "Singapore: Academic & Research", "sentiment": "horizon"},
    {"query": "How do Singapore's retirement challenges compare to regional neighbors and advanced economies 2024-2025? Benchmarking and gap analysis.", "label": "Singapore: Comparative Analysis", "sentiment": "horizon"},
    
    # ============================================
    # TIER 7: EXPERT OPINIONS & THOUGHT LEADERSHIP
    # ============================================
    {"query": "What are leading economists, pension experts, and thought leaders predicting about retirement systems 2024-2025? Include Nobel laureates, IMF economists.", "label": "Expert Opinions: Leading Economists", "sentiment": "horizon"},
    {"query": "What are investment managers, asset allocators, and sovereign wealth funds saying about pension fund strategies 2024-2025? BlackRock, Vanguard, GIC insights.", "label": "Expert Opinions: Investment Perspectives", "sentiment": "horizon"},
    {"query": "What are demographic experts and population researchers warning about aging societies and pension sustainability 2024-2025?", "label": "Expert Opinions: Demographics & Aging", "sentiment": "horizon"},
]

# Combine all knowledge queries
KNOWLEDGE_QUERIES = KNOWLEDGE_QUERIES_TIER

# ============================================
# Sentiment Icons
# ============================================
SENTIMENT_ICONS = {
    "positive": "✅",
    "negative": "⚠️",
    "neutral": "ℹ️",
    "sentiment": "💭",
    "horizon": "🔭"
}
