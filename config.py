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
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0  # Deterministic output
LLM_MAX_TOKENS = 10000
MAX_RETRIES = 3  # Retry attempts for parsing failures

# ============================================
# Research Configuration
# ============================================
MIN_YEAR_FILTER = 2024  # Only include sources from this year onwards
MAX_PAST_REPORTS = 2  # Number of past reports to compare for deduplication
FUZZY_MATCH_THRESHOLD = 0.7  # Similarity threshold for repeated event detection
MIN_URLS_PER_EVENT = 3  # Minimum number of source URLs required per event

# ============================================
# Knowledge Queries - Tier 1: Global Macro Trends
# ============================================
KNOWLEDGE_QUERIES_TIER1 = [
    {
        "query": "What are the biggest global economic, demographic, or geopolitical risks that could impact retirement systems and pension funds worldwide 2024-2025?",
        "label": "Global Macro: Systemic Risks to Retirement",
        "sentiment": "horizon"
    },
    {
        "query": "What are international organizations (IMF, World Bank, OECD, BIS) warning about regarding pension sustainability and retirement adequacy 2024-2025?",
        "label": "Global Macro: International Warnings",
        "sentiment": "horizon"
    },
    {
        "query": "What are the most significant pension crises, reforms, or failures happening globally 2024-2025? Include Europe, Asia, Americas, and emerging markets.",
        "label": "Global Macro: Pension Crises Worldwide",
        "sentiment": "horizon"
    },
]

# ============================================
# Knowledge Queries - Tier 2: Cross-Border Trends
# ============================================
KNOWLEDGE_QUERIES_TIER2 = [
    {
        "query": "What innovative or experimental pension reforms are being tested in Nordic countries, UK, Australia, Canada, Japan 2024-2025?",
        "label": "International: Advanced Economy Experiments",
        "sentiment": "horizon"
    },
    {
        "query": "What retirement and social security challenges are Asian countries (Japan, South Korea, Taiwan, Hong Kong, Malaysia) facing 2024-2025? Regional comparisons.",
        "label": "International: Asian Retirement Challenges",
        "sentiment": "horizon"
    },
    {
        "query": "What lessons from international pension failures or controversies could apply to Singapore 2024-2025? Include UK, US, European cases.",
        "label": "International: Cautionary Tales & Failures",
        "sentiment": "horizon"
    },
    {
        "query": "What are global think tanks and research institutions publishing about future-of-retirement and pension sustainability 2024-2025? Include Brookings, CSIS, Peterson Institute.",
        "label": "International: Think Tank Research",
        "sentiment": "horizon"
    },
]

# ============================================
# Knowledge Queries - Tier 3: Weak Signals
# ============================================
KNOWLEDGE_QUERIES_TIER3 = [
    {
        "query": "What are the most surprising, unconventional, or contrarian views on retirement and pensions from blogs, podcasts, and alternative media 2024-2025?",
        "label": "Weak Signals: Alternative Media & Contrarians",
        "sentiment": "horizon"
    },
    {
        "query": "What are early warning signals, emerging risks, or 'canary in the coal mine' indicators for retirement systems from forums, Reddit, Twitter/X 2024-2025?",
        "label": "Weak Signals: Social Media Early Warnings",
        "sentiment": "horizon"
    },
    {
        "query": "What speculative scenarios, black swan events, or 'what if' analyses exist for pension and retirement systems 2024-2025? Include scenario planning.",
        "label": "Weak Signals: Black Swan Scenarios",
        "sentiment": "horizon"
    },
    {
        "query": "What are fringe communities, subcultures, or movements saying about retirement (FIRE movement, anti-work, digital nomads) 2024-2025?",
        "label": "Weak Signals: Fringe Movements & Subcultures",
        "sentiment": "horizon"
    },
]

# Combine all knowledge queries
KNOWLEDGE_QUERIES = KNOWLEDGE_QUERIES_TIER1 + KNOWLEDGE_QUERIES_TIER2 + KNOWLEDGE_QUERIES_TIER3

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
