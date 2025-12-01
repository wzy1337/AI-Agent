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
MIN_YEAR_FILTER = NOW.year -1  # Dynamically uses current year (2025)
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

]

# ============================================
# Knowledge Queries - Tier 3: Weak Signals
# ============================================
KNOWLEDGE_QUERIES_TIER3 = [
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
