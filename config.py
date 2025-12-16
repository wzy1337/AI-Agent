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
# CURRENT: Gemini 2.5 Flash (free tier)
LLM_MODEL = "gemini-2.5-flash-lite"
LLM_TEMPERATURE = 0  # Deterministic output
LLM_MAX_TOKENS = 3000  # Maximum output tokens for Gemini

# FALLBACK: To use OpenAI instead, uncomment below and comment out Gemini above:
# Ensure you have credits at https://platform.openai.com/account/billing
# LLM_MODEL = "gpt-4o-mini"
# LLM_MAX_TOKENS = 16384

MAX_RETRIES = 3  # Retry attempts for parsing failures

# ============================================
# Rate Limiting (API Protection)
# ============================================
API_CALLS_PER_MINUTE = 10  # Max LLM calls per minute (conservative limit)
API_DELAY_SECONDS = 6  # Delay between API calls (60s / 10 calls = 6s)

# ============================================
# Research Configuration
# ============================================
MIN_YEAR_FILTER = NOW.year -1  # Dynamically uses current year
MAX_PAST_REPORTS = 2  # Number of past reports to compare for deduplication
FUZZY_MATCH_THRESHOLD = 0.7  # Similarity threshold for repeated event detection
MIN_URLS_PER_EVENT = 1  # Minimum number of source URLs required per event

# Dynamic year range for queries (e.g., "2024-2025" when current year is 2025)
YEAR_RANGE = f"{NOW.year - 1}-{NOW.year}"
CURRENT_YEAR = str(NOW.year)
LAST_YEAR = str(NOW.year - 1)

# ============================================
# Knowledge Queries - Tier 1: Global Macro Trends
# ============================================
KNOWLEDGE_QUERIES_TIER = [
    # ============================================
    # TIER 1: GLOBAL MACRO TRENDS & SYSTEMIC RISKS
    # ============================================
    {"query": f"Unexpected developments in global pension systems retirement funds {YEAR_RANGE}", "label": "Global Macro: Unexpected Developments", "sentiment": "horizon"},
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
