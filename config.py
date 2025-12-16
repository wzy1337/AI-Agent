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
LLM_MODEL = "gemini-1.5-flash"  # Using Gemini (OpenAI quota exhausted)
LLM_TEMPERATURE = 0  # Deterministic output
LLM_MAX_TOKENS = 4000  # Maximum output tokens for Gemini
MAX_RETRIES = 3  # Retry attempts for parsing failures

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
    {"query": f"Emerging concerns from IMF World Bank OECD BIS about pension sustainability retirement adequacy {YEAR_RANGE}", "label": "Global Macro: International Warnings", "sentiment": "horizon"},
    
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
