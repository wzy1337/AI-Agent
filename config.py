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
LLM_MAX_TOKENS = 16384  # Maximum output tokens for gpt-4o
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
    {"query": f"Pension crises reforms failures backlash worldwide Europe Asia Americas {YEAR_RANGE}", "label": "Global Macro: Pension Crises Worldwide", "sentiment": "horizon"},
    {"query": f"Global economic demographic geopolitical risks impacting retirement systems {YEAR_RANGE}", "label": "Global Macro: Systemic Risks", "sentiment": "horizon"},
    
    # ============================================
    # TIER 2: CROSS-BORDER TRENDS & PRECEDENTS
    # ============================================
    {"query": f"Innovative experimental pension reforms Nordic countries Netherlands UK Australia Canada {YEAR_RANGE}", "label": "International: Advanced Economy Experiments", "sentiment": "horizon"},
    {"query": f"Japan South Korea Taiwan Hong Kong Malaysia retirement social security challenges aging {YEAR_RANGE}", "label": "International: Asian Retirement Challenges", "sentiment": "horizon"},
    {"query": f"Pension failures controversies scandals lessons UK US Europe {YEAR_RANGE}", "label": "International: Cautionary Tales & Failures", "sentiment": "horizon"},
    {"query": f"Think tank research future of retirement pension sustainability Brookings CSIS Peterson Institute {YEAR_RANGE}", "label": "International: Think Tank Research", "sentiment": "horizon"},
    {"query": f"Latin America pension reforms changes Chile Argentina Brazil {YEAR_RANGE}", "label": "International: Latin America Experiments", "sentiment": "horizon"},
    
    # ============================================
    # TIER 3: TECHNOLOGY & DISRUPTION
    # ============================================
    {"query": f"Fintech crypto DeFi tokenization innovations retirement planning pension management {YEAR_RANGE}", "label": "Tech Disruption: Fintech & Web3 Pensions", "sentiment": "horizon"},
    {"query": f"Cybersecurity risks data breaches pension funds retirement systems {YEAR_RANGE}", "label": "Tech Disruption: Cyber Risks to Pensions", "sentiment": "horizon"},
    {"query": f"Longevity breakthroughs aging science living to 100 retirement planning implications {YEAR_RANGE}", "label": "Tech Disruption: Longevity & Healthtech", "sentiment": "horizon"},
    
    
    # ============================================
    # TIER 5: ANOMALY & SURPRISE DETECTION
    # ============================================
    {"query": f"Surprising pension fund performance outliers controversies unexpected {YEAR_RANGE}", "label": "Anomalies: Unexpected Developments", "sentiment": "horizon"},
    {"query": f"Pension retirement policy reversals U-turns abandoned experiments {YEAR_RANGE}", "label": "Anomalies: Policy Reversals", "sentiment": "horizon"},
    {"query": f"Emerging retirement risks nobody is talking about underreported {YEAR_RANGE}", "label": "Anomalies: Underreported Risks", "sentiment": "horizon"},
    {"query": f"Pension fund failures lawsuits scandals mismanagement {YEAR_RANGE}", "label": "Anomalies: Failures & Scandals", "sentiment": "horizon"},
    
    # ============================================
    # TIER 6: INTERDISCIPLINARY & ADJACENT DOMAINS
    # ============================================
    {"query": f"Climate change environmental risks ESG pension fund strategies retirement security {YEAR_RANGE}", "label": "Adjacent: Climate & ESG Impact", "sentiment": "horizon"},
    {"query": f"Behavioral economics psychology retirement savings nudge theory pension engagement {YEAR_RANGE}", "label": "Adjacent: Behavioral Economics", "sentiment": "horizon"},
    {"query": f"Housing affordability crisis real estate retirement adequacy homeownership {YEAR_RANGE}", "label": "Adjacent: Housing & Retirement", "sentiment": "horizon"},
    {"query": f"Healthcare cost inflation long-term care crisis medical bankruptcy retirement {YEAR_RANGE}", "label": "Adjacent: Healthcare Costs & Retirement", "sentiment": "horizon"},
    {"query": f"Insurance industry disruption affecting retirement annuities {YEAR_RANGE}", "label": "Adjacent: Insurance Disruption", "sentiment": "horizon"},
    {"query": f"Banking sector changes impacting retirement savings products {YEAR_RANGE}", "label": "Adjacent: Banking Evolution", "sentiment": "horizon"},
    
    # ============================================
    # TIER 7: NARRATIVE SHIFTS & PUBLIC DISCOURSE
    # ============================================
    {"query": f"Changing attitudes toward retirement age working longer public opinion {YEAR_RANGE}", "label": "Narratives: Retirement Age Debate", "sentiment": "horizon"},
    {"query": f"Intergenerational fairness pension inequality young workers older generations {YEAR_RANGE}", "label": "Narratives: Generational Conflict", "sentiment": "horizon"},
    {"query": f"Retirement is dead changing concept of retirement new models {YEAR_RANGE}", "label": "Narratives: Retirement Redefined", "sentiment": "horizon"},
    {"query": f"Public pension underfunding default risk warnings {YEAR_RANGE}", "label": "Narratives: Funding Crisis Warnings", "sentiment": "horizon"},
    

    # ============================================
    # TIER 9: EXPERT OPINIONS & THOUGHT LEADERSHIP
    # ============================================
    {"query": f"Economists pension experts predictions retirement systems Nobel laureates IMF {YEAR_RANGE}", "label": "Expert Opinions: Leading Economists", "sentiment": "horizon"},
    {"query": f"Investment managers sovereign wealth funds pension strategies BlackRock Vanguard GIC {YEAR_RANGE}", "label": "Expert Opinions: Investment Perspectives", "sentiment": "horizon"},
    {"query": f"Demographic experts population researchers aging societies pension sustainability warnings {YEAR_RANGE}", "label": "Expert Opinions: Demographics & Aging", "sentiment": "horizon"},
    {"query": f"Retirement savings shortfall crisis inadequacy warnings experts {YEAR_RANGE}", "label": "Expert Opinions: Adequacy Warnings", "sentiment": "horizon"},
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
