# -*- coding: utf-8 -*-
"""
File management and research report utilities.
"""

import json
import glob
from typing import Dict, List, Set
from collections import Counter


def get_all_research_files(max_n: int = 20) -> List[str]:
    """Get all research output JSON files, sorted newest first."""
    files = glob.glob("research_output_2025*.json")
    files += glob.glob("research_output_2024*.json")
    files = sorted(files, reverse=True)
    return files[:max_n]


def extract_event_names_list(filepath: str) -> List[str]:
    """Extract list of event names from a research output file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Support both 'events' at top-level or inside 'ResearchResponse'
        events = data.get('events')
        if events is None and 'ResearchResponse' in data:
            events = data['ResearchResponse'].get('events', [])
        if not events:
            return []
        return [e.get('event') for e in events if 'event' in e]
    except Exception as e:
        print(f"[PAST REPORTS] Error reading {filepath}: {e}")
        return []


def extract_event_details(filepath: str) -> Dict[str, str]:
    """Return a dict of event name to description for a given report file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        events = data.get('events')
        if events is None and 'ResearchResponse' in data:
            events = data['ResearchResponse'].get('events', [])
        if not events:
            return {}
        return {e.get('event'): e.get('description', '') for e in events if 'event' in e}
    except Exception as e:
        print(f"[PAST REPORTS] Error reading {filepath}: {e}")
        return {}


def get_previous_events(max_files: int = 2) -> tuple[Set[str], Counter, List[str]]:
    """
    Get previous events for deduplication.
    
    Returns:
        Tuple of (previous_event_names, event_counter, all_files)
    """
    recent_files = get_all_research_files(max_n=max_files)
    previous_events = set()
    for f in recent_files:
        previous_events |= set(extract_event_names_list(f))
    
    # Track event frequency for repeated event summaries
    all_files = get_all_research_files(max_n=max_files)
    all_event_names = []
    for f in all_files:
        all_event_names.extend(extract_event_names_list(f))
    event_counter = Counter(all_event_names)
    
    return previous_events, event_counter, all_files
