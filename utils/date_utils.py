# -*- coding: utf-8 -*-
"""
Date extraction and URL filtering utilities.
"""

import re
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from dateutil import parser as date_parser


def extract_year_from_url(url: str) -> Optional[int]:
    """
    Extract publication year from URL patterns.
    Returns year as int if found, None otherwise.
    
    Common patterns:
    - /2024/01/article
    - /article-2024-01-15
    - ?date=2024-01-15
    """
    # Pattern 1: /YYYY/ in path
    year_match = re.search(r'/(\d{4})/', url)
    if year_match:
        return int(year_match.group(1))
    
    # Pattern 2: YYYY-MM-DD anywhere
    date_match = re.search(r'(\d{4})-\d{2}-\d{2}', url)
    if date_match:
        return int(date_match.group(1))
    
    # Pattern 3: /YYYYMMDD/ format
    compact_date = re.search(r'/(\d{4})\d{4}/', url)
    if compact_date:
        return int(compact_date.group(1))
    
    # Pattern 4: Query parameters (e.g., ?date=2024-01-15)
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        for key in ['date', 'published', 'publishDate', 'pubdate']:
            if key in params:
                date_str = params[key][0]
                date_match = re.search(r'(\d{4})', date_str)
                if date_match:
                    return int(date_match.group(1))
    except:
        pass
    
    return None


def extract_date_from_tavily_metadata(output: str, url: str) -> Optional[datetime]:
    """
    Extract published date from Tavily search result metadata.
    Looks for the 'Published: YYYY-MM-DD' line after the URL.
    """
    try:
        # Find the section containing this URL
        url_escaped = re.escape(url)
        pattern = rf'🔗 COMPLETE URL: {url_escaped}\s*\n📅 Published: ([^\n]+)'
        match = re.search(pattern, output)
        
        if match:
            date_str = match.group(1).strip()
            if date_str and date_str != 'Date not available':
                try:
                    return date_parser.parse(date_str)
                except:
                    pass
    except:
        pass
    
    return None


def extract_dates_from_search_output(output: str) -> Dict[str, datetime]:
    """
    Extract all URLs and their published dates from Tavily search output.
    Returns dict mapping URL -> datetime object.
    """
    url_dates = {}
    
    # Pattern to match URL and its published date
    pattern = r'🔗 COMPLETE URL: ([^\n]+)\s*\n📅 Published: ([^\n]+)'
    matches = re.findall(pattern, output)
    
    for url, date_str in matches:
        url = url.strip()
        date_str = date_str.strip()
        
        if date_str and date_str != 'Date not available':
            try:
                parsed_date = date_parser.parse(date_str)
                url_dates[url] = parsed_date
            except:
                pass
    
    return url_dates


def is_url_from_2024_onwards(url: str, verbose: bool = False, metadata_date: Optional[datetime] = None) -> bool:
    """
    Check if URL is from 2024 or later.
    
    Priority:
    1. Use metadata_date if provided (from Tavily)
    2. Extract year from URL pattern
    3. Default to True (benefit of doubt)
    
    Returns True if year >= 2024 or year cannot be determined.
    Returns False if year < 2024.
    """
    # Priority 1: Use metadata date if available
    if metadata_date:
        year = metadata_date.year
        if verbose:
            date_source = "metadata"
            if year >= 2024:
                print(f"   ✅ Year {year} from {date_source} (valid): {url[:60]}...")
            else:
                print(f"   ❌ Year {year} from {date_source} (DISCARDED): {url[:60]}...")
        return year >= 2024
    
    # Priority 2: Extract from URL
    year = extract_year_from_url(url)
    
    if year is None:
        if verbose:
            print(f"   ⚠️ No date found (allowing): {url[:60]}...")
        return True  # Cannot determine, allow it
    
    if year >= 2024:
        if verbose:
            print(f"   ✅ Year {year} from URL (valid): {url[:60]}...")
        return True
    else:
        if verbose:
            print(f"   ❌ Year {year} from URL (DISCARDED): {url[:60]}...")
        return False


def filter_urls_by_date(urls: List[str], min_year: int = 2024, verbose: bool = False, 
                        url_metadata: Optional[Dict[str, datetime]] = None) -> Tuple[List[str], Dict[str, str]]:
    """
    Filter list of URLs to only include those from min_year onwards.
    URLs without detectable dates are included (benefit of doubt).
    
    Args:
        urls: List of URLs to filter
        min_year: Minimum year to keep (default: 2024)
        verbose: Print filtering details
        url_metadata: Optional dict mapping URL -> datetime from search metadata
    
    Returns:
        Tuple of (filtered_urls, date_info) where date_info maps URL -> date source
    """
    filtered = []
    discarded = []
    date_info = {}  # Track where date came from
    
    for url in urls:
        # Check metadata first
        metadata_date = url_metadata.get(url) if url_metadata else None
        
        if is_url_from_2024_onwards(url, verbose=verbose, metadata_date=metadata_date):
            filtered.append(url)
            
            # Track date source
            if metadata_date:
                date_info[url] = f"metadata:{metadata_date.strftime('%Y-%m-%d')}"
            else:
                year = extract_year_from_url(url)
                if year:
                    date_info[url] = f"url:{year}"
                else:
                    date_info[url] = "unknown"
        else:
            discarded.append(url)
    
    if verbose and discarded:
        print(f"\n🗑️ DISCARDED {len(discarded)} URLs from before {min_year}:")
        for url in discarded[:5]:  # Show first 5
            print(f"   - {url}")
        if len(discarded) > 5:
            print(f"   ... and {len(discarded) - 5} more")
    
    return filtered, date_info
