# -*- coding: utf-8 -*-
"""
Utility modules for CPF Research Intelligence system.
"""

from .date_utils import (
    extract_year_from_url,
    extract_date_from_tavily_metadata,
    extract_dates_from_search_output,
    is_url_from_2024_onwards,
    filter_urls_by_date
)

from .file_utils import (
    get_all_research_files,
    extract_event_names_list,
    extract_event_details,
    get_previous_events
)

from .pdf_export import save_research_output_as_pdf

__all__ = [
    # Date utilities
    'extract_year_from_url',
    'extract_date_from_tavily_metadata',
    'extract_dates_from_search_output',
    'is_url_from_2024_onwards',
    'filter_urls_by_date',
    
    # File utilities
    'get_all_research_files',
    'extract_event_names_list',
    'extract_event_details',
    'get_previous_events',
    
    # PDF export
    'save_research_output_as_pdf',
]
