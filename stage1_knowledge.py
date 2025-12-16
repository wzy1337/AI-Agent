# -*- coding: utf-8 -*-
"""
Stage 1: Knowledge building through research queries.
"""

import re
import time
from collections import Counter
from typing import List, Dict, Tuple
from langchain_classic.agents import AgentExecutor

from config import KNOWLEDGE_QUERIES, SENTIMENT_ICONS, MIN_YEAR_FILTER, API_DELAY_SECONDS
from utils.date_utils import extract_dates_from_search_output, filter_urls_by_date


def execute_knowledge_queries(agent_executor: AgentExecutor) -> Tuple[str, List[str], Dict[str, any]]:
    """
    Execute all knowledge queries and collect information.
    
    Returns:
        Tuple of (combined_knowledge, unique_urls, url_stats)
    """
    all_knowledge = []
    all_sources = []
    found_urls = []
    url_metadata_dates = {}
    
    print("=" * 80)
    print("🔍 STAGE 1: BUILDING KNOWLEDGE BASE")
    print("=" * 80)
    
    for idx, kq in enumerate(KNOWLEDGE_QUERIES, 1):
        sentiment_icon = SENTIMENT_ICONS.get(kq.get('sentiment', 'neutral'), "ℹ️")
        print(f"📚 [{idx}/{len(KNOWLEDGE_QUERIES)}] {sentiment_icon} {kq['label']}")
        print(f"    Query: {kq['query']}")
        
        # Rate limiting: Add delay between API calls
        if idx > 1:  # Skip delay for first query
            print(f"⏳ Rate limit: waiting {API_DELAY_SECONDS}s before next API call...")
            time.sleep(API_DELAY_SECONDS)
        
        try:
            knowledge_response = agent_executor.invoke({"query": kq['query']})
            output = knowledge_response.get("output", "")
            
            if output:
                # Extract URLs from output
                urls_in_output = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', output)
                
                # Extract published dates from Tavily metadata
                output_metadata = extract_dates_from_search_output(output)
                url_metadata_dates.update(output_metadata)
                
                # Filter URLs by date
                urls_before_filter = len(urls_in_output)
                filtered_urls, date_info = filter_urls_by_date(
                    urls_in_output,
                    min_year=MIN_YEAR_FILTER,
                    verbose=False,
                    url_metadata=output_metadata
                )
                urls_after_filter = len(filtered_urls)
                
                found_urls.extend(filtered_urls)
                
                # Add sentiment label to knowledge
                sentiment_label = kq.get('sentiment', 'neutral').upper()
                all_knowledge.append(f"## {kq['label']} [SENTIMENT: {sentiment_label}]{output}")
                all_sources.append(output)
                
                # Show filtering results
                filtered_count = urls_before_filter - urls_after_filter
                metadata_count = sum(1 for url in filtered_urls if url in output_metadata)
                
                if filtered_count > 0:
                    print(f"✅ Collected {len(output)} characters, {urls_after_filter} URLs (🗑️ filtered {filtered_count} pre-2024, 📅 {metadata_count} with metadata)")
                else:
                    print(f"✅ Collected {len(output)} characters, {urls_after_filter} URLs (📅 {metadata_count} with metadata)")
                
                # Show found URLs
                if filtered_urls:
                    for url in filtered_urls[:2]:  # Show first 2
                        date_source = date_info.get(url, 'unknown')
                        print(f"       📎 {url} [{date_source}]")
            else:
                print(f"⚠️ No output received")
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Process URL statistics
    url_frequency = Counter(found_urls)
    unique_urls = list(dict.fromkeys(found_urls))
    hot_topic_urls = {url: count for url, count in url_frequency.items() if count >= 2}
    
    print(f"📊 Total unique URLs collected: {len(unique_urls)}")
    print(f"🔥 Hot topic URLs (cited ≥2 times): {len(hot_topic_urls)}")
    
    # Generate URL statistics report
    url_stats = {
        'unique_urls': unique_urls,
        'url_metadata_dates': url_metadata_dates,
        'hot_topic_urls': hot_topic_urls,
        'url_frequency': url_frequency,
        'all_sources': all_sources
    }
    
    # Print detailed date filtering report
    _print_date_filtering_report(unique_urls, url_metadata_dates)
    
    # Show collected URLs
    if unique_urls:
        print("=" * 80)
        print("📎 COLLECTED URLS FOR STAGE 2")
        print("=" * 80)
        for i, url in enumerate(unique_urls[:20], 1):
            print(f"{i}. {url}")
        if len(unique_urls) > 20:
            print(f"... and {len(unique_urls) - 20} more")
    
    # Combine all knowledge
    combined_knowledge = "=" * 80 + "".join(all_knowledge)
    print(f"✅ Knowledge building complete. Total knowledge: {len(combined_knowledge)} characters")
    
    return combined_knowledge, unique_urls, url_stats


def _print_date_filtering_report(unique_urls: List[str], url_metadata_dates: Dict):
    """Print detailed date filtering report."""
    from utils.date_utils import extract_year_from_url
    
    print("=" * 80)
    print("📅 URL DATE FILTERING REPORT")
    print("=" * 80)
    
    urls_with_metadata = 0
    urls_with_url_dates = 0
    urls_without_dates = 0
    year_distribution = Counter()
    
    for url in unique_urls:
        # Check if we have metadata date
        if url in url_metadata_dates:
            urls_with_metadata += 1
            year = url_metadata_dates[url].year
            year_distribution[year] += 1
        else:
            # Check URL pattern
            year = extract_year_from_url(url)
            if year:
                urls_with_url_dates += 1
                year_distribution[year] += 1
            else:
                urls_without_dates += 1
    
    total_with_dates = urls_with_metadata + urls_with_url_dates
    
    print(f"📊 Date Detection Summary:")
    print(f"   ✅ URLs with metadata dates (from Tavily): {urls_with_metadata}")
    print(f"   ✅ URLs with dates in URL pattern: {urls_with_url_dates}")
    print(f"   ⚠️ URLs without detectable dates: {urls_without_dates}")
    print(f"   📈 Total with dates: {total_with_dates}/{len(unique_urls)} ({100*total_with_dates/len(unique_urls):.1f}%)")
    
    print(f"\n📊 Year Distribution:")
    for year in sorted(year_distribution.keys(), reverse=True):
        bar = "█" * min(50, year_distribution[year])
        print(f"   {year}: {bar} ({year_distribution[year]} URLs)")
    
    if urls_without_dates > 0:
        print(f"\n⚠️ WARNING: {urls_without_dates} URLs have no detectable date")
        print(f"   These are INCLUDED (benefit of doubt) but should be manually verified:")
        
        # Show sample URLs without dates
        no_date_urls = [url for url in unique_urls[:50]
                        if url not in url_metadata_dates and extract_year_from_url(url) is None]
        for url in no_date_urls[:5]:
            print(f"   - {url}")
        if len(no_date_urls) > 5:
            print(f"   ... and {len(no_date_urls) - 5} more")


def create_url_content_mapping(knowledge_queries: List[Dict], all_sources: List[str]) -> Dict:
    """Create mapping of URLs to their content context."""
    import re
    
    print("=" * 80)
    print("🔍 DEBUGGING: URL-TO-CONTENT MAPPING")
    print("=" * 80)
    
    url_to_content = {}
    for idx, kq in enumerate(knowledge_queries[:9], 1):
        output = all_sources[idx-1] if idx-1 < len(all_sources) else ""
        urls_in_section = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', output)
        
        for url in urls_in_section:
            if url not in url_to_content:
                # Extract ~200 chars of context around the URL
                url_pos = output.find(url)
                context_start = max(0, url_pos - 100)
                context_end = min(len(output), url_pos + len(url) + 100)
                context = output[context_start:context_end]
                
                url_to_content[url] = {
                    'topic': kq['label'],
                    'context': context,
                    'full_section': output
                }
    
    print(f"📊 Mapped {len(url_to_content)} unique URLs to content")
    
    # Show sample mappings
    for i, (url, data) in enumerate(list(url_to_content.items())[:3], 1):
        print(f"{i}. {url[:60]}...")
        print(f"   Topic: {data['topic']}")
        print(f"   Context: {data['context'][:100]}...")
    
    return url_to_content
