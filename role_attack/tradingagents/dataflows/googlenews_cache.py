"""
Google News caching utilities for TradingAgents.
Provides functions to cache and retrieve Google News results.
"""

import os
import json
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def normalize_relative_date(relative_date: str, reference_date: str) -> str:
    """
    Convert relative date strings to absolute ISO format dates.

    Args:
        relative_date: Relative date string (e.g., "2 hours ago", "3 days ago", "1 week ago")
        reference_date: Reference date in yyyy-mm-dd format

    Returns:
        Date string in yyyy-mm-dd format
    """
    ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")
    relative_date_lower = relative_date.lower().strip()

    # Handle various relative date formats
    if "hour" in relative_date_lower or "minute" in relative_date_lower or "just now" in relative_date_lower:
        # Same day
        return reference_date

    # Extract number from string
    numbers = re.findall(r'\d+', relative_date_lower)
    num = int(numbers[0]) if numbers else 1

    if "day" in relative_date_lower:
        result_dt = ref_dt - timedelta(days=num)
    elif "week" in relative_date_lower:
        result_dt = ref_dt - timedelta(weeks=num)
    elif "month" in relative_date_lower:
        result_dt = ref_dt - timedelta(days=num * 30)
    elif "year" in relative_date_lower:
        result_dt = ref_dt - timedelta(days=num * 365)
    else:
        # Try to parse as absolute date
        try:
            # Handle formats like "Jan 15, 2024" or "15 Jan 2024"
            for fmt in ["%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y"]:
                try:
                    result_dt = datetime.strptime(relative_date, fmt)
                    return result_dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
            # If all parsing fails, return reference date
            return reference_date
        except Exception:
            return reference_date

    return result_dt.strftime("%Y-%m-%d")


def compute_news_hash(news_item: dict) -> str:
    """
    Compute a hash for a news item to detect duplicates.

    Args:
        news_item: Dictionary containing news data

    Returns:
        MD5 hash string
    """
    unique_str = f"{news_item.get('link', '')}{news_item.get('title', '')}"
    return hashlib.md5(unique_str.encode()).hexdigest()


def get_cache_path(query: str, cache_dir: str) -> str:
    """
    Get the cache file path for a query.

    Args:
        query: Search query (ticker symbol or search term)
        cache_dir: Base cache directory

    Returns:
        Full path to cache file
    """
    # Sanitize query for filename
    safe_query = re.sub(r'[^\w\-]', '_', query.upper())
    news_cache_dir = os.path.join(cache_dir, "google_news")
    os.makedirs(news_cache_dir, exist_ok=True)
    return os.path.join(news_cache_dir, f"{safe_query}_news.json")


def load_cached_google_news(
    query: str,
    start_date: str,
    end_date: str,
    cache_dir: str
) -> Dict[str, List[dict]]:
    """
    Load cached Google News results for a date range.

    Args:
        query: Search query
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
        cache_dir: Cache directory path

    Returns:
        Dictionary mapping dates to lists of news items
    """
    cache_path = get_cache_path(query, cache_dir)

    if not os.path.exists(cache_path):
        return {}

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

    # Filter by date range
    filtered = {}
    for date_key, news_list in cache_data.items():
        if start_date <= date_key <= end_date:
            filtered[date_key] = news_list

    return filtered


def cache_google_news(
    query: str,
    news_results: List[dict],
    cache_dir: str,
    reference_date: str
) -> None:
    """
    Cache Google News results to JSON file.

    Args:
        query: Search query
        news_results: List of news items from getNewsData()
        cache_dir: Cache directory path
        reference_date: Reference date for normalizing relative dates
    """
    cache_path = get_cache_path(query, cache_dir)

    # Load existing cache if available
    cache_data = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            cache_data = {}

    # Track existing hashes for deduplication
    existing_hashes = set()
    for date_key, news_list in cache_data.items():
        for news in news_list:
            if 'hash' in news:
                existing_hashes.add(news['hash'])

    # Process and add new results
    for news in news_results:
        news_hash = compute_news_hash(news)

        # Skip duplicates
        if news_hash in existing_hashes:
            continue

        # Normalize the date
        date_key = normalize_relative_date(news.get('date', ''), reference_date)

        # Add metadata
        news_with_meta = news.copy()
        news_with_meta['date_original'] = news.get('date', '')
        news_with_meta['retrieved_at'] = datetime.now().isoformat()
        news_with_meta['hash'] = news_hash

        # Add to cache data
        if date_key not in cache_data:
            cache_data[date_key] = []
        cache_data[date_key].append(news_with_meta)
        existing_hashes.add(news_hash)

    # Sort cache data by date (newest first)
    sorted_cache = dict(sorted(cache_data.items(), reverse=True))

    # Save cache
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_cache, f, ensure_ascii=False, indent=2)


def is_date_range_cached(
    query: str,
    start_date: str,
    end_date: str,
    cache_dir: str,
    min_coverage: float = 0.5
) -> bool:
    """
    Check if a date range has sufficient cached data.

    Args:
        query: Search query
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
        cache_dir: Cache directory path
        min_coverage: Minimum fraction of days that must have data (default 0.5)

    Returns:
        True if cache has sufficient coverage
    """
    cached_data = load_cached_google_news(query, start_date, end_date, cache_dir)

    if not cached_data:
        return False

    # Calculate expected trading days (rough estimate: exclude weekends)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1

    # Count days with cached data
    cached_days = len(cached_data)

    # Check coverage
    coverage = cached_days / max(total_days, 1)
    return coverage >= min_coverage


def get_cache_stats(query: str, cache_dir: str) -> dict:
    """
    Get statistics about cached news for a query.

    Args:
        query: Search query
        cache_dir: Cache directory path

    Returns:
        Dictionary with cache statistics
    """
    cache_path = get_cache_path(query, cache_dir)

    if not os.path.exists(cache_path):
        return {
            "exists": False,
            "total_articles": 0,
            "total_days": 0,
            "date_range": None
        }

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {
            "exists": False,
            "total_articles": 0,
            "total_days": 0,
            "date_range": None
        }

    total_articles = sum(len(news_list) for news_list in cache_data.values())
    dates = sorted(cache_data.keys())

    return {
        "exists": True,
        "total_articles": total_articles,
        "total_days": len(dates),
        "date_range": (dates[0], dates[-1]) if dates else None
    }
