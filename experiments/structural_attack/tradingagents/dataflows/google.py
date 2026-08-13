from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .googlenews_utils import getNewsData
from .googlenews_cache import (
    load_cached_google_news,
    cache_google_news,
)
from .config import get_config


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    use_cache: bool = True,
) -> str:
    """
    Get Google News for a query with caching support.

    Args:
        query: Search query (e.g., ticker symbol like "AAPL")
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back
        use_cache: Whether to use cached data (default True)

    Returns:
        Formatted string of news articles
    """
    query_formatted = query.replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before_str = before.strftime("%Y-%m-%d")

    config = get_config()
    cache_dir = config.get("data_cache_dir", "data_cache")

    news_results = []

    # Try to load from cache first
    if use_cache:
        cached_data = load_cached_google_news(query, before_str, curr_date, cache_dir)
        if cached_data:
            # Flatten cached data into list format
            for news_list in cached_data.values():
                news_results.extend(news_list)

    # If no cache or insufficient data, fetch fresh
    if not news_results:
        fresh_results = getNewsData(query_formatted, before_str, curr_date)
        if fresh_results:
            # Cache the fresh results
            cache_google_news(query, fresh_results, cache_dir, curr_date)
            news_results = fresh_results

    # Format output
    news_str = ""
    for news in news_results:
        title = news.get('title', '')
        source = news.get('source', 'Unknown')
        snippet = news.get('snippet', '')
        news_str += f"### {title} (source: {source}) \n\n{snippet}\n\n"

    if not news_results:
        return ""

    return f"## {query} Google News, from {before_str} to {curr_date}:\n\n{news_str}"