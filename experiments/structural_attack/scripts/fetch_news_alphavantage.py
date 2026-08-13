#!/usr/bin/env python3
"""
Fetch news data from Alpha Vantage NEWS_SENTIMENT API.
Fetches quarterly to handle the 1000 article limit per request.

Usage:
    python scripts/fetch_news_alphavantage.py --ticker AAPL --year 2025
    python scripts/fetch_news_alphavantage.py --all-tickers --year 2025
    python scripts/fetch_news_alphavantage.py --all-tickers --year 2025 --quarter 1  # Q1 only
    python scripts/fetch_news_alphavantage.py --all-tickers --start-date 2025-01-01 --end-date 2025-03-31

Free tier: 25 requests/day
Each ticker + quarter = 1 request
7 tickers × 4 quarters = 28 requests = 2 days
"""

import os
import sys
import json
import argparse
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.config import get_config

# Alpha Vantage API settings
API_BASE_URL = "https://www.alphavantage.co/query"
REQUESTS_PER_DAY = 25  # Free tier limit


def get_local_data_dir():
    """Get local_data directory path."""
    config = get_config()
    base_dir = config.get("data_cache_dir", "tradingagents/dataflows/data_cache")
    return os.path.join(os.path.dirname(base_dir), "local_data")


def get_api_key():
    """Get Alpha Vantage API key from environment or config."""
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        config = get_config()
        api_key = config.get("alpha_vantage_api_key")
    if not api_key:
        raise ValueError(
            "Alpha Vantage API key not found. Set ALPHA_VANTAGE_API_KEY environment variable "
            "or add 'alpha_vantage_api_key' to config."
        )
    return api_key


def fetch_news_sentiment(ticker: str, time_from: str, time_to: str,
                         api_key: str, limit: int = 1000) -> dict:
    """
    Fetch news sentiment data from Alpha Vantage.

    Args:
        ticker: Stock ticker symbol
        time_from: Start datetime (YYYYMMDDTHHMM format)
        time_to: End datetime (YYYYMMDDTHHMM format)
        api_key: Alpha Vantage API key
        limit: Max results (default 1000, max 1000)

    Returns:
        API response as dict
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "time_to": time_to,
        "limit": min(limit, 1000),
        "apikey": api_key
    }

    response = requests.get(API_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def parse_alphavantage_news(api_response: dict, ticker: str) -> dict:
    """
    Parse Alpha Vantage news response into date-keyed format.

    Returns:
        dict: {date_str: [articles]}
    """
    news_by_date = {}

    feed = api_response.get("feed", [])

    for item in feed:
        # Parse publish time (format: 20250115T120000)
        time_published = item.get("time_published", "")
        try:
            dt = datetime.strptime(time_published[:8], "%Y%m%d")
            date_key = dt.strftime("%Y-%m-%d")
        except:
            continue

        # Get ticker-specific sentiment
        ticker_sentiment = None
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker") == ticker:
                ticker_sentiment = ts
                break

        article = {
            "headline": item.get("title", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source", ""),
            "link": item.get("url", ""),
            "content": item.get("summary", ""),  # Alpha Vantage provides summary as content
            "authors": item.get("authors", []),
            "overall_sentiment_score": item.get("overall_sentiment_score"),
            "overall_sentiment_label": item.get("overall_sentiment_label"),
            "ticker_sentiment_score": ticker_sentiment.get("ticker_sentiment_score") if ticker_sentiment else None,
            "ticker_sentiment_label": ticker_sentiment.get("ticker_sentiment_label") if ticker_sentiment else None,
            "relevance_score": ticker_sentiment.get("relevance_score") if ticker_sentiment else None,
            "fetch_source": "alpha_vantage"
        }

        if date_key not in news_by_date:
            news_by_date[date_key] = []
        news_by_date[date_key].append(article)

    return news_by_date


def get_quarter_ranges(year: int) -> list:
    """Get date ranges for each quarter of a year."""
    return [
        (f"{year}-01-01", f"{year}-03-31", "Q1"),
        (f"{year}-04-01", f"{year}-06-30", "Q2"),
        (f"{year}-07-01", f"{year}-09-30", "Q3"),
        (f"{year}-10-01", f"{year}-12-31", "Q4"),
    ]


def date_to_api_format(date_str: str, end_of_day: bool = False) -> str:
    """Convert YYYY-MM-DD to YYYYMMDDTHHMM format."""
    if end_of_day:
        return date_str.replace("-", "") + "T2359"
    return date_str.replace("-", "") + "T0000"


def fetch_ticker_news(ticker: str, start_date: str, end_date: str,
                      api_key: str, desc: str = "") -> tuple:
    """
    Fetch news for a single ticker and date range.

    Returns:
        tuple: (news_by_date dict, article_count, api_calls_used)
    """
    time_from = date_to_api_format(start_date)
    time_to = date_to_api_format(end_date, end_of_day=True)

    try:
        response = fetch_news_sentiment(ticker, time_from, time_to, api_key)

        # Check for API errors
        if "Error Message" in response:
            print(f"  API Error: {response['Error Message']}")
            return {}, 0, 1

        if "Note" in response:
            print(f"  API Note: {response['Note']}")
            return {}, 0, 1

        news_by_date = parse_alphavantage_news(response, ticker)
        article_count = sum(len(articles) for articles in news_by_date.values())

        return news_by_date, article_count, 1

    except Exception as e:
        print(f"  Error fetching {ticker} {desc}: {e}")
        return {}, 0, 1


def merge_news_data(existing: dict, new: dict) -> dict:
    """Merge new news data into existing, avoiding duplicates."""
    merged = existing.copy()

    for date_key, articles in new.items():
        if date_key not in merged:
            merged[date_key] = []

        # Get existing headlines for deduplication
        existing_headlines = {a.get("headline", "") for a in merged[date_key]}

        for article in articles:
            if article.get("headline", "") not in existing_headlines:
                merged[date_key].append(article)

    return merged


def main():
    parser = argparse.ArgumentParser(description='Fetch news from Alpha Vantage')
    parser.add_argument('--ticker', type=str, help='Single ticker')
    parser.add_argument('--all-tickers', action='store_true', help='Process all 7 tickers')
    parser.add_argument('--year', type=int, help='Year to fetch (fetches quarterly)')
    parser.add_argument('--quarter', type=int, choices=[1, 2, 3, 4], help='Specific quarter only')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, default=1000, help='Max articles per request')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without fetching')
    args = parser.parse_args()

    # Determine tickers
    DEFAULT_TICKERS = ["TSLA", "NDX", "NVDA", "MSFT", "AMZN", "GOOGL", "PLTR"]

    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.all_tickers:
        tickers = DEFAULT_TICKERS
    else:
        print("Please specify --ticker or --all-tickers")
        print(f"Available tickers: {', '.join(DEFAULT_TICKERS)}")
        return

    # Determine date ranges
    fetch_tasks = []  # List of (ticker, start_date, end_date, description)

    if args.year:
        quarters = get_quarter_ranges(args.year)
        if args.quarter:
            quarters = [quarters[args.quarter - 1]]

        for ticker in tickers:
            for start, end, q_name in quarters:
                fetch_tasks.append((ticker, start, end, f"{args.year} {q_name}"))

    elif args.start_date and args.end_date:
        for ticker in tickers:
            fetch_tasks.append((ticker, args.start_date, args.end_date,
                              f"{args.start_date}~{args.end_date}"))
    else:
        print("Please specify --year or both --start-date and --end-date")
        return

    # Setup
    local_dir = get_local_data_dir()
    news_dir = os.path.join(local_dir, "news_data", "alpha_vantage")
    os.makedirs(news_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("ALPHA VANTAGE NEWS FETCH")
    print(f"{'='*60}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Tasks: {len(fetch_tasks)} API calls")
    print(f"Free tier limit: {REQUESTS_PER_DAY}/day")
    print(f"Estimated days needed: {(len(fetch_tasks) + REQUESTS_PER_DAY - 1) // REQUESTS_PER_DAY}")
    print(f"Output: {news_dir}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("Planned API calls:")
        for i, (ticker, start, end, desc) in enumerate(fetch_tasks, 1):
            print(f"  {i:2d}. {ticker:5s} | {start} ~ {end} | {desc}")
        print(f"\nTotal: {len(fetch_tasks)} calls")
        return

    # Get API key
    try:
        api_key = get_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Track progress
    total_calls = 0
    total_articles = 0
    ticker_stats = {t: 0 for t in tickers}

    # Process tasks
    for ticker, start_date, end_date, desc in tqdm(fetch_tasks, desc="Fetching"):
        # Check daily limit
        if total_calls >= REQUESTS_PER_DAY:
            print(f"\n⚠️  Daily limit reached ({REQUESTS_PER_DAY} calls)")
            print("Run again tomorrow to continue.")
            break

        # Fetch news
        news_by_date, count, calls = fetch_ticker_news(
            ticker, start_date, end_date, api_key, desc
        )

        total_calls += calls
        total_articles += count
        ticker_stats[ticker] += count

        if news_by_date:
            # Load existing data
            news_file = os.path.join(news_dir, f"{ticker}_news.json")
            if os.path.exists(news_file):
                with open(news_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {}

            # Merge and save
            merged_data = merge_news_data(existing_data, news_by_date)
            with open(news_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)

        # Rate limiting (5 calls/min for free tier)
        time.sleep(12)  # 12 seconds between calls = 5/min

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"API calls used: {total_calls}/{REQUESTS_PER_DAY}")
    print(f"Total articles fetched: {total_articles}")
    print(f"\nPer ticker:")
    for ticker, count in ticker_stats.items():
        print(f"  {ticker}: {count} articles")

    remaining_tasks = len(fetch_tasks) - total_calls
    if remaining_tasks > 0:
        print(f"\n⚠️  {remaining_tasks} tasks remaining. Run again tomorrow.")


if __name__ == "__main__":
    main()
