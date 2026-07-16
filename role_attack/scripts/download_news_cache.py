#!/usr/bin/env python3
"""
Download and cache Google News for specified stock tickers.

Usage:
    uv run python scripts/download_news_cache.py
"""

import os
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.googlenews_utils import getNewsData
from tradingagents.dataflows.googlenews_cache import (
    cache_google_news,
    get_cache_stats,
)
from tradingagents.dataflows.config import get_config


# Configuration
TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "TSLA", "GOOGL", "META"]
START_DATE = "2024-01-01"
MONTHS_TO_DOWNLOAD = 6


def download_news_for_ticker(ticker: str, cache_dir: str, start_date: str, end_date: str) -> dict:
    """Download news for a single ticker."""
    print(f"\n{'='*60}")
    print(f"Downloading news for {ticker}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*60}")

    try:
        # gnews가 일별로 fetch하므로 한 번에 전체 기간 요청
        news_results = getNewsData(ticker, start_date, end_date, fetch_full_article=False)

        if news_results:
            cache_google_news(ticker, news_results, cache_dir, end_date)
            print(f"Cached {len(news_results)} articles")
            return {"ticker": ticker, "total_articles": len(news_results), "errors": 0}
        else:
            print("No articles found")
            return {"ticker": ticker, "total_articles": 0, "errors": 0}

    except Exception as e:
        print(f"ERROR: {e}")
        return {"ticker": ticker, "total_articles": 0, "errors": 1}


def main():
    print("=" * 60)
    print("Google News Cache Downloader")
    print("=" * 60)
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Period: {START_DATE} + {MONTHS_TO_DOWNLOAD} months")

    config = get_config()
    cache_dir = config.get("data_cache_dir", "tradingagents/dataflows/data_cache")
    print(f"Cache directory: {cache_dir}")

    os.makedirs(os.path.join(cache_dir, "google_news"), exist_ok=True)

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = start_dt + relativedelta(months=MONTHS_TO_DOWNLOAD)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    results = []
    for i, ticker in enumerate(TICKERS):
        print(f"\n[{i+1}/{len(TICKERS)}] Processing {ticker}...")
        result = download_news_for_ticker(ticker, cache_dir, start_str, end_str)
        results.append(result)

        stats = get_cache_stats(ticker, cache_dir)
        if stats["exists"]:
            print(f"  Cache: {stats['total_articles']} articles, {stats['total_days']} days")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = sum(r["total_articles"] for r in results)
    errors = sum(r["errors"] for r in results)
    for r in results:
        print(f"  {r['ticker']}: {r['total_articles']} articles")
    print(f"\nTotal: {total} articles, {errors} errors")
    print("\nDone!")


if __name__ == "__main__":
    main()
