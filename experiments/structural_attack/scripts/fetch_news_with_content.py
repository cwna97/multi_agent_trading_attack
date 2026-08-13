#!/usr/bin/env python3
"""
Fetch news with full article content using gnews + newspaper3k.
This script fetches fresh news and saves with full article text.

Usage:
    python scripts/fetch_news_with_content.py --ticker AAPL --start 2024-01-01 --end 2024-03-31
    python scripts/fetch_news_with_content.py --tickers AAPL,NVDA,TSLA --start 2024-01-01 --end 2024-12-31
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from newspaper import Article
from gnews import GNews
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.config import get_config


def get_local_data_dir():
    """Get local_data directory path."""
    config = get_config()
    base_dir = config.get("data_cache_dir", "tradingagents/dataflows/data_cache")
    return os.path.join(os.path.dirname(base_dir), "local_data")


def fetch_article_content(url: str, timeout: int = 10) -> dict:
    """
    Fetch full article content using newspaper3k.

    Returns dict with 'text', 'authors', 'publish_date', 'success'
    """
    result = {
        'text': '',
        'authors': [],
        'publish_date': None,
        'success': False,
        'error': None
    }

    try:
        article = Article(url)
        article.download()
        article.parse()

        if article.text and len(article.text) > 100:
            result['text'] = article.text[:5000]  # Limit to 5000 chars
            result['authors'] = article.authors
            result['publish_date'] = str(article.publish_date) if article.publish_date else None
            result['success'] = True
        else:
            result['error'] = 'Content too short'

    except Exception as e:
        result['error'] = str(e)[:100]

    return result


def fetch_news_for_date_range(ticker: str, start_date: str, end_date: str,
                               max_results_per_period: int = 10) -> dict:
    """
    Fetch news for a ticker over a date range.

    Breaks range into monthly chunks to get better coverage.
    Returns dict: {date_str: [articles]}
    """
    all_news = {}

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Process month by month
    current = start
    while current <= end:
        month_end = min(
            datetime(current.year, current.month + 1, 1) - timedelta(days=1) if current.month < 12
            else datetime(current.year, 12, 31),
            end
        )

        print(f"  Fetching {current.strftime('%Y-%m')}...")

        # Initialize GNews with date range
        google_news = GNews(
            language='en',
            country='US',
            start_date=(current.year, current.month, current.day),
            end_date=(month_end.year, month_end.month, month_end.day),
            max_results=max_results_per_period
        )

        try:
            # Search for ticker-related news
            news_items = google_news.get_news(f'{ticker} stock')

            for item in news_items:
                # Get article URL
                url = item.get('url', '')
                if not url:
                    continue

                # Parse published date
                pub_date_str = item.get('published date', '')
                try:
                    # gnews returns date like "Mon, 01 Jan 2024 12:00:00 GMT"
                    pub_date = datetime.strptime(pub_date_str[:16], "%a, %d %b %Y")
                    date_key = pub_date.strftime("%Y-%m-%d")
                except:
                    date_key = current.strftime("%Y-%m-%d")

                # Fetch full article content
                content_result = fetch_article_content(url)

                article_data = {
                    'headline': item.get('title', ''),
                    'summary': item.get('description', ''),
                    'source': item.get('publisher', {}).get('title', 'Unknown'),
                    'link': url,
                    'content': content_result['text'] if content_result['success'] else '',
                    'authors': content_result['authors'],
                    'fetch_success': content_result['success'],
                }

                if date_key not in all_news:
                    all_news[date_key] = []
                all_news[date_key].append(article_data)

                # Small delay to be polite
                time.sleep(0.5)

        except Exception as e:
            print(f"    Error fetching news: {e}")

        # Move to next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    return all_news


def main():
    parser = argparse.ArgumentParser(description='Fetch news with full article content')
    parser.add_argument('--ticker', type=str, help='Single ticker (e.g., AAPL)')
    parser.add_argument('--tickers', type=str, help='Comma-separated tickers (e.g., AAPL,NVDA,TSLA)')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--max-per-month', type=int, default=20, help='Max articles per month')
    args = parser.parse_args()

    # Determine tickers
    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    else:
        print("Please specify --ticker or --tickers")
        return

    local_dir = get_local_data_dir()
    news_dir = os.path.join(local_dir, "news_data")
    os.makedirs(news_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("NEWS FETCH WITH CONTENT")
    print(f"{'='*60}")
    print(f"Tickers: {tickers}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Max per month: {args.max_per_month}")
    print(f"Output dir: {news_dir}")

    for ticker in tickers:
        print(f"\n[{ticker}] Fetching news...")

        # Fetch news
        news_data = fetch_news_for_date_range(
            ticker, args.start, args.end,
            max_results_per_period=args.max_per_month
        )

        # Load existing data if present
        news_file = os.path.join(news_dir, f"{ticker}_news.json")
        if os.path.exists(news_file):
            with open(news_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            # Merge: new data takes precedence
            existing_data.update(news_data)
            news_data = existing_data

        # Save
        with open(news_file, 'w', encoding='utf-8') as f:
            json.dump(news_data, f, indent=2, ensure_ascii=False)

        # Statistics
        total_articles = sum(len(articles) for articles in news_data.values())
        with_content = sum(
            1 for articles in news_data.values()
            for a in articles if a.get('content') and len(a.get('content', '')) > 100
        )

        print(f"  Saved: {news_file}")
        print(f"  Total articles: {total_articles}")
        print(f"  With content: {with_content} ({with_content/total_articles*100:.1f}%)" if total_articles > 0 else "")

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
