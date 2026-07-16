#!/usr/bin/env python3
"""
Enrich news JSON files with article content.
Fetches the actual article text from news links and adds it to the JSON.

Usage:
    python scripts/enrich_news_content.py
    python scripts/enrich_news_content.py --ticker AAPL
    python scripts/enrich_news_content.py --all
"""

import os
import sys
import json
import time
import argparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urlparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.config import get_config

# Request headers to mimic browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

# Timeout for requests
REQUEST_TIMEOUT = 15

# Max content length (characters)
MAX_CONTENT_LENGTH = 5000


def get_local_data_dir():
    """Get local_data directory path."""
    config = get_config()
    base_dir = config.get("data_cache_dir", "tradingagents/dataflows/data_cache")
    return os.path.join(os.path.dirname(base_dir), "local_data")


def clean_text(text: str) -> str:
    """Clean extracted text."""
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_article_content(html: str, url: str) -> str:
    """
    Extract main article content from HTML.
    Uses multiple strategies based on common news site structures.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Remove unwanted elements
    for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer',
                                   'aside', 'advertisement', 'iframe', 'noscript']):
        element.decompose()

    content = ""

    # Strategy 1: Look for article tag
    article = soup.find('article')
    if article:
        paragraphs = article.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])

    # Strategy 2: Look for common content class names
    if not content or len(content) < 200:
        content_selectors = [
            {'class_': re.compile(r'article-body|article-content|story-body|post-content|entry-content', re.I)},
            {'class_': re.compile(r'content|body|main', re.I)},
            {'itemprop': 'articleBody'},
            {'data-component': 'text-block'},
        ]

        for selector in content_selectors:
            container = soup.find(['div', 'section', 'article'], selector)
            if container:
                paragraphs = container.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
                if len(content) > 200:
                    break

    # Strategy 3: Get all paragraphs from body
    if not content or len(content) < 200:
        body = soup.find('body')
        if body:
            paragraphs = body.find_all('p')
            # Filter out short paragraphs (likely navigation/ads)
            content = ' '.join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])

    # Clean and truncate
    content = clean_text(content)

    if len(content) > MAX_CONTENT_LENGTH:
        content = content[:MAX_CONTENT_LENGTH] + "..."

    return content


def resolve_google_news_url(url: str) -> str:
    """
    Resolve Google News redirect URL to actual article URL.
    Google News URLs redirect to the actual source.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT,
                              allow_redirects=True)
        return response.url
    except:
        return url


def fetch_article_content(url: str) -> dict:
    """
    Fetch and extract content from a news article URL.

    Returns:
        dict with 'content', 'resolved_url', 'success', 'error'
    """
    result = {
        'content': '',
        'resolved_url': url,
        'success': False,
        'error': None
    }

    try:
        # Handle Google News URLs
        if 'news.google.com' in url:
            resolved_url = resolve_google_news_url(url)
            result['resolved_url'] = resolved_url
        else:
            resolved_url = url

        # Fetch the page
        response = requests.get(resolved_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        # Extract content
        content = extract_article_content(response.text, resolved_url)

        if content and len(content) > 100:
            result['content'] = content
            result['success'] = True
        else:
            result['error'] = 'Content too short or empty'

    except requests.exceptions.Timeout:
        result['error'] = 'Timeout'
    except requests.exceptions.RequestException as e:
        result['error'] = f'Request failed: {str(e)[:50]}'
    except Exception as e:
        result['error'] = f'Error: {str(e)[:50]}'

    return result


def enrich_news_file(news_file: str, max_workers: int = 3) -> dict:
    """
    Enrich a news JSON file with article content.

    Args:
        news_file: Path to news JSON file
        max_workers: Number of parallel workers for fetching

    Returns:
        Statistics dict
    """
    stats = {
        'total_articles': 0,
        'already_enriched': 0,
        'newly_enriched': 0,
        'failed': 0
    }

    # Load existing data
    with open(news_file, 'r', encoding='utf-8') as f:
        news_data = json.load(f)

    # Collect articles that need content
    articles_to_fetch = []

    for date_str, articles in news_data.items():
        for idx, article in enumerate(articles):
            stats['total_articles'] += 1

            # Skip if already has content
            if article.get('content') and len(article.get('content', '')) > 100:
                stats['already_enriched'] += 1
                continue

            # Skip if no link
            if not article.get('link'):
                continue

            articles_to_fetch.append({
                'date': date_str,
                'idx': idx,
                'url': article['link'],
                'headline': article.get('headline', '')[:50]
            })

    if not articles_to_fetch:
        print(f"  No articles need enrichment")
        return stats

    print(f"  Fetching content for {len(articles_to_fetch)} articles...")

    # Fetch articles with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_article = {
            executor.submit(fetch_article_content, article['url']): article
            for article in articles_to_fetch
        }

        for future in tqdm(as_completed(future_to_article), total=len(articles_to_fetch),
                          desc="  Fetching", leave=False):
            article_info = future_to_article[future]
            result = future.result()

            if result['success']:
                # Update the article in news_data
                news_data[article_info['date']][article_info['idx']]['content'] = result['content']
                news_data[article_info['date']][article_info['idx']]['resolved_url'] = result['resolved_url']
                stats['newly_enriched'] += 1
            else:
                stats['failed'] += 1

            # Small delay to be respectful to servers
            time.sleep(0.5)

    # Save updated data
    with open(news_file, 'w', encoding='utf-8') as f:
        json.dump(news_data, f, indent=2, ensure_ascii=False)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Enrich news JSON files with article content')
    parser.add_argument('--ticker', type=str, help='Process single ticker (e.g., AAPL)')
    parser.add_argument('--all', action='store_true', help='Process all news files')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers')
    args = parser.parse_args()

    local_dir = get_local_data_dir()
    news_dir = os.path.join(local_dir, "news_data")

    if not os.path.exists(news_dir):
        print(f"News directory not found: {news_dir}")
        return

    # Find news files to process
    if args.ticker:
        news_files = [os.path.join(news_dir, f"{args.ticker}_news.json")]
        if not os.path.exists(news_files[0]):
            print(f"News file not found: {news_files[0]}")
            return
    elif args.all:
        news_files = [
            os.path.join(news_dir, f)
            for f in os.listdir(news_dir)
            if f.endswith('_news.json')
        ]
    else:
        # Default: show available files and ask
        available_files = [f for f in os.listdir(news_dir) if f.endswith('_news.json')]
        print("Available news files:")
        for f in available_files:
            print(f"  - {f}")
        print("\nUsage:")
        print("  python scripts/enrich_news_content.py --ticker AAPL")
        print("  python scripts/enrich_news_content.py --all")
        return

    print(f"\n{'='*60}")
    print("NEWS CONTENT ENRICHMENT")
    print(f"{'='*60}")

    total_stats = {
        'total_articles': 0,
        'already_enriched': 0,
        'newly_enriched': 0,
        'failed': 0
    }

    for news_file in news_files:
        ticker = os.path.basename(news_file).replace('_news.json', '')
        print(f"\n[{ticker}] Processing {news_file}...")

        stats = enrich_news_file(news_file, max_workers=args.workers)

        for key in total_stats:
            total_stats[key] += stats[key]

        print(f"  ✓ Total: {stats['total_articles']}, "
              f"Already enriched: {stats['already_enriched']}, "
              f"Newly enriched: {stats['newly_enriched']}, "
              f"Failed: {stats['failed']}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total articles processed: {total_stats['total_articles']}")
    print(f"Already had content: {total_stats['already_enriched']}")
    print(f"Newly enriched: {total_stats['newly_enriched']}")
    print(f"Failed to fetch: {total_stats['failed']}")

    success_rate = (total_stats['newly_enriched'] /
                   (total_stats['total_articles'] - total_stats['already_enriched']) * 100
                   if (total_stats['total_articles'] - total_stats['already_enriched']) > 0 else 0)
    print(f"Success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    main()
