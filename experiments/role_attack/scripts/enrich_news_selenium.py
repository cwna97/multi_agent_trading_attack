#!/usr/bin/env python3
"""
Enrich news JSON files with article content using Selenium (Parallel + Optimized).
Fetches actual article text by following Google News redirects via browser automation.

Usage:
    python scripts/enrich_news_selenium.py --ticker AAPL --limit 10
    python scripts/enrich_news_selenium.py --all-tickers --start-date 2025-10-19 --end-date 2025-11-03
    python scripts/enrich_news_selenium.py --all-tickers --start-date 2025-10-19 --end-date 2025-11-03 --exclude AAPL,META
    python scripts/enrich_news_selenium.py --all-tickers --workers 8  # 8개 병렬 처리
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

from tradingagents.dataflows.config import get_config

# Thread-local storage for drivers
thread_local = threading.local()

# Lock for file writing
file_lock = threading.Lock()


def get_local_data_dir():
    """Get local_data directory path."""
    config = get_config()
    base_dir = config.get("data_cache_dir", "tradingagents/dataflows/data_cache")
    return os.path.join(os.path.dirname(base_dir), "local_data")


def create_driver(headless: bool = True):
    """Create and configure optimized Firefox WebDriver."""
    options = Options()
    if headless:
        options.add_argument("--headless")

    # Performance optimizations
    options.set_preference("general.useragent.override",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0")

    # Disable images for faster loading
    options.set_preference("permissions.default.image", 2)

    # Disable CSS
    options.set_preference("permissions.default.stylesheet", 2)

    # Disable Flash
    options.set_preference("dom.ipc.plugins.enabled.libflashplayer.so", False)

    # Disable unnecessary features
    options.set_preference("media.autoplay.default", 0)
    options.set_preference("media.autoplay.enabled", False)
    options.set_preference("browser.cache.disk.enable", False)
    options.set_preference("browser.cache.memory.enable", False)
    options.set_preference("browser.cache.offline.enable", False)
    options.set_preference("network.http.use-cache", False)

    # Faster page load strategy
    options.page_load_strategy = 'eager'  # Don't wait for all resources

    try:
        service = Service("/snap/bin/geckodriver")
        driver = webdriver.Firefox(service=service, options=options)
        driver.set_page_load_timeout(10)  # 10 second timeout
        return driver
    except Exception as e:
        print(f"Error creating driver: {e}")
        return None


def get_driver(headless: bool = True):
    """Get or create a thread-local driver."""
    if not hasattr(thread_local, "driver") or thread_local.driver is None:
        thread_local.driver = create_driver(headless)
    return thread_local.driver


def close_thread_driver():
    """Close the thread-local driver."""
    if hasattr(thread_local, "driver") and thread_local.driver:
        try:
            thread_local.driver.quit()
        except:
            pass
        thread_local.driver = None


def extract_article_content(driver, url: str, timeout: int = 5) -> dict:
    """
    Navigate to Google News URL and extract article content.
    Optimized for speed with shorter timeouts.
    """
    result = {
        'content': '',
        'resolved_url': '',
        'title': '',
        'success': False,
        'error': None
    }

    try:
        driver.get(url)

        # Wait for redirect (shorter timeout)
        WebDriverWait(driver, timeout).until(
            lambda d: 'news.google.com' not in d.current_url
        )

        # Minimal wait for content
        time.sleep(0.3)

        result['resolved_url'] = driver.current_url

        try:
            result['title'] = driver.title
        except:
            pass

        # Extract content - try fastest methods first
        content = ""

        # Strategy 1: article tag
        try:
            article = driver.find_element(By.TAG_NAME, "article")
            paragraphs = article.find_elements(By.TAG_NAME, "p")
            content = ' '.join([p.text for p in paragraphs if p.text.strip()])
        except:
            pass

        # Strategy 2: common selectors (only if needed)
        if not content or len(content) < 200:
            for selector in ["[class*='article-body']", "[class*='story-body']",
                           "[itemprop='articleBody']", "main p"]:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        content = ' '.join([e.text for e in elements if e.text.strip()])
                        if len(content) > 200:
                            break
                except:
                    continue

        # Strategy 3: all paragraphs (last resort)
        if not content or len(content) < 200:
            try:
                paragraphs = driver.find_elements(By.TAG_NAME, "p")
                content = ' '.join([p.text for p in paragraphs if len(p.text) > 50])
            except:
                pass

        content = ' '.join(content.split())

        if content and len(content) > 100:
            result['content'] = content[:5000] + "..." if len(content) > 5000 else content
            result['success'] = True
        else:
            result['error'] = 'Content too short'

    except TimeoutException:
        result['error'] = 'Timeout'
    except WebDriverException as e:
        result['error'] = f'WebDriver: {str(e)[:30]}'
    except Exception as e:
        result['error'] = f'Error: {str(e)[:30]}'

    return result


def process_article(article_info: dict, headless: bool = True) -> dict:
    """Process a single article (for parallel execution)."""
    driver = get_driver(headless)
    if not driver:
        return {**article_info, 'result': {'success': False, 'error': 'No driver'}}

    result = extract_article_content(driver, article_info['url'])
    return {**article_info, 'result': result}


def enrich_news_file(news_file: str, limit: int = None, headless: bool = True,
                     start_date: str = None, end_date: str = None,
                     ticker_name: str = "", num_workers: int = 4) -> dict:
    """
    Enrich a news JSON file with article content using parallel Selenium.
    """
    stats = {
        'total_articles': 0,
        'already_enriched': 0,
        'newly_enriched': 0,
        'failed': 0,
        'skipped': 0,
        'out_of_range': 0
    }

    with open(news_file, 'r', encoding='utf-8') as f:
        news_data = json.load(f)

    articles_to_fetch = []

    for date_str, articles in news_data.items():
        if start_date and date_str < start_date:
            stats['out_of_range'] += len(articles)
            continue
        if end_date and date_str > end_date:
            stats['out_of_range'] += len(articles)
            continue

        for idx, article in enumerate(articles):
            stats['total_articles'] += 1

            if article.get('content') and len(article.get('content', '')) > 100:
                stats['already_enriched'] += 1
                continue

            link = article.get('link', '')
            if not link or 'news.google.com' not in link:
                stats['skipped'] += 1
                continue

            articles_to_fetch.append({
                'date': date_str,
                'idx': idx,
                'url': link,
                'headline': article.get('headline', '')[:50]
            })

    if not articles_to_fetch:
        return stats

    if limit:
        articles_to_fetch = articles_to_fetch[:limit]

    # Progress bar
    pbar = tqdm(
        total=len(articles_to_fetch),
        desc=f"{ticker_name:>6}",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
    )

    # Parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_article, article, headless): article
            for article in articles_to_fetch
        }

        processed = 0
        for future in as_completed(futures):
            result_data = future.result()
            result = result_data['result']

            if result['success']:
                with file_lock:
                    news_data[result_data['date']][result_data['idx']]['content'] = result['content']
                    news_data[result_data['date']][result_data['idx']]['resolved_url'] = result['resolved_url']
                stats['newly_enriched'] += 1
            else:
                with file_lock:
                    news_data[result_data['date']][result_data['idx']]['fetch_error'] = result['error']
                stats['failed'] += 1

            processed += 1
            pbar.update(1)

            success_rate = stats['newly_enriched'] / (stats['newly_enriched'] + stats['failed']) * 100 if (stats['newly_enriched'] + stats['failed']) > 0 else 0
            pbar.set_postfix({
                'ok': stats['newly_enriched'],
                'fail': stats['failed'],
                'rate': f"{success_rate:.0f}%"
            })

            # Save periodically
            if processed % 20 == 0:
                with file_lock:
                    with open(news_file, 'w', encoding='utf-8') as f:
                        json.dump(news_data, f, indent=2, ensure_ascii=False)

    pbar.close()

    # Final save
    with open(news_file, 'w', encoding='utf-8') as f:
        json.dump(news_data, f, indent=2, ensure_ascii=False)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Enrich news with Selenium (Parallel)')
    parser.add_argument('--ticker', type=str, help='Process single ticker')
    parser.add_argument('--all-tickers', action='store_true', help='Process all news files')
    parser.add_argument('--exclude', type=str, help='Exclude tickers (comma-separated)')
    parser.add_argument('--limit', type=int, help='Max articles per ticker')
    parser.add_argument('--all', action='store_true', help='Process all articles')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--show-browser', action='store_true', help='Show browser')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    args = parser.parse_args()

    local_dir = get_local_data_dir()
    news_dir = os.path.join(local_dir, "news_data")

    if not os.path.exists(news_dir):
        print(f"News directory not found: {news_dir}")
        return

    excluded_tickers = set()
    if args.exclude:
        excluded_tickers = set(t.strip().upper() for t in args.exclude.split(','))

    if args.ticker:
        news_files = [os.path.join(news_dir, f"{args.ticker}_news.json")]
        if not os.path.exists(news_files[0]):
            print(f"News file not found: {news_files[0]}")
            return
    elif args.all_tickers:
        news_files = [
            os.path.join(news_dir, f)
            for f in sorted(os.listdir(news_dir))
            if f.endswith('_news.json')
        ]
        if excluded_tickers:
            news_files = [
                f for f in news_files
                if os.path.basename(f).replace('_news.json', '').upper() not in excluded_tickers
            ]
    else:
        available = [f for f in os.listdir(news_dir) if f.endswith('_news.json')]
        print("Available news files:")
        for f in available:
            print(f"  - {f}")
        print("\nUsage:")
        print("  python scripts/enrich_news_selenium.py --all-tickers --start-date 2025-10-19 --end-date 2025-11-03 --exclude AAPL,META --workers 4")
        return

    limit = None if args.all else (args.limit if args.limit else None)
    tickers = [os.path.basename(f).replace('_news.json', '') for f in news_files]

    print(f"\n{'='*60}")
    print("NEWS CONTENT ENRICHMENT (Parallel Selenium)")
    print(f"{'='*60}")
    print(f"Tickers: {', '.join(tickers)}")
    if excluded_tickers:
        print(f"Excluded: {', '.join(excluded_tickers)}")
    print(f"Date range: {args.start_date or 'Any'} ~ {args.end_date or 'Any'}")
    print(f"Limit: {'All' if limit is None else limit}")
    print(f"Workers: {args.workers}")
    print(f"{'='*60}\n")

    total_stats = {
        'total_articles': 0,
        'already_enriched': 0,
        'newly_enriched': 0,
        'failed': 0,
        'skipped': 0,
        'out_of_range': 0
    }

    for news_file in news_files:
        ticker = os.path.basename(news_file).replace('_news.json', '')

        stats = enrich_news_file(
            news_file,
            limit=limit,
            headless=not args.show_browser,
            start_date=args.start_date,
            end_date=args.end_date,
            ticker_name=ticker,
            num_workers=args.workers
        )

        for key in total_stats:
            total_stats[key] += stats[key]

        print(f"  [{ticker}] Done: {stats['newly_enriched']} ok, {stats['failed']} failed")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total in range: {total_stats['total_articles']}")
    print(f"Newly enriched: {total_stats['newly_enriched']}")
    print(f"Failed: {total_stats['failed']}")
    print(f"Already had content: {total_stats['already_enriched']}")

    if total_stats['newly_enriched'] + total_stats['failed'] > 0:
        success_rate = total_stats['newly_enriched'] / (total_stats['newly_enriched'] + total_stats['failed']) * 100
        print(f"Success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    main()
