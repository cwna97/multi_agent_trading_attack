#!/usr/bin/env python3
"""Collect Reddit (and optionally X) social data for given tickers
and save to local_data/social_data/{TICKER}_social.json.

Reddit collection uses the public JSON API (no API key required).
X collection requires X_BEARER_TOKEN in .env.

Usage:
    python collect_social_data.py [--tickers TSLA NVDA ...] [--days 90] [--limit 500]

Optional .env (only needed for X):
    X_BEARER_TOKEN=<token>
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

import requests

SOCIAL_COLLECTOR_SRC = (
    Path(__file__).parent.parent.parent
    / "social_data"
    / "financial-social-collector"
    / "src"
)
sys.path.insert(0, str(SOCIAL_COLLECTOR_SRC))

DEFAULT_OUTPUT_DIR = (
    Path(__file__).parent
    / "tradingagents"
    / "dataflows"
    / "local_data"
    / "social_data"
)

TICKER_CONFIG: dict[str, dict] = {
    "TSLA": {
        "subreddits": ["wallstreetbets", "stocks", "investing", "teslamotors", "StockMarket"],
        "keywords": ["TSLA", "Tesla"],
    },
    "NVDA": {
        "subreddits": ["wallstreetbets", "stocks", "investing", "nvidia", "StockMarket"],
        "keywords": ["NVDA", "Nvidia"],
    },
    "MSFT": {
        "subreddits": ["wallstreetbets", "stocks", "investing", "microsoft", "StockMarket"],
        "keywords": ["MSFT", "Microsoft"],
    },
    "AAPL": {
        "subreddits": ["wallstreetbets", "stocks", "investing", "apple", "StockMarket"],
        "keywords": ["AAPL", "Apple"],
    },
    "AMZN": {
        "subreddits": ["wallstreetbets", "stocks", "investing", "amazon", "StockMarket"],
        "keywords": ["AMZN", "Amazon"],
    },
    "GOOGL": {
        "subreddits": ["wallstreetbets", "stocks", "investing", "google", "StockMarket"],
        "keywords": ["GOOGL", "GOOG", "Google", "Alphabet"],
    },
    "META": {
        "subreddits": ["wallstreetbets", "stocks", "investing", "facebook", "StockMarket"],
        "keywords": ["META", "Meta", "Facebook"],
    },
    "PLTR": {
        "subreddits": ["wallstreetbets", "stocks", "investing", "palantir", "StockMarket"],
        "keywords": ["PLTR", "Palantir"],
    },
    "BTC-USD": {
        "subreddits": ["CryptoCurrency", "Bitcoin", "wallstreetbets", "CryptoMarkets"],
        "keywords": ["BTC", "Bitcoin"],
    },
}

REDDIT_HEADERS = {
    "User-Agent": "financial-social-collector/1.0 (academic research)"
}
REDDIT_REQUEST_DELAY = 2.0  # seconds between requests to avoid rate limiting


def _reddit_search(
    subreddit: str,
    query: str,
    after_token: str | None,
    time_filter: str = "year",
) -> tuple[list[dict], str | None]:
    """Call Reddit search.json and return (posts, next_after_token)."""
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params: dict = {
        "q": query,
        "sort": "new",
        "t": time_filter,
        "limit": 100,
        "restrict_sr": 1,
    }
    if after_token:
        params["after"] = after_token

    try:
        resp = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=30)
        if resp.status_code == 429:
            print(f"    [RATE LIMIT] r/{subreddit} — sleeping 60s")
            time.sleep(60)
            return [], None
        if resp.status_code == 403:
            print(f"    [SKIP] r/{subreddit} is private or restricted")
            return [], None
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"    [WARN] r/{subreddit} request failed: {exc}")
        return [], None

    children = data.get("data", {}).get("children", [])
    next_after = data.get("data", {}).get("after")
    posts = [c["data"] for c in children if c.get("kind") == "t3"]
    return posts, next_after


def collect_reddit_for_ticker(
    ticker: str,
    start_dt: datetime,
    end_dt: datetime,
    limit: int,
) -> list[dict]:
    """Collect Reddit posts via the public JSON API (no credentials needed)."""
    config = TICKER_CONFIG.get(ticker)
    if not config:
        print(f"  [WARN] No config for {ticker}, skipping Reddit")
        return []

    query = " OR ".join(config["keywords"])
    results: list[dict] = []
    seen_ids: set[str] = set()

    for subreddit_name in config["subreddits"]:
        after_token: str | None = None
        page = 0
        stop_subreddit = False

        while not stop_subreddit:
            posts, after_token = _reddit_search(subreddit_name, query, after_token)
            time.sleep(REDDIT_REQUEST_DELAY)

            if not posts:
                break

            for post in posts:
                post_id = post.get("id", "")
                if post_id in seen_ids:
                    continue

                created_utc = datetime.fromtimestamp(
                    float(post.get("created_utc", 0)), tz=timezone.utc
                )

                # Posts are sorted newest-first; stop when we pass start_dt
                if created_utc < start_dt:
                    stop_subreddit = True
                    break
                if created_utc > end_dt:
                    continue

                title = (post.get("title") or "").strip()
                body = (post.get("selftext") or "").strip()
                if body.lower() in {"[removed]", "[deleted]"}:
                    body = ""
                if not title and not body:
                    continue

                seen_ids.add(post_id)
                results.append({
                    "platform": "reddit",
                    "title": title,
                    "text": body,
                    "author": post.get("author"),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "subreddit": subreddit_name,
                    "created_at": created_utc.strftime("%Y-%m-%dT%H:%M:%S"),
                })

                if len(results) >= limit:
                    return results

            page += 1
            if not after_token or page >= 10:
                break

    return results


def collect_x_for_ticker(
    ticker: str,
    start_dt: datetime,
    end_dt: datetime,
    limit: int,
    bearer_token: str,
) -> list[dict]:
    """Collect X posts via the official recent search API."""
    config = TICKER_CONFIG.get(ticker)
    if not config:
        return []

    try:
        from financial_social_collector.collectors.x_collector import XCollector
        from financial_social_collector.config import Settings
    except ImportError as exc:
        print(f"  [WARN] financial-social-collector not importable: {exc}")
        return []

    cashtag_keywords = [
        f"${kw}" if not kw.startswith("$") else kw for kw in config["keywords"]
    ]
    query = " OR ".join(cashtag_keywords) + " -is:retweet lang:en"

    settings = Settings(x_bearer_token=bearer_token)
    try:
        collector = XCollector(
            asset=ticker,
            query=query,
            start_time=start_dt,
            end_time=end_dt,
            max_results=limit,
            settings=settings,
        )
        social_posts = collector.collect()
        return [
            {
                "platform": "x",
                "title": None,
                "text": post.text,
                "author": post.author,
                "score": post.engagement.score,
                "num_comments": post.engagement.replies,
                "url": post.meta.get("url"),
                "subreddit": None,
                "created_at": post.created_at[:19],
            }
            for post in social_posts
        ]
    except Exception as exc:
        print(f"  [WARN] X collection failed for {ticker}: {exc}")
        return []


def group_posts_by_date(posts: list[dict]) -> dict[str, list[dict]]:
    by_date: dict[str, list[dict]] = defaultdict(list)
    for post in posts:
        date_str = post["created_at"][:10]
        by_date[date_str].append(post)
    return dict(by_date)


def merge_with_existing(
    output_file: Path, new_by_date: dict[str, list[dict]]
) -> dict[str, list[dict]]:
    if not output_file.exists():
        return new_by_date

    with open(output_file, "r", encoding="utf-8") as f:
        existing: dict[str, list[dict]] = json.load(f)

    merged = dict(existing)
    for date_str, posts in new_by_date.items():
        if date_str in merged:
            existing_urls = {p.get("url") for p in merged[date_str]}
            merged[date_str].extend(p for p in posts if p.get("url") not in existing_urls)
        else:
            merged[date_str] = posts
    return merged


def collect_ticker(
    ticker: str,
    start_dt: datetime,
    end_dt: datetime,
    limit: int,
    output_dir: Path,
    x_bearer_token: str | None,
) -> None:
    print(f"\n[{ticker}] {start_dt.date()} ~ {end_dt.date()} ...")

    reddit_posts = collect_reddit_for_ticker(ticker, start_dt, end_dt, limit)
    x_posts = (
        collect_x_for_ticker(ticker, start_dt, end_dt, limit, x_bearer_token)
        if x_bearer_token
        else []
    )

    all_posts = reddit_posts + x_posts
    print(f"  Reddit: {len(reddit_posts)}  X: {len(x_posts)}")

    if not all_posts:
        print(f"  [WARN] No posts collected for {ticker}")
        return

    by_date = group_posts_by_date(all_posts)
    output_file = output_dir / f"{ticker}_social.json"
    merged = merge_with_existing(output_file, by_date)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in merged.values())
    print(f"  Saved {total} total posts -> {output_file}")


def load_x_token(env_file: str | None) -> str | None:
    """Load X_BEARER_TOKEN from environment or .env file."""
    if env_file and Path(env_file).exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("X_BEARER_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    return token if token else None
    return os.getenv("X_BEARER_TOKEN") or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect social media data for trading tickers")
    parser.add_argument(
        "--tickers", nargs="+", default=list(TICKER_CONFIG.keys()),
        help="Tickers to collect (default: all)",
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Number of past days to collect (default: 90)",
    )
    parser.add_argument(
        "--limit", type=int, default=500,
        help="Max posts per ticker (default: 500)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for social JSON files",
    )
    parser.add_argument(
        "--env-file", type=str,
        default=str(SOCIAL_COLLECTOR_SRC.parent / ".env"),
        help="Path to .env file for X_BEARER_TOKEN",
    )
    args = parser.parse_args()

    x_bearer_token = load_x_token(args.env_file)
    if x_bearer_token:
        print("X Bearer Token: found — X collection enabled")
    else:
        print("X Bearer Token: not found — Reddit only")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)

    print(f"Tickers : {args.tickers}")
    print(f"Range   : {start_dt.date()} ~ {end_dt.date()}")
    print(f"Output  : {output_dir}")

    for ticker in args.tickers:
        collect_ticker(ticker, start_dt, end_dt, args.limit, output_dir, x_bearer_token)

    print("\nDone!")


if __name__ == "__main__":
    main()
