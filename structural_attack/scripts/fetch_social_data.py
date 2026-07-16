#!/usr/bin/env python3
"""
Fetch and cache Reddit social data for BTC-USD, MSFT, NVDA, TSLA.

Source: ArcticShift public API (no auth required) — historical Reddit posts.

Subreddit mapping:
  BTC-USD  → r/Bitcoin, r/CryptoCurrency          (no keyword filter)
  MSFT     → r/stocks, r/investing, r/microsoft    (keyword: "Microsoft MSFT")
  NVDA     → r/stocks, r/investing, r/nvidia       (keyword: "Nvidia NVDA")
  TSLA     → r/stocks, r/investing, r/teslainvestorsclub  (keyword: "Tesla TSLA")

Output: local_data/social_data/{ticker}_social.json
Format: { "YYYY-MM-DD": [ {title, content, score, num_comments,
                            upvote_ratio, subreddit, url}, ... ] }

Usage:
    uv run python scripts/fetch_social_data.py --start-date 2025-01-01 --end-date 2025-03-31
    uv run python scripts/fetch_social_data.py --ticker BTC-USD --start-date 2025-01-01 --end-date 2025-03-31
    uv run python scripts/fetch_social_data.py --start-date 2025-01-01 --end-date 2025-03-31 --dry-run
"""

import os
import sys
import json
import argparse
import time
import requests
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.config import get_config

# ── constants ─────────────────────────────────────────────────────────────────
API_BASE = "https://arctic-shift.photon-reddit.com/api/posts/search"
HEADERS  = {"User-Agent": "trading-research-social/1.0"}

# posts per subreddit per day to keep (top-N by score)
TOP_N_PER_SUB  = 10
# minimum score threshold (filter noise)
MIN_SCORE      = 5
# sleep between API calls (seconds) — be polite
SLEEP_BETWEEN  = 0.5

SUBREDDIT_MAP = {
    "BTC-USD": [
        {"subreddit": "Bitcoin",        "query": None},
        {"subreddit": "CryptoCurrency", "query": None},
    ],
    "MSFT": [
        {"subreddit": "stocks",    "query": "Microsoft MSFT"},
        {"subreddit": "investing", "query": "Microsoft MSFT"},
        {"subreddit": "microsoft", "query": None},
    ],
    "NVDA": [
        {"subreddit": "stocks",    "query": "Nvidia NVDA"},
        {"subreddit": "investing", "query": "Nvidia NVDA"},
        {"subreddit": "nvidia",    "query": None},
    ],
    "TSLA": [
        {"subreddit": "stocks",             "query": "Tesla TSLA"},
        {"subreddit": "investing",          "query": "Tesla TSLA"},
        {"subreddit": "teslainvestorsclub", "query": None},
    ],
}


# ── helpers ───────────────────────────────────────────────────────────────────
def get_local_data_dir() -> str:
    config = get_config()
    base_dir = config.get("data_cache_dir", "tradingagents/dataflows/data_cache")
    return os.path.join(os.path.dirname(base_dir), "local_data")


def fetch_posts(subreddit: str, after: str, before: str, query: str | None,
                limit: int = 100) -> list[dict]:
    """
    Fetch posts from ArcticShift for a single subreddit and date window.
    Returns a list of raw post dicts (empty list on error).
    """
    params = {
        "subreddit": subreddit,
        "after":     after,
        "before":    before,
        "limit":     limit,
        "sort":      "score",
    }
    if query:
        params["q"] = query

    try:
        r = requests.get(API_BASE, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json().get("data") or []
    except Exception as e:
        tqdm.write(f"    [WARN] {subreddit} {after}: {e}")
        return []


def parse_post(raw: dict) -> dict | None:
    """Extract relevant fields from a raw ArcticShift post dict."""
    score = raw.get("score", 0) or 0
    if score < MIN_SCORE:
        return None

    ts = raw.get("created_utc", 0) or 0
    try:
        date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return None

    selftext = raw.get("selftext", "") or ""
    # Remove deleted/removed placeholders
    if selftext in ("[deleted]", "[removed]"):
        selftext = ""

    return {
        "date":         date_str,
        "title":        raw.get("title", ""),
        "content":      selftext[:1000],   # cap at 1000 chars
        "score":        score,
        "num_comments": raw.get("num_comments", 0) or 0,
        "upvote_ratio": raw.get("upvote_ratio"),
        "subreddit":    raw.get("subreddit", ""),
        "url":          raw.get("url", ""),
        "fetch_source": "reddit_arcticshift",
    }


def daterange(start: str, end: str) -> list[str]:
    """Yield YYYY-MM-DD strings from start to end inclusive."""
    cur = datetime.strptime(start, "%Y-%m-%d")
    fin = datetime.strptime(end,   "%Y-%m-%d")
    days = []
    while cur <= fin:
        days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return days


# ── core fetch logic ───────────────────────────────────────────────────────────
def fetch_ticker_social(ticker: str, start_date: str, end_date: str,
                        verbose: bool = True) -> dict[str, list]:
    """
    Fetch all Reddit social data for a ticker across its subreddits.
    Returns {date_str: [post, ...]} sorted by score desc.
    """
    sources = SUBREDDIT_MAP.get(ticker)
    if not sources:
        print(f"[WARN] No subreddit mapping for {ticker}")
        return {}

    all_posts: dict[str, list] = {}  # date → list of posts

    days = daterange(start_date, end_date)

    for src in sources:
        sub    = src["subreddit"]
        query  = src["query"]
        label  = f"r/{sub}" + (f" q={query!r}" if query else "")

        if verbose:
            tqdm.write(f"  Fetching {label} ({start_date} ~ {end_date})")

        # Fetch in weekly windows to avoid missing data
        cur = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date,   "%Y-%m-%d")

        while cur <= end:
            window_end = min(cur + timedelta(days=6), end)
            after  = cur.strftime("%Y-%m-%d")
            before = window_end.strftime("%Y-%m-%d")

            raw_posts = fetch_posts(sub, after, before, query)
            time.sleep(SLEEP_BETWEEN)

            for raw in raw_posts:
                post = parse_post(raw)
                if post is None:
                    continue
                d = post["date"]
                if d < start_date or d > end_date:
                    continue
                all_posts.setdefault(d, []).append(post)

            cur = window_end + timedelta(days=1)

    # Deduplicate (same URL) and keep top-N per day by score
    result: dict[str, list] = {}
    for d in days:
        posts = all_posts.get(d, [])
        # dedup by url
        seen_urls: set[str] = set()
        deduped = []
        for p in posts:
            u = p.get("url", "")
            if u not in seen_urls:
                seen_urls.add(u)
                deduped.append(p)
        # sort by score desc, keep top-N
        deduped.sort(key=lambda x: x["score"], reverse=True)
        result[d] = deduped[:TOP_N_PER_SUB * len(sources)]

    return result


# ── save / load ────────────────────────────────────────────────────────────────
def load_existing(filepath: str) -> dict:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def merge_social(existing: dict, incoming: dict) -> dict:
    """Merge incoming into existing; skip exact URL duplicates."""
    merged = {k: list(v) for k, v in existing.items()}
    for date_key, posts in incoming.items():
        existing_urls = {p.get("url", "") for p in merged.get(date_key, [])}
        for p in posts:
            if p.get("url", "") not in existing_urls:
                merged.setdefault(date_key, []).append(p)
                existing_urls.add(p.get("url", ""))
    return merged


def save_social(data: dict, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_stats(data: dict, label: str = "") -> None:
    if not data:
        print(f"  {label}(no data)")
        return
    dates  = sorted(data.keys())
    total  = sum(len(v) for v in data.values())
    filled = sum(1 for v in data.values() if v)
    print(f"  {label}{dates[0]} ~ {dates[-1]}  |  {filled}/{len(dates)} days  |  {total} posts")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch Reddit social data")
    parser.add_argument("--ticker",     type=str,
                        help="Single ticker (BTC-USD/MSFT/NVDA/TSLA). Default: all.")
    parser.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date",   type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Show plan without fetching")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else list(SUBREDDIT_MAP.keys())

    local_dir   = get_local_data_dir()
    social_dir  = os.path.join(local_dir, "social_data")

    print("=" * 60)
    print("Reddit Social Data Fetcher")
    print("=" * 60)
    print(f"Period  : {args.start_date} ~ {args.end_date}")
    print(f"Tickers : {', '.join(tickers)}")
    print(f"Output  : {social_dir}")
    print()

    for ticker in tickers:
        sources = SUBREDDIT_MAP.get(ticker, [])
        print(f"  {ticker}: {[s['subreddit'] for s in sources]}")
    print()

    if args.dry_run:
        print("[dry-run] Exiting without fetching.")
        return

    for ticker in tickers:
        output_file = os.path.join(social_dir, f"{ticker}_social.json")
        print(f"{'='*60}")
        print(f"[{ticker}]")
        existing = load_existing(output_file)
        print(f"  Existing: ", end="")
        print_stats(existing)

        new_data = fetch_ticker_social(ticker, args.start_date, args.end_date)
        merged   = merge_social(existing, new_data)
        save_social(merged, output_file)

        print(f"  Saved  : ", end="")
        print_stats(merged)
        print()

    print("=" * 60)
    print("DONE")
    print("=" * 60)
    for ticker in tickers:
        output_file = os.path.join(social_dir, f"{ticker}_social.json")
        print(f"  {ticker} → {output_file}")


if __name__ == "__main__":
    main()
