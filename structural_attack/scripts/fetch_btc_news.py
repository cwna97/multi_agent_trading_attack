#!/usr/bin/env python3
"""
Fetch and cache BTC-USD news data.

Two sources:
  1. Alpha Vantage NEWS_SENTIMENT (CRYPTO:BTC) — primary, includes sentiment scores
  2. Google News (gnews) — supplement for days with no Alpha Vantage data

Output: local_data/news_data/alpha_vantage/BTC-USD_news.json
        (same format as MSFT_news.json etc., so get_local_news works without changes)

Usage:
    # Fetch Q1 2025 (experiment default period) — ~13 weekly calls
    uv run python scripts/fetch_btc_news.py --start-date 2025-01-01 --end-date 2025-03-31

    # Fetch full year weekly (~52 calls, needs 3 days on free tier)
    uv run python scripts/fetch_btc_news.py --year 2025

    # Fetch full year, one specific quarter
    uv run python scripts/fetch_btc_news.py --year 2025 --quarter 1

    # Dry-run: show plan without fetching
    uv run python scripts/fetch_btc_news.py --year 2025 --dry-run

    # Skip Alpha Vantage, use only gnews
    uv run python scripts/fetch_btc_news.py --start-date 2025-01-01 --end-date 2025-03-31 --gnews-only

Alpha Vantage free tier: 25 requests/day.
BTC has very high news volume — quarterly window saturates the 1000-article limit,
so we use weekly windows (7-day chunks) for complete daily coverage.
Q1 = ~13 calls, full year = ~52 calls (3 days on free tier).
"""

import os
import sys
import json
import argparse
import requests
import time
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.googlenews_utils import getNewsData

# ── constants ─────────────────────────────────────────────────────────────────
AV_BASE_URL    = "https://www.alphavantage.co/query"
AV_TICKER      = "CRYPTO:BTC"   # Alpha Vantage crypto ticker symbol
OUTPUT_TICKER  = "BTC-USD"      # filename / display name

GNEWS_QUERY    = "Bitcoin BTC cryptocurrency"


# ── helpers ───────────────────────────────────────────────────────────────────
def get_local_data_dir() -> str:
    config = get_config()
    base_dir = config.get("data_cache_dir", "tradingagents/dataflows/data_cache")
    return os.path.join(os.path.dirname(base_dir), "local_data")


def get_api_key() -> str:
    key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set in environment / .env")
    return key


def date_to_av_format(date_str: str, end_of_day: bool = False) -> str:
    """YYYY-MM-DD → YYYYMMDDTHHMM (Alpha Vantage format)."""
    return date_str.replace("-", "") + ("T2359" if end_of_day else "T0000")


def get_quarter_ranges(year: int) -> list[tuple[str, str, str]]:
    return [
        (f"{year}-01-01", f"{year}-03-31", "Q1"),
        (f"{year}-04-01", f"{year}-06-30", "Q2"),
        (f"{year}-07-01", f"{year}-09-30", "Q3"),
        (f"{year}-10-01", f"{year}-12-31", "Q4"),
    ]


def get_weekly_ranges(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    """Split a date range into 7-day windows for BTC's high news volume."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
    ranges   = []
    cur = start_dt
    while cur <= end_dt:
        window_end = min(cur + timedelta(days=6), end_dt)
        label = f"W{cur.strftime('%m/%d')}"
        ranges.append((cur.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d"), label))
        cur = window_end + timedelta(days=1)
    return ranges


# ── Alpha Vantage fetch ────────────────────────────────────────────────────────
def fetch_av_news(start_date: str, end_date: str, api_key: str) -> dict:
    """
    Fetch from Alpha Vantage NEWS_SENTIMENT for CRYPTO:BTC.
    Returns {date_str: [article, ...]} or empty dict on failure.
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers":  AV_TICKER,
        "time_from": date_to_av_format(start_date),
        "time_to":   date_to_av_format(end_date, end_of_day=True),
        "limit":     1000,
        "sort":      "LATEST",
        "apikey":    api_key,
    }

    try:
        resp = requests.get(AV_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [AV] Request failed: {e}")
        return {}

    # API-level errors
    if "Error Message" in data:
        print(f"  [AV] API error: {data['Error Message']}")
        return {}
    if "Note" in data:
        print(f"  [AV] Rate-limit note: {data['Note']}")
        return {}
    if "Information" in data:
        print(f"  [AV] Info: {data['Information']}")
        return {}

    news_by_date: dict[str, list] = {}
    for item in data.get("feed", []):
        time_published = item.get("time_published", "")
        try:
            date_key = datetime.strptime(time_published[:8], "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            continue

        # BTC-specific sentiment from ticker_sentiment list
        btc_sentiment = next(
            (ts for ts in item.get("ticker_sentiment", []) if ts.get("ticker") == AV_TICKER),
            None,
        )

        article = {
            "headline": item.get("title", ""),
            "summary":  item.get("summary", ""),
            "source":   item.get("source", ""),
            "link":     item.get("url", ""),
            "content":  item.get("summary", ""),
            "authors":  item.get("authors", []),
            "overall_sentiment_score": item.get("overall_sentiment_score"),
            "overall_sentiment_label": item.get("overall_sentiment_label"),
            "ticker_sentiment_score":  btc_sentiment.get("ticker_sentiment_score")  if btc_sentiment else None,
            "ticker_sentiment_label":  btc_sentiment.get("ticker_sentiment_label")  if btc_sentiment else None,
            "relevance_score":         btc_sentiment.get("relevance_score")         if btc_sentiment else None,
            "fetch_source": "alpha_vantage",
        }

        news_by_date.setdefault(date_key, []).append(article)

    return news_by_date


# ── Google News fetch ──────────────────────────────────────────────────────────
def fetch_gnews(start_date: str, end_date: str) -> dict:
    """
    Fetch Bitcoin news via gnews (Google News).
    Returns {date_str: [article, ...]} in the same schema (no sentiment scores).
    """
    print(f"  [gnews] Fetching '{GNEWS_QUERY}' from {start_date} to {end_date}...")
    try:
        articles = getNewsData(GNEWS_QUERY, start_date, end_date, fetch_full_article=False)
    except Exception as e:
        print(f"  [gnews] Failed: {e}")
        return {}

    news_by_date: dict[str, list] = {}
    for art in articles:
        date_key = art.get("date", "")
        if not date_key:
            continue
        article = {
            "headline": art.get("title", ""),
            "summary":  art.get("snippet", ""),
            "source":   art.get("source", ""),
            "link":     art.get("link", ""),
            "content":  art.get("snippet", ""),
            "authors":  [],
            "overall_sentiment_score": None,
            "overall_sentiment_label": None,
            "ticker_sentiment_score":  None,
            "ticker_sentiment_label":  None,
            "relevance_score":         None,
            "fetch_source": "gnews",
        }
        news_by_date.setdefault(date_key, []).append(article)

    return news_by_date


# ── merge helper ───────────────────────────────────────────────────────────────
def merge_news(existing: dict, incoming: dict) -> dict:
    """Merge incoming into existing; skip exact headline duplicates."""
    merged = {k: list(v) for k, v in existing.items()}
    for date_key, articles in incoming.items():
        existing_headlines = {a["headline"] for a in merged.get(date_key, [])}
        for art in articles:
            if art["headline"] not in existing_headlines:
                merged.setdefault(date_key, []).append(art)
                existing_headlines.add(art["headline"])
    return merged


# ── save / load helpers ────────────────────────────────────────────────────────
def load_existing(filepath: str) -> dict:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_news(news_by_date: dict, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(news_by_date, f, indent=2, ensure_ascii=False)


def print_summary(news_by_date: dict, label: str = "") -> None:
    if not news_by_date:
        print(f"  {label}(no data)")
        return
    dates  = sorted(news_by_date.keys())
    total  = sum(len(v) for v in news_by_date.values())
    av_cnt = sum(1 for v in news_by_date.values() for a in v if a.get("fetch_source") == "alpha_vantage")
    gn_cnt = total - av_cnt
    print(f"  {label}{dates[0]} ~ {dates[-1]}  |  {len(dates)} days  |  {total} articles  "
          f"(AV: {av_cnt}, gnews: {gn_cnt})")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch BTC-USD news data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year",       type=int, help="Fetch full year (4 quarterly AV calls)")
    group.add_argument("--start-date", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date",   type=str, help="End date YYYY-MM-DD (required with --start-date)")
    parser.add_argument("--quarter",    type=int, choices=[1,2,3,4], help="Specific quarter (use with --year)")
    parser.add_argument("--gnews-only", action="store_true", help="Skip Alpha Vantage, use gnews only")
    parser.add_argument("--supplement-gnews", action="store_true",
                        help="Also run gnews for days missing in AV data")
    parser.add_argument("--dry-run",    action="store_true", help="Show plan without fetching")
    args = parser.parse_args()

    # Resolve fetch tasks: list of (start_date, end_date, label)
    if args.year:
        quarters = get_quarter_ranges(args.year)
        if args.quarter:
            quarters = [quarters[args.quarter - 1]]
        # Flatten each quarter into weekly windows
        tasks = []
        for q_start, q_end, _ in quarters:
            tasks.extend(get_weekly_ranges(q_start, q_end))
    else:
        if not args.end_date:
            parser.error("--end-date is required when using --start-date")
        tasks = get_weekly_ranges(args.start_date, args.end_date)

    # Output path
    local_dir  = get_local_data_dir()
    output_dir = os.path.join(local_dir, "news_data", "alpha_vantage")
    output_file = os.path.join(output_dir, f"{OUTPUT_TICKER}_news.json")

    print("=" * 60)
    print("BTC-USD News Fetcher")
    print("=" * 60)
    print(f"AV ticker  : {AV_TICKER}")
    print(f"Output     : {output_file}")
    print(f"Tasks      : {len(tasks)}")
    for s, e, lbl in tasks:
        print(f"  • {lbl:5s}  {s} ~ {e}")
    print()

    if args.dry_run:
        print("[dry-run] Exiting without fetching.")
        return

    # Load any existing cached data
    accumulated = load_existing(output_file)
    print(f"Existing data: ", end="")
    print_summary(accumulated)
    print()

    if not args.gnews_only:
        try:
            api_key = get_api_key()
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        for i, (start, end, label) in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] Alpha Vantage  {label}  ({start} ~ {end})")
            av_data = fetch_av_news(start, end, api_key)
            if av_data:
                count = sum(len(v) for v in av_data.values())
                print(f"  ✓ {count} articles across {len(av_data)} days")
                accumulated = merge_news(accumulated, av_data)
                save_news(accumulated, output_file)
            else:
                print(f"  ✗ No data returned")

            if i < len(tasks):
                print("  Sleeping 12 s (rate limit)...")
                time.sleep(12)

        print()

    # Gnews supplement: fill days that have no AV data
    if args.gnews_only or args.supplement_gnews:
        full_start = tasks[0][0]
        full_end   = tasks[-1][1]

        if args.supplement_gnews and accumulated:
            # Find date gaps in the accumulated data within the range
            start_dt = datetime.strptime(full_start, "%Y-%m-%d")
            end_dt   = datetime.strptime(full_end,   "%Y-%m-%d")
            missing  = []
            cur = start_dt
            while cur <= end_dt:
                ds = cur.strftime("%Y-%m-%d")
                if ds not in accumulated:
                    missing.append(ds)
                cur += timedelta(days=1)

            if missing:
                print(f"[gnews] Supplementing {len(missing)} missing days...")
                gn_data = fetch_gnews(missing[0], missing[-1])
                # Keep only dates that were actually missing
                gn_data = {d: v for d, v in gn_data.items() if d in set(missing)}
                accumulated = merge_news(accumulated, gn_data)
                save_news(accumulated, output_file)
            else:
                print("[gnews] No missing days — skipping supplement.")
        else:
            # gnews-only: fetch the full range
            print(f"[gnews] Fetching full range {full_start} ~ {full_end}...")
            gn_data = fetch_gnews(full_start, full_end)
            accumulated = merge_news(accumulated, gn_data)
            save_news(accumulated, output_file)

    # Final summary
    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Saved → {output_file}")
    print_summary(accumulated, "Final: ")


if __name__ == "__main__":
    main()
