#!/usr/bin/env python3
"""
Fetch all data required for 2026-01-01 ~ 2026-03-31 experiments.

Collects:
  1. Market data       (yfinance)
  2. Technical indicators (stockstats from market data)
  3. News              (Alpha Vantage NEWS_SENTIMENT, needs ALPHA_VANTAGE_API_KEY)
  4. Social media      (Reddit public JSON API, no key needed)

Usage:
    cd multi_agents_vulnerablilty
    python fetch_2026_data.py

Environment:
    ALPHA_VANTAGE_API_KEY  — required for news
    (loaded from .env automatically)
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

# ─── Load .env ────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ─── Config ───────────────────────────────────────────────────
TICKERS     = ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]
START_DATE  = "2026-01-01"
END_DATE    = "2026-03-31"
SOCIAL_DAYS = 110  # from today back to ~Jan 1 2026

BASE_DIR  = Path(__file__).parent / "tradingagents" / "dataflows" / "local_data"
PRICE_DIR = BASE_DIR / "market_data" / "price_data"
IND_DIR   = BASE_DIR / "indicators"
NEWS_DIR  = BASE_DIR / "news_data" / "alpha_vantage"
SOC_DIR   = BASE_DIR / "social_data"

INDICATORS = [
    "close_50_sma", "close_200_sma", "close_10_ema",
    "macd", "macds", "macdh",
    "rsi",
    "boll", "boll_ub", "boll_lb",
    "atr", "vwma", "mfi",
]

# Reddit subreddit / keyword mapping
TICKER_CONFIG = {
    "TSLA":    {"subreddits": ["wallstreetbets", "stocks", "investing", "teslamotors", "StockMarket"], "keywords": ["TSLA", "Tesla"]},
    "NVDA":    {"subreddits": ["wallstreetbets", "stocks", "investing", "nvidia", "StockMarket"],      "keywords": ["NVDA", "Nvidia"]},
    "MSFT":    {"subreddits": ["wallstreetbets", "stocks", "investing", "microsoft", "StockMarket"],   "keywords": ["MSFT", "Microsoft"]},
    "BTC-USD": {"subreddits": ["CryptoCurrency", "Bitcoin", "wallstreetbets", "CryptoMarkets"],        "keywords": ["BTC", "Bitcoin"]},
    "AAPL":    {"subreddits": ["wallstreetbets", "stocks", "investing", "apple", "StockMarket"],        "keywords": ["AAPL", "Apple"]},
}

REDDIT_HEADERS = {"User-Agent": "financial-social-collector/1.0 (academic research)"}
REDDIT_DELAY   = 2.0  # seconds between Reddit requests


# ══════════════════════════════════════════════════════════════
# 1. Market Data
# ══════════════════════════════════════════════════════════════

def fetch_market_data(ticker: str) -> bool:
    print(f"  [Market] {ticker} ...")
    try:
        import yfinance as yf
        import pandas as pd
        from dateutil.relativedelta import relativedelta

        # Extend 1 year back for indicator warm-up
        ext_start = (datetime.strptime(START_DATE, "%Y-%m-%d") - relativedelta(years=1)).strftime("%Y-%m-%d")
        ext_end   = (datetime.strptime(END_DATE,   "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

        data = yf.download(ticker, start=ext_start, end=ext_end, progress=False, auto_adjust=True)
        if data.empty:
            print(f"    WARN: no data for {ticker}")
            return False

        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        PRICE_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{ticker}-YFin-data-{ext_start}-{END_DATE}.csv"
        data.to_csv(PRICE_DIR / fname, index=False)
        print(f"    ✓ {len(data)} rows → {fname}")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# 2. Technical Indicators
# ══════════════════════════════════════════════════════════════

def fetch_indicators(ticker: str) -> bool:
    print(f"  [Indicators] {ticker} ...")
    try:
        import pandas as pd
        from stockstats import wrap

        from dateutil.relativedelta import relativedelta
        ext_start = (datetime.strptime(START_DATE, "%Y-%m-%d") - relativedelta(years=1)).strftime("%Y-%m-%d")
        price_file = PRICE_DIR / f"{ticker}-YFin-data-{ext_start}-{END_DATE}.csv"

        if not price_file.exists():
            print(f"    WARN: price file not found, skipping")
            return False

        data = pd.read_csv(price_file)
        data["Date"] = pd.to_datetime(data["Date"])
        df = wrap(data)

        for ind in INDICATORS:
            try:
                df[ind]
            except Exception:
                pass

        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        new_data: dict = {}
        for _, row in df.iterrows():
            d = row["Date"]
            if d < START_DATE or d > END_DATE:
                continue
            new_data[d] = {
                ind: (None if pd.isna(row[ind]) else float(row[ind]))
                for ind in INDICATORS if ind in df.columns
            }

        # Merge with existing JSON
        IND_DIR.mkdir(parents=True, exist_ok=True)
        ind_file = IND_DIR / f"{ticker}_indicators.json"
        existing: dict = {}
        if ind_file.exists():
            with open(ind_file) as f:
                existing = json.load(f)
        existing.update(new_data)

        with open(ind_file, "w") as f:
            json.dump(existing, f, indent=2)

        print(f"    ✓ {len(new_data)} new days merged → {ind_file.name}")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# 3. News (Alpha Vantage)
# ══════════════════════════════════════════════════════════════

def _av_news_request(ticker: str, time_from: str, time_to: str) -> list[dict]:
    """Single Alpha Vantage NEWS_SENTIMENT call. Returns list of article dicts."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "time_to": time_to,
        "sort": "LATEST",
        "limit": "1000",
        "apikey": api_key,
    }
    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "Information" in data:
        raise RuntimeError(f"AV rate limit: {data['Information']}")

    return data.get("feed", [])


def _format_av_date(dt_str: str) -> str:
    """YYYYMMDDTHHMM → YYYY-MM-DD"""
    try:
        return datetime.strptime(dt_str[:8], "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        return dt_str[:10]


def fetch_news(ticker: str) -> bool:
    print(f"  [News] {ticker} ...")
    try:
        # AV date format: YYYYMMDDTHHMM
        time_from = datetime.strptime(START_DATE, "%Y-%m-%d").strftime("%Y%m%dT0000")
        time_to   = datetime.strptime(END_DATE,   "%Y-%m-%d").strftime("%Y%m%dT2359")

        articles = _av_news_request(ticker, time_from, time_to)
        if not articles:
            print(f"    WARN: no articles returned")
            return False

        new_by_date: dict[str, list] = defaultdict(list)
        for art in articles:
            date_str = _format_av_date(art.get("time_published", ""))
            if not (START_DATE <= date_str <= END_DATE):
                continue

            # Extract ticker-specific sentiment
            ts = next(
                (t for t in art.get("ticker_sentiment", []) if t.get("ticker") == ticker),
                {}
            )
            new_by_date[date_str].append({
                "headline":                art.get("title", ""),
                "summary":                 art.get("summary", ""),
                "source":                  art.get("source", ""),
                "link":                    art.get("url", ""),
                "content":                 art.get("summary", ""),
                "authors":                 art.get("authors", []),
                "overall_sentiment_score": art.get("overall_sentiment_score"),
                "overall_sentiment_label": art.get("overall_sentiment_label", ""),
                "ticker_sentiment_score":  ts.get("ticker_sentiment_score", ""),
                "ticker_sentiment_label":  ts.get("ticker_sentiment_label", ""),
                "relevance_score":         ts.get("relevance_score", ""),
                "fetch_source":            "alpha_vantage",
            })

        # Merge with existing JSON
        NEWS_DIR.mkdir(parents=True, exist_ok=True)
        news_file = NEWS_DIR / f"{ticker}_news.json"
        existing: dict = {}
        if news_file.exists():
            with open(news_file) as f:
                existing = json.load(f)
        existing.update(new_by_date)

        with open(news_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        total = sum(len(v) for v in new_by_date.values())
        print(f"    ✓ {total} articles ({len(new_by_date)} days) merged → {news_file.name}")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# 4. Social Data (Reddit public JSON)
# ══════════════════════════════════════════════════════════════

def _reddit_search(subreddit: str, query: str, after: str | None) -> tuple[list[dict], str | None]:
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params: dict = {"q": query, "sort": "new", "t": "year", "limit": 100, "restrict_sr": 1}
    if after:
        params["after"] = after
    try:
        resp = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=30)
        if resp.status_code == 429:
            print(f"      [RATE LIMIT] r/{subreddit} — sleeping 60s")
            time.sleep(60)
            return [], None
        if resp.status_code in (403, 404):
            return [], None
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"      WARN r/{subreddit}: {exc}")
        return [], None

    children  = data.get("data", {}).get("children", [])
    next_after = data.get("data", {}).get("after")
    posts = [c["data"] for c in children if c.get("kind") == "t3"]
    return posts, next_after


def fetch_social(ticker: str) -> bool:
    print(f"  [Social] {ticker} ...")
    cfg = TICKER_CONFIG.get(ticker)
    if not cfg:
        print(f"    WARN: no config for {ticker}")
        return False

    end_dt   = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=SOCIAL_DAYS)

    query = " OR ".join(cfg["keywords"])
    results: list[dict] = []
    seen: set[str] = set()

    for sub in cfg["subreddits"]:
        after_token = None
        for _ in range(10):  # max 10 pages per subreddit
            posts, after_token = _reddit_search(sub, query, after_token)
            time.sleep(REDDIT_DELAY)
            stop = False
            for post in posts:
                pid = post.get("id", "")
                if pid in seen:
                    continue
                created = datetime.fromtimestamp(float(post.get("created_utc", 0)), tz=timezone.utc)
                if created < start_dt:
                    stop = True
                    break
                if created > end_dt:
                    continue
                title = (post.get("title") or "").strip()
                body  = (post.get("selftext") or "").strip()
                if body.lower() in {"[removed]", "[deleted]"}:
                    body = ""
                if not title and not body:
                    continue
                seen.add(pid)
                results.append({
                    "platform":     "reddit",
                    "title":        title,
                    "text":         body,
                    "author":       post.get("author"),
                    "score":        post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "url":          f"https://reddit.com{post.get('permalink', '')}",
                    "subreddit":    sub,
                    "created_at":   created.strftime("%Y-%m-%dT%H:%M:%S"),
                })
            if stop or not after_token:
                break

    if not results:
        print(f"    WARN: no posts collected")
        return False

    # Group by date
    by_date: dict[str, list] = defaultdict(list)
    for p in results:
        by_date[p["created_at"][:10]].append(p)

    # Filter to experiment window only
    by_date = {d: v for d, v in by_date.items() if START_DATE <= d <= END_DATE}

    if not by_date:
        print(f"    WARN: no posts in {START_DATE}~{END_DATE} window (collected {len(results)} total)")
        return False

    # Merge with existing
    SOC_DIR.mkdir(parents=True, exist_ok=True)
    soc_file = SOC_DIR / f"{ticker}_social.json"
    existing: dict = {}
    if soc_file.exists():
        with open(soc_file) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
    for d, posts in by_date.items():
        if d in existing:
            ex_urls = {p.get("url") for p in existing[d]}
            existing[d].extend(p for p in posts if p.get("url") not in ex_urls)
        else:
            existing[d] = posts

    with open(soc_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in by_date.values())
    print(f"    ✓ {total} posts in window ({len(results)} total scraped) → {soc_file.name}")
    return True


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print(f"Fetching 2026 data: {START_DATE} ~ {END_DATE}")
    print(f"Tickers: {TICKERS}")
    print("=" * 60)

    summary: dict[str, dict[str, bool]] = {t: {} for t in TICKERS}

    for ticker in TICKERS:
        print(f"\n── {ticker} ──")

        summary[ticker]["market"]     = fetch_market_data(ticker)
        summary[ticker]["indicators"] = fetch_indicators(ticker)

        # Alpha Vantage: 5 req/min limit — add delay between tickers
        summary[ticker]["news"] = fetch_news(ticker)
        time.sleep(13)  # stay within 5 req/min

        summary[ticker]["social"] = fetch_social(ticker)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ticker, res in summary.items():
        parts = "  ".join(
            f"{k}:{'✓' if v else '✗'}" for k, v in res.items()
        )
        print(f"  {ticker:10s}  {parts}")
    print("\nDone!")


if __name__ == "__main__":
    main()
