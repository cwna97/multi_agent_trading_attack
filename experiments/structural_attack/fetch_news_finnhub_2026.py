#!/usr/bin/env python3
"""
Finnhub 뉴스 수집 — 2026-01-01 ~ 2026-03-31

전략:
  - 주 단위(7일) 분할 → ~13 windows × 5 티커 = ~65 calls
  - 월 단위로 하면 Finnhub가 최근 N개만 반환해 말미 며칠만 채워짐
  - 주 단위로 쪼개면 매 구간 최신 기사가 그 주 초까지 커버됨
  - Finnhub 무료: 60 calls/분, 일 제한 없음 → 전체 ~1분 완료
  - Alpha Vantage와 동일 포맷으로 저장 (merge 가능)
  - 상태 파일(.finnhub_fetch_state.json)로 재실행 가능

Usage:
    .venv/bin/python3 fetch_news_finnhub_2026.py
    .venv/bin/python3 fetch_news_finnhub_2026.py --status
    .venv/bin/python3 fetch_news_finnhub_2026.py --reset   # 상태 초기화 후 재수집
"""

import os
import json
import time
import argparse
import requests
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
START_DATE = "2026-01-01"
END_DATE   = "2026-03-31"
TICKERS    = ["MSFT", "NVDA", "TSLA", "AAPL", "BTC-USD"]

# Finnhub symbol 매핑
FINNHUB_SYMBOL = {
    "MSFT":    "MSFT",
    "NVDA":    "NVDA",
    "TSLA":    "TSLA",
    "AAPL":    "AAPL",
    "BTC-USD": "BINANCE:BTCUSDT",  # 실패 시 crypto 카테고리 뉴스로 fallback
}

CALL_DELAY = 1.1   # 초 (60 calls/min 안전 마진)

BASE_DIR   = Path(__file__).parent
NEWS_DIR   = BASE_DIR / "tradingagents" / "dataflows" / "local_data" / "news_data" / "alpha_vantage"
STATE_FILE = BASE_DIR / ".finnhub_fetch_state.json"


# ── 구간 생성 (주 단위) ───────────────────────────────────────────────────────
def generate_weekly_windows(start: str, end: str, window_days: int = 7) -> list[tuple[str, str]]:
    windows = []
    cur = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    while cur <= end_dt:
        w_end = min(cur + timedelta(days=window_days - 1), end_dt)
        windows.append((cur.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))
        cur = w_end + timedelta(days=1)
    return windows


WINDOWS = generate_weekly_windows(START_DATE, END_DATE)


# ── 상태 관리 ─────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"done": []}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def state_key(ticker: str, w_start: str, w_end: str) -> str:
    return f"{ticker}|{w_start}|{w_end}"


# ── Finnhub 요청 ──────────────────────────────────────────────────────────────
def finnhub_company_news(symbol: str, from_date: str, to_date: str) -> list[dict]:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY not set in .env")
    resp = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={"symbol": symbol, "from": from_date, "to": to_date, "token": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Finnhub error: {data['error']}")
    return data if isinstance(data, list) else []


def finnhub_crypto_news(from_ts: int, to_ts: int) -> list[dict]:
    """BTC fallback: 일반 crypto 카테고리 뉴스"""
    api_key = os.getenv("FINNHUB_API_KEY")
    resp = requests.get(
        "https://finnhub.io/api/v1/news",
        params={"category": "crypto", "minId": 0, "token": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return []
    return [a for a in data if from_ts <= a.get("datetime", 0) <= to_ts]


# ── 파일 merge ────────────────────────────────────────────────────────────────
def merge_into_file(ticker: str, new_by_date: dict) -> int:
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    news_file = NEWS_DIR / f"{ticker}_news.json"
    existing: dict = {}
    if news_file.exists():
        with open(news_file, encoding="utf-8") as f:
            existing = json.load(f)
    added = 0
    for date_str, arts in new_by_date.items():
        if date_str in existing:
            ex_links = {a.get("link") for a in existing[date_str]}
            new_arts = [a for a in arts if a.get("link") not in ex_links]
            existing[date_str].extend(new_arts)
            added += len(new_arts)
        else:
            existing[date_str] = arts
            added += len(arts)
    with open(news_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return added


# ── 단일 구간 수집 ────────────────────────────────────────────────────────────
def fetch_window(ticker: str, w_start: str, w_end: str) -> int:
    symbol = FINNHUB_SYMBOL.get(ticker, ticker)
    from_ts = int(datetime.strptime(w_start, "%Y-%m-%d").timestamp())
    to_ts   = int(datetime.strptime(w_end,   "%Y-%m-%d").timestamp()) + 86399

    articles = []
    if ticker == "BTC-USD":
        # company-news 먼저 시도, 실패 시 crypto category fallback
        try:
            articles = finnhub_company_news(symbol, w_start, w_end)
        except Exception:
            pass
        if not articles:
            articles = finnhub_crypto_news(from_ts, to_ts)
    else:
        articles = finnhub_company_news(symbol, w_start, w_end)

    if not articles:
        return 0

    new_by_date: dict = defaultdict(list)
    for art in articles:
        ts = art.get("datetime", 0)
        if not (from_ts <= ts <= to_ts):
            continue
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        if not (w_start <= date_str <= w_end):
            continue
        new_by_date[date_str].append({
            "headline":                art.get("headline", ""),
            "summary":                 art.get("summary", ""),
            "source":                  art.get("source", ""),
            "link":                    art.get("url", ""),
            "content":                 art.get("summary", ""),
            "authors":                 [],
            "overall_sentiment_score": None,
            "overall_sentiment_label": "",
            "ticker_sentiment_score":  "",
            "ticker_sentiment_label":  "",
            "relevance_score":         "",
            "fetch_source":            "finnhub",
        })

    return merge_into_file(ticker, new_by_date)


# ── 커버리지 출력 ─────────────────────────────────────────────────────────────
def print_coverage() -> None:
    print("\n" + "=" * 70)
    print(f"커버리지 ({START_DATE} ~ {END_DATE})")
    print("=" * 70)
    print(f"  {'Ticker':10s}  {'Days':>4s}  {'Articles':>8s}  "
          f"{'Jan':>5s}  {'Feb':>5s}  {'Mar':>5s}  상태")
    print("  " + "-" * 62)
    for ticker in TICKERS:
        news_file = NEWS_DIR / f"{ticker}_news.json"
        if not news_file.exists():
            print(f"  {ticker:10s}: 파일 없음")
            continue
        with open(news_file, encoding="utf-8") as f:
            data = json.load(f)
        in_range = {d: v for d, v in data.items() if START_DATE <= d <= END_DATE}
        total = sum(len(v) for v in in_range.values())
        days  = len(in_range)
        jan = sum(1 for d in in_range if d.startswith("2026-01"))
        feb = sum(1 for d in in_range if d.startswith("2026-02"))
        mar = sum(1 for d in in_range if d.startswith("2026-03"))
        status = "✓" if days >= 60 else f"⚠ ({days}일)"
        print(f"  {ticker:10s}  {days:4d}  {total:8d}  {jan:5d}  {feb:5d}  {mar:5d}  {status}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="현황만 출력")
    parser.add_argument("--reset",  action="store_true", help="상태 초기화 후 전체 재수집")
    args = parser.parse_args()

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        print("상태 초기화 완료. 전체 재수집합니다.")

    if not os.getenv("FINNHUB_API_KEY"):
        print("ERROR: FINNHUB_API_KEY가 .env에 없습니다.")
        print("  https://finnhub.io/register 에서 무료 발급 후 .env에 추가하세요.")
        print("  FINNHUB_API_KEY=your_key_here")
        return

    state = load_state()
    done_set = set(state["done"])

    if args.status:
        total = len(TICKERS) * len(WINDOWS)
        done  = sum(1 for t in TICKERS for w in WINDOWS if state_key(t, *w) in done_set)
        print(f"진행: {done}/{total} 구간 완료")
        for w_start, w_end in WINDOWS:
            parts = [("✓" if state_key(t, w_start, w_end) in done_set else "·") + t for t in TICKERS]
            print(f"  {w_start}~{w_end}  " + "  ".join(parts))
        print_coverage()
        return

    pending = [
        (t, w_start, w_end)
        for t in TICKERS
        for w_start, w_end in WINDOWS
        if state_key(t, w_start, w_end) not in done_set
    ]

    total_all  = len(TICKERS) * len(WINDOWS)
    done_count = total_all - len(pending)

    print("=" * 70)
    print(f"Finnhub 뉴스 수집: {START_DATE} ~ {END_DATE}")
    print(f"전체 {total_all} 구간 중 {done_count} 완료, {len(pending)} 미완료")
    print("=" * 70)

    if not pending:
        print("\n모든 구간 수집 완료!")
        print_coverage()
        return

    for i, (ticker, w_start, w_end) in enumerate(pending):
        print(f"\n[{ticker}] {w_start} ~ {w_end}", end="  ", flush=True)
        try:
            added = fetch_window(ticker, w_start, w_end)
            print(f"+{added} articles")
            state["done"].append(state_key(ticker, w_start, w_end))
            save_state(state)
        except Exception as e:
            print(f"ERROR: {e}")

        if i < len(pending) - 1:
            time.sleep(CALL_DELAY)

    print_coverage()
    print("\nDone!")


if __name__ == "__main__":
    main()
