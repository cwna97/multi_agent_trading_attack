#!/usr/bin/env python3
"""
TASK 1: 2026-01-01 ~ 2026-03-31 뉴스 수집 — 주간 분할 + 재실행 가능 버전

전략:
  - 90일을 2주(14일) 단위로 분할 → 7 구간 × 5 티커 = 35 calls 총 필요
  - 상태 파일(.news_fetch_state.json)로 완료 구간 추적 → 매일 재실행 가능
  - 하루 최대 calls 수를 --max-calls 로 제어 (기본 23, FREE 25/day 중 여유 2 확보)
  - sort=EARLIEST → 월초부터 채워져 고른 날짜 커버리지 확보
  - 기존 news JSON에 merge (덮어쓰기 X)

Usage:
    # 매일 KST 09:00 이후 (API 리셋 후) 실행
    .venv/bin/python3 fetch_news_only_2026.py
    .venv/bin/python3 fetch_news_only_2026.py --max-calls 5   # 오늘 남은 quota 적을 때
"""

import os
import sys
import json
import time
import argparse
import requests
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── Config ───────────────────────────────────────────────────────────────────
START_DATE = "2026-01-01"
END_DATE   = "2026-03-31"
TICKERS    = ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]

WINDOW_DAYS  = 14    # 2주 단위 분할
LIMIT        = 50    # FREE 플랜 최대 articles/call
TICKER_DELAY = 13    # 티커 전환 딜레이(초)
WINDOW_DELAY = 3     # 같은 티커 내 구간 전환 딜레이(초)

# AV FREE 기준 하루 25 req, 여유 2 확보 → 기본 23
DEFAULT_MAX_CALLS = 23

# BTC-USD 대안 ticker 후보
AV_TICKER_MAP = {
    "BTC-USD": ["BTC", "CRYPTO:BTC", "BTC-USD"],
}

BASE_DIR  = Path(__file__).parent
NEWS_DIR  = BASE_DIR / "tradingagents" / "dataflows" / "local_data" / "news_data" / "alpha_vantage"
STATE_FILE = BASE_DIR / ".news_fetch_state.json"


# ── 구간 생성 ─────────────────────────────────────────────────────────────────
def generate_windows(start: str, end: str, window_days: int) -> list[tuple[str, str]]:
    windows = []
    cur = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while cur <= end_dt:
        w_end = min(cur + timedelta(days=window_days - 1), end_dt)
        windows.append((cur.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))
        cur = w_end + timedelta(days=1)
    return windows


WINDOWS = generate_windows(START_DATE, END_DATE, WINDOW_DAYS)


# ── 상태 관리 ─────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"done": []}   # done: list of "TICKER|start|end"


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def state_key(ticker: str, w_start: str, w_end: str) -> str:
    return f"{ticker}|{w_start}|{w_end}"


# ── Alpha Vantage 요청 ────────────────────────────────────────────────────────
def av_request(av_ticker: str, time_from: str, time_to: str) -> list[dict]:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set in .env")
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers":  av_ticker,
        "time_from": time_from,
        "time_to":   time_to,
        "sort":      "EARLIEST",
        "limit":     str(LIMIT),
        "apikey":    api_key,
    }
    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "Information" in data:
        raise RuntimeError(f"AV rate limit: {data['Information']}")
    if "Note" in data:
        raise RuntimeError(f"AV note: {data['Note']}")
    return data.get("feed", [])


def format_date(dt_str: str) -> str:
    try:
        return datetime.strptime(dt_str[:8], "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        return dt_str[:10]


# ── 파일 merge ────────────────────────────────────────────────────────────────
def merge_into_file(ticker: str, new_by_date: dict[str, list]) -> int:
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
    av_candidates = AV_TICKER_MAP.get(ticker, [ticker])
    time_from = datetime.strptime(w_start, "%Y-%m-%d").strftime("%Y%m%dT0000")
    time_to   = datetime.strptime(w_end,   "%Y-%m-%d").strftime("%Y%m%dT2359")

    articles = []
    used_ticker = av_candidates[0]
    for av_ticker in av_candidates:
        try:
            result = av_request(av_ticker, time_from, time_to)
            if result:
                articles = result
                used_ticker = av_ticker
                break
        except RuntimeError as e:
            print(f"      WARN [{av_ticker}] {e}")
            return -1   # rate limit → 즉시 중단 신호
        except Exception as e:
            print(f"      WARN [{av_ticker}] {e}")

    if not articles:
        return 0

    new_by_date: dict[str, list] = defaultdict(list)
    for art in articles:
        date_str = format_date(art.get("time_published", ""))
        if not (w_start <= date_str <= w_end):
            continue
        ts = {}
        for t in art.get("ticker_sentiment", []):
            t_name = t.get("ticker", "")
            if t_name == ticker or t_name == used_ticker or ticker.replace("-USD", "") in t_name:
                ts = t
                break
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

    added = merge_into_file(ticker, new_by_date)
    return added


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
        total    = sum(len(v) for v in in_range.values())
        days     = len(in_range)
        jan = sum(1 for d in in_range if d.startswith("2026-01"))
        feb = sum(1 for d in in_range if d.startswith("2026-02"))
        mar = sum(1 for d in in_range if d.startswith("2026-03"))
        status = "✓" if days >= 60 else f"⚠ ({days}일)"
        print(f"  {ticker:10s}  {days:4d}  {total:8d}  {jan:5d}  {feb:5d}  {mar:5d}  {status}")


# ── 진행 현황 출력 ─────────────────────────────────────────────────────────────
def print_progress(state: dict) -> None:
    done_set = set(state["done"])
    total = len(TICKERS) * len(WINDOWS)
    done  = sum(1 for t in TICKERS for w in WINDOWS if state_key(t, *w) in done_set)
    print(f"\n진행: {done}/{total} 구간 완료  "
          f"(남은 구간: {total-done}, 남은 예상 calls: {total-done})")
    print(f"구간 목록 ({len(WINDOWS)}개 × {len(TICKERS)} tickers):")
    for w_start, w_end in WINDOWS:
        parts = []
        for t in TICKERS:
            flag = "✓" if state_key(t, w_start, w_end) in done_set else "·"
            parts.append(f"{flag}{t}")
        print(f"  {w_start}~{w_end}  " + "  ".join(parts))


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS,
                        help=f"오늘 사용할 최대 API calls (기본: {DEFAULT_MAX_CALLS})")
    parser.add_argument("--status", action="store_true",
                        help="수집하지 않고 현황만 출력")
    args = parser.parse_args()

    state = load_state()
    done_set = set(state["done"])

    if args.status:
        print_progress(state)
        print_coverage()
        return

    # 미완료 (ticker, window) 목록
    pending = [
        (t, w_start, w_end)
        for t in TICKERS
        for w_start, w_end in WINDOWS
        if state_key(t, w_start, w_end) not in done_set
    ]

    total_needed = len(pending)
    total_all    = len(TICKERS) * len(WINDOWS)
    done_count   = total_all - total_needed

    print("=" * 70)
    print(f"뉴스 수집 (주간 분할): {START_DATE} ~ {END_DATE}")
    print(f"전체 {total_all} 구간 중 {done_count} 완료, {total_needed} 미완료")
    print(f"오늘 최대 {args.max_calls} calls 실행")
    print("=" * 70)

    if not pending:
        print("\n모든 구간 수집 완료!")
        print_coverage()
        return

    calls_made = 0
    prev_ticker = None

    for ticker, w_start, w_end in pending:
        if calls_made >= args.max_calls:
            remaining = total_needed - calls_made
            print(f"\n오늘 quota 소진 ({args.max_calls} calls). "
                  f"남은 구간: {remaining}개 — 내일 KST 09:00 이후 재실행하세요.")
            break

        # 티커 전환 딜레이
        if prev_ticker is not None and prev_ticker != ticker:
            print(f"  → {TICKER_DELAY}초 대기...", flush=True)
            time.sleep(TICKER_DELAY)
        elif prev_ticker == ticker:
            time.sleep(WINDOW_DELAY)

        print(f"\n[{ticker}] {w_start} ~ {w_end}", end="  ", flush=True)
        added = fetch_window(ticker, w_start, w_end)
        calls_made += 1

        if added == -1:   # rate limit
            print("RATE LIMIT — 오늘 중단. 내일 재실행하세요.")
            break

        print(f"+{added} articles")
        state["done"].append(state_key(ticker, w_start, w_end))
        save_state(state)
        prev_ticker = ticker

    print(f"\n이번 실행: {calls_made} calls 사용")
    print_progress(state)
    print_coverage()
    print("\nDone!")


if __name__ == "__main__":
    main()
