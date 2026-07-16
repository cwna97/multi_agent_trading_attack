#!/usr/bin/env python3
"""
Q3 2025 데이터 수집 (2025-07-01 ~ 2025-09-30)

이미 있는 것 (재수집 불필요):
  - 가격 데이터  : 전 티커 (*-2023-01-01-2025-12-31.csv에 Q3 2025 포함)
  - 기술지표     : AAPL / MSFT / NVDA / TSLA (이미 Q3 2025 보유)
  - 뉴스         : MSFT / NVDA / TSLA (이미 Q3 2025 보유)

수집 대상:
  1. 기술지표 (stockstats) — BTC-USD 만
  2. 뉴스                  — AAPL (Finnhub), BTC-USD (Alpha Vantage)
  3. 소셜 (Reddit 공개 API) — 전 티커

무료 API 선택 근거:
  ┌──────────────────┬──────────────────────┬────────────────────────────────────┐
  │ API              │ 무료 한도            │ 선택 이유                          │
  ├──────────────────┼──────────────────────┼────────────────────────────────────┤
  │ Finnhub          │ 60 req/min, 일 제한  │ AAPL 뉴스 주 단위 13 req → 2분    │
  │                  │ 없음                 │ 완료. Q3 2025 확인됨               │
  ├──────────────────┼──────────────────────┼────────────────────────────────────┤
  │ Alpha Vantage    │ 25 req/day           │ BTC-USD 유일 옵션 (Finnhub 유료    │
  │                  │ 5 req/min            │ 전용). 7 req → 일 한도 28% 사용   │
  ├──────────────────┼──────────────────────┼────────────────────────────────────┤
  │ Reddit 공개 JSON │ ~1 req/2s            │ 소셜 유일 무료 옵션.               │
  │                  │ 날짜 필터 없음       │ PullPush.io 날짜 필터 실제 0건     │
  │                  │                      │ 반환 → 사용 불가 확인              │
  └──────────────────┴──────────────────────┴────────────────────────────────────┘

설계 원칙:
  - 누락 방지: 주 단위 window 분할로 건수 편중 회피; 저활성 서브레딧 우선
  - 쏠림 방지: 날짜별 상한(뉴스 MAX_ARTICLES_PER_DAY, 소셜 MAX_POSTS_PER_DAY)
  - Fill 경계: Forward fill 시 윈도우 직전 데이터를 seed로 사용

Usage:
    .venv/bin/python3 fetch_q3_2025_data.py
    .venv/bin/python3 fetch_q3_2025_data.py --skip-news
    .venv/bin/python3 fetch_q3_2025_data.py --skip-social
    .venv/bin/python3 fetch_q3_2025_data.py --dry-run
"""

import argparse
import json
import os
import time
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ─── 날짜 범위 ───────────────────────────────────────────────
START_DATE = "2025-07-01"
END_DATE   = "2025-09-30"

start_dt_utc = datetime(2025, 7,  1,  0,  0,  0, tzinfo=timezone.utc)
end_dt_utc   = datetime(2025, 9, 30, 23, 59, 59, tzinfo=timezone.utc)

ALL_DAYS = [
    (date.fromisoformat(START_DATE) + timedelta(days=i)).isoformat()
    for i in range((date.fromisoformat(END_DATE) - date.fromisoformat(START_DATE)).days + 1)
]

# ─── 경로 ────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent / "tradingagents" / "dataflows" / "local_data"
PRICE_DIR = BASE_DIR / "market_data" / "price_data"
IND_DIR   = BASE_DIR / "indicators"
NEWS_SRC  = BASE_DIR / "news_data" / "alpha_vantage"
NEWS_DEST = BASE_DIR / "news_data" / "alpha_vantage_filled"
SOC_SRC   = BASE_DIR / "social_data"
SOC_DEST  = BASE_DIR / "social_data_filled"

# ─── 티커 설정 ───────────────────────────────────────────────
ALL_TICKERS        = ["AAPL", "MSFT", "NVDA", "TSLA", "BTC-USD"]
# Q3 2025 뉴스 누락 현황 (NYSE 거래일 64일 기준):
#   AAPL:    64일 누락 (전체 없음)
#   MSFT:     3일 누락
#   NVDA:     1일 누락
#   TSLA:    15일 누락
#   BTC-USD: 64일 누락 (기존 조회 심볼 BTC-USD로는 Alpha Vantage 뉴스 0건)
# → Finnhub으로 주식 4종 전체 재수집 (merge 방식으로 기존 보존)
# → 14 windows × 4 tickers = 56 req, ~62초 완료 (무료 60 req/min 한도 내)
NEWS_FINNHUB       = ["AAPL", "MSFT", "NVDA", "TSLA"]  # 전종 Finnhub으로 보충
NEWS_ALPHAVANTAGE  = ["BTC-USD"]     # Alpha Vantage만 가능 (Finnhub BINANCE:BTCUSDT = 유료 전용)
IND_TICKERS        = ["BTC-USD"]     # 나머지는 이미 Q3 2025 지표 보유

# Finnhub 심볼 매핑
FINNHUB_SYMBOL = {"AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA", "TSLA": "TSLA"}

# Alpha Vantage Market News & Sentiment crypto symbols use CRYPTO:<asset>.
# Keep local filenames keyed by BTC-USD, but query/match the provider symbol.
ALPHAVANTAGE_SYMBOL = {"BTC-USD": "CRYPTO:BTC"}

# Forward fill은 짧은 휴장/주말 gap 보정용이다. 몇 달 전 데이터를 새 분기
# 전체에 복제하지 않도록 윈도우 직전 seed는 이 기간 이내일 때만 사용한다.
MAX_FILL_SEED_GAP_DAYS = 7

# ─── 쏠림 방지 상한 ─────────────────────────────────────────
MAX_ARTICLES_PER_DAY = 30   # 뉴스: 날짜별 최대 기사 수 (relevance_score 상위 유지)
MAX_POSTS_PER_DAY    = 20   # 소셜: 날짜별 최대 포스트 수 (score 상위 유지)

# ─── ArcticShift (Reddit 아카이브) ──────────────────────────
# arctic-shift.photon-reddit.com: 날짜 범위 직접 지정 가능한 Reddit 공개 아카이브.
# after/before 파라미터로 Q3 2025 정확히 수집 가능 (Reddit 공개 API 한계 우회).
# 실측 결과 (2026-05-14, Q3 2025):
#   NVDA    : r/NVDA_Stock(993) + r/nvidia(5000)        → 92/92일 커버 ✓
#   BTC-USD : r/btc(4240) + r/CryptoCurrency(5000)      → 92/92일 커버 ✓
#   TSLA    : r/teslamotors(1546) + r/teslainvestorsclub(415) → 92/92일 커버 ✓
#   MSFT    : r/MSFT(2) + r/microsoft(1450)              → 92/92일 커버 ✓
#   AAPL    : r/aapl(244)                                → 81/92일 커버
#             (r/aapl이 극소형 ~3개/일 → 11일 게시물 자체 없음 → forward fill 처리)
TICKER_CONFIG = {
    "AAPL":    {"subreddits": ["aapl"]},
    "MSFT":    {"subreddits": ["MSFT", "microsoft"]},
    "NVDA":    {"subreddits": ["NVDA_Stock", "nvidia"]},
    "TSLA":    {"subreddits": ["teslamotors", "teslainvestorsclub"]},
    "BTC-USD": {"subreddits": ["btc", "CryptoCurrency"]},
}

ARCTICSHIFT_BASE  = "https://arctic-shift.photon-reddit.com/api/posts/search"
REDDIT_HEADERS    = {"User-Agent": "financial-social-collector/1.0 (academic research)"}
ARCTICSHIFT_DELAY = 0.4  # 초: ArcticShift는 공식 rate limit 문서 없음, 보수적으로 설정

INDICATORS = [
    "close_50_sma", "close_200_sma", "close_10_ema",
    "macd", "macds", "macdh",
    "rsi",
    "boll", "boll_ub", "boll_lb",
    "atr", "vwma", "mfi",
]


# ══════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════

def _cap_by_day(by_date: dict, max_per_day: int, score_key: str) -> dict:
    """날짜별 포스트/기사를 score_key 기준 상위 max_per_day개로 제한."""
    capped: dict = {}
    for d, items in by_date.items():
        if len(items) <= max_per_day:
            capped[d] = items
        else:
            sorted_items = sorted(
                items,
                key=lambda x: float(x.get(score_key) or 0),
                reverse=True,
            )
            capped[d] = sorted_items[:max_per_day]
    return capped


def _print_day_stats(label: str, by_date: dict) -> None:
    if not by_date:
        print(f"    [{label}] 데이터 없음")
        return
    ppd = [len(v) for v in by_date.values()]
    print(
        f"    [{label}] {len(ppd)}일 커버, "
        f"min={min(ppd)} max={max(ppd)} avg={statistics.mean(ppd):.1f}/일"
    )


# ══════════════════════════════════════════════════════════════
# 1. 기술지표 — BTC-USD
# ══════════════════════════════════════════════════════════════

def fetch_indicators(ticker: str, dry_run: bool) -> bool:
    print(f"  [지표] {ticker} ...")
    try:
        import pandas as pd
        from stockstats import wrap

        price_files = sorted(PRICE_DIR.glob(f"{ticker}-YFin-data-*2025-12-31.csv"))
        if not price_files:
            price_files = sorted(PRICE_DIR.glob(f"{ticker}-YFin-data-*.csv"))
        if not price_files:
            print(f"    WARN: 가격 파일 없음 → 스킵")
            return False

        price_file = price_files[-1]
        print(f"    가격 파일: {price_file.name}")

        df_raw = pd.read_csv(price_file)
        date_col = next(
            (c for c in df_raw.columns if c.lower() in ("date", "datetime")),
            df_raw.columns[0],
        )
        df_raw[date_col] = pd.to_datetime(df_raw[date_col])
        df_raw = df_raw.rename(columns={date_col: "Date"})
        df = wrap(df_raw.copy())

        for ind in INDICATORS:
            try:
                df[ind]
            except Exception:
                pass

        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        new_data: dict = {}
        for _, row in df.iterrows():
            d = row["Date"]
            if not (START_DATE <= d <= END_DATE):
                continue
            new_data[d] = {
                ind: (
                    None
                    if ind not in df.columns or str(row.get(ind, "nan")) == "nan"
                    else float(row[ind])
                )
                for ind in INDICATORS
                if ind in df.columns
            }

        if not new_data:
            print(f"    WARN: Q3 2025 범위 데이터 없음")
            return False

        ind_file = IND_DIR / f"{ticker}_indicators.json"
        existing: dict = {}
        if ind_file.exists():
            with open(ind_file) as f:
                existing = json.load(f)
        existing.update(new_data)

        if not dry_run:
            IND_DIR.mkdir(parents=True, exist_ok=True)
            with open(ind_file, "w") as f:
                json.dump(existing, f, indent=2)

        print(f"    ✓ {len(new_data)}일 {'(dry-run)' if dry_run else f'→ {ind_file.name}'}")
        return True

    except Exception as e:
        import traceback
        print(f"    ERROR: {e}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════
# 2a. 뉴스 — Finnhub (AAPL)
#     무료 60 req/min, 일 제한 없음, 날짜 범위 직접 지정 가능
# ══════════════════════════════════════════════════════════════

def _build_weekly_windows() -> list[tuple[str, str]]:
    """7일 단위 분할 — Finnhub는 window당 최신 N개 반환 특성상 짧을수록 균등."""
    windows = []
    cur = date.fromisoformat(START_DATE)
    end = date.fromisoformat(END_DATE)
    while cur <= end:
        w_end = min(cur + timedelta(days=6), end)
        windows.append((cur.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))
        cur = w_end + timedelta(days=1)
    return windows


def _finnhub_request(symbol: str, from_date: str, to_date: str) -> list:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY 환경변수 없음")
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


def fetch_news_finnhub(ticker: str, dry_run: bool) -> bool:
    """Finnhub로 뉴스 수집 — 무료 60 req/min, 일 제한 없음."""
    symbol = FINNHUB_SYMBOL.get(ticker, ticker)
    print(f"  [뉴스/Finnhub] {ticker} ({symbol}) ...")
    try:
        windows = _build_weekly_windows()
        print(f"    주 단위 {len(windows)}개 window (Finnhub 60 req/min, 일 제한 없음)")

        new_by_date: dict = defaultdict(list)
        for w_start, w_end in windows:
            articles = _finnhub_request(symbol, w_start, w_end)
            print(f"    {w_start}~{w_end}: {len(articles)}개 기사")
            time.sleep(1.1)   # 60 req/min 안전 마진

            for art in articles:
                ts_epoch = art.get("datetime", 0)
                date_str = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).strftime("%Y-%m-%d")
                if not (START_DATE <= date_str <= END_DATE):
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
                    "relevance_score":         1.0,   # Finnhub는 별도 relevance 없음
                    "fetch_source":            "finnhub",
                })

        return _save_news(ticker, new_by_date, dry_run, "Finnhub")

    except Exception as e:
        print(f"    ERROR: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# 2b. 뉴스 — Alpha Vantage (BTC-USD)
#     무료 25 req/day. BTC-USD는 Finnhub 유료 전용 → Alpha Vantage 불가피.
#     7개 window × 1 ticker = 7 req → 일 한도(25) 28% 사용
# ══════════════════════════════════════════════════════════════

def _build_biweekly_windows_av() -> list[tuple[str, str]]:
    """2주 단위 분할 — Alpha Vantage 1000건 한도 회피, 7 req로 Q3 2025 전체 커버."""
    windows = []
    cur = date.fromisoformat(START_DATE)
    end = date.fromisoformat(END_DATE)
    while cur <= end:
        next_cur = cur + timedelta(days=13)
        if next_cur > end:
            next_cur = end
        windows.append((
            cur.strftime("%Y%m%dT0000"),
            next_cur.strftime("%Y%m%dT2359"),
        ))
        cur = next_cur + timedelta(days=1)
    return windows


def _av_request(symbol: str, time_from: str, time_to: str) -> list:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY 환경변수 없음")
    params = {
        "function":  "NEWS_SENTIMENT",
        "tickers":   symbol,
        "time_from": time_from,
        "time_to":   time_to,
        "sort":      "RELEVANCE",
        "limit":     "1000",
        "apikey":    api_key,
    }
    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage 일일 한도 초과: {data['Information']}")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage note: {data['Note']}")
    return data.get("feed", [])


def _av_date(dt_str: str) -> str:
    try:
        return datetime.strptime(dt_str[:8], "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        return dt_str[:10]


def _av_window_days(time_from: str, time_to: str) -> list[str]:
    start = datetime.strptime(time_from[:8], "%Y%m%d").date()
    end = datetime.strptime(time_to[:8], "%Y%m%d").date()
    return [
        (start + timedelta(days=i)).isoformat()
        for i in range((end - start).days + 1)
    ]


def fetch_news_av(ticker: str, dry_run: bool, max_calls: int | None = None) -> bool:
    """Alpha Vantage로 뉴스 수집 — BTC-USD 전용. 무료 25 req/day."""
    symbol = ALPHAVANTAGE_SYMBOL.get(ticker, ticker)
    windows = _build_biweekly_windows_av()
    print(f"  [뉴스/AlphaVantage] {ticker} ({symbol}) ...")
    print(f"    2주 단위 {len(windows)}개 window → {len(windows)} req / 25 daily 한도")
    if max_calls is not None:
        print(f"    이번 실행 최대 요청 수: {max_calls}")
    existing: dict = {}
    news_file = NEWS_SRC / f"{ticker}_news.json"
    if news_file.exists():
        with open(news_file) as f:
            existing = json.load(f)

    try:
        new_by_date: dict = defaultdict(list)
        calls_used = 0
        for time_from, time_to in windows:
            window_days = _av_window_days(time_from, time_to)
            if all(existing.get(d) for d in window_days):
                print(f"    {time_from[:8]}~{time_to[:8]}: 이미 수집됨 → 스킵")
                continue

            if max_calls is not None and calls_used >= max_calls:
                print(f"    max_calls={max_calls} 도달 → 중단")
                break

            try:
                articles = _av_request(symbol, time_from, time_to)
                calls_used += 1
            except Exception:
                if new_by_date:
                    print("    WARN: 중간 중단 — 지금까지 받은 BTC 뉴스 먼저 저장")
                    _save_news(ticker, new_by_date, dry_run, "Alpha Vantage")
                raise
            print(f"    {time_from[:8]}~{time_to[:8]}: {len(articles)}개 기사")
            time.sleep(13)   # Alpha Vantage 5 req/min 제한 (=12s/req + 여유)

            for art in articles:
                date_str = _av_date(art.get("time_published", ""))
                if not (START_DATE <= date_str <= END_DATE):
                    continue
                ts = next(
                    (t for t in art.get("ticker_sentiment", []) if t.get("ticker") == symbol),
                    {}
                )
                try:
                    rel_score = float(ts.get("relevance_score") or 0)
                except (TypeError, ValueError):
                    rel_score = 0.0
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
                    "relevance_score":         rel_score,
                    "fetch_source":            "alpha_vantage",
                })

            for d in new_by_date:
                existing[d] = new_by_date[d]

        if not new_by_date and calls_used == 0:
            remaining = [
                (a, b)
                for a, b in windows
                if not all(existing.get(d) for d in _av_window_days(a, b))
            ]
            if remaining:
                print(f"    WARN: 수집할 window {len(remaining)}개 남음")
            else:
                print("    모든 window가 이미 수집됨")

        return _save_news(ticker, new_by_date, dry_run, "Alpha Vantage")

    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def _save_news(ticker: str, new_by_date: dict, dry_run: bool, source: str) -> bool:
    """날짜별 상한 적용 후 파일에 merge 저장."""
    # 쏠림 방지: relevance_score 상위 MAX_ARTICLES_PER_DAY개만 유지
    before_cap = sum(len(v) for v in new_by_date.values())
    new_by_date = _cap_by_day(new_by_date, MAX_ARTICLES_PER_DAY, "relevance_score")
    after_cap   = sum(len(v) for v in new_by_date.values())
    if before_cap != after_cap:
        print(f"    상한 적용: {before_cap}건 → {after_cap}건 (cap={MAX_ARTICLES_PER_DAY}/일)")

    _print_day_stats(f"뉴스/{source}", new_by_date)

    if not new_by_date:
        print(f"    WARN: Q3 2025 기사 없음")
        return False

    news_file = NEWS_SRC / f"{ticker}_news.json"
    existing: dict = {}
    if news_file.exists():
        with open(news_file) as f:
            existing = json.load(f)

    # 기존에 이미 데이터가 있는 날짜는 유지, 없는 날짜만 추가
    added_days = 0
    for d, articles in new_by_date.items():
        if d not in existing:
            existing[d] = articles
            added_days += 1

    already = len(new_by_date) - added_days
    print(f"    merge: {added_days}일 신규 추가, {already}일 기존 유지 (덮어쓰기 없음)")

    if not dry_run:
        NEWS_SRC.mkdir(parents=True, exist_ok=True)
        with open(news_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"    → {news_file.name}")

    return True


# ══════════════════════════════════════════════════════════════
# 3. 소셜 (ArcticShift) — 전 티커
#    arctic-shift.photon-reddit.com: 날짜 범위 직접 지정 가능한 Reddit 공개 아카이브.
#    after/before 파라미터로 Q3 2025 정확히 수집, Reddit 공개 API의 1000개 한계 없음.
# ══════════════════════════════════════════════════════════════

def _arcticshift_page(subreddit: str, before_ts: str) -> list:
    """ArcticShift에서 한 페이지(최대 100개) 수집. before_ts 이전 데이터를 최신순으로."""
    params = {
        "subreddit": subreddit,
        "after":     START_DATE,
        "before":    before_ts,
        "limit":     100,
        "sort":      "desc",
    }
    try:
        resp = requests.get(ARCTICSHIFT_BASE, headers=REDDIT_HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("data") or []
    except Exception as e:
        print(f"      WARN r/{subreddit}: {e}")
        return []


def _collect_subreddit_arcticshift(subreddit: str) -> list:
    """ArcticShift로 Q3 2025 전체 수집 (페이지 제한 없음)."""
    all_posts: list = []
    before_ts = END_DATE + "T23:59:59"
    page = 0

    while True:
        posts = _arcticshift_page(subreddit, before_ts)
        time.sleep(ARCTICSHIFT_DELAY)
        if not posts:
            break
        all_posts.extend(posts)
        page += 1
        oldest = min(int(p.get("created_utc", 0)) for p in posts)
        oldest_date = datetime.fromtimestamp(oldest, tz=timezone.utc).strftime("%Y-%m-%d")
        before_ts = datetime.fromtimestamp(oldest, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        print(f"      page {page:3d} | 누적 {len(all_posts):5d}개 | 현재 {oldest_date}", end="\r", flush=True)
        if len(posts) < 100:
            break

    print(f"      page {page:3d} | 누적 {len(all_posts):5d}개 | 완료{' ' * 20}")
    return all_posts


def fetch_social(ticker: str, dry_run: bool) -> bool:
    print(f"  [소셜/ArcticShift] {ticker} ...")
    cfg = TICKER_CONFIG.get(ticker)
    if not cfg:
        print(f"    WARN: 설정 없음")
        return False

    results: list = []
    seen_urls: set = set()

    for sub in cfg["subreddits"]:
        posts = _collect_subreddit_arcticshift(sub)
        # URL 중복 제거
        new_posts = []
        for p in posts:
            url = p.get("url", p.get("permalink", ""))
            if url and url not in seen_urls:
                seen_urls.add(url)
                new_posts.append(p)
        results.extend(new_posts)
        print(f"    r/{sub:20s}: {len(new_posts):4d}개")

    # 날짜별 그룹화 + Q3 2025 필터
    by_date: dict = defaultdict(list)
    for p in results:
        ts = int(p.get("created_utc", 0))
        d  = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        if not (START_DATE <= d <= END_DATE):
            continue
        title = (p.get("title") or "").strip()
        body  = (p.get("selftext") or "").strip()
        if body.lower() in {"[removed]", "[deleted]"}:
            body = ""
        if not title and not body:
            continue
        by_date[d].append({
            "platform":     "reddit",
            "title":        title,
            "text":         body,
            "author":       p.get("author"),
            "score":        p.get("score", 0),
            "num_comments": p.get("num_comments", 0),
            "url":          p.get("url") or f"https://reddit.com{p.get('permalink', '')}",
            "subreddit":    p.get("subreddit", ""),
            "created_at":   datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })

    if not by_date:
        print(f"    WARN: Q3 2025 범위 게시물 없음")
        return False

    # 날짜별 쏠림 방지: score 상위 MAX_POSTS_PER_DAY개만 유지
    before_cap = sum(len(v) for v in by_date.values())
    by_date = _cap_by_day(by_date, MAX_POSTS_PER_DAY, "score")
    after_cap  = sum(len(v) for v in by_date.values())
    if before_cap != after_cap:
        print(f"    상한 적용: {before_cap}개 → {after_cap}개 (cap={MAX_POSTS_PER_DAY}/일)")

    _print_day_stats("소셜", by_date)

    soc_file = SOC_SRC / f"{ticker}_social.json"
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
            existing[d] = sorted(
                existing[d], key=lambda x: float(x.get("score") or 0), reverse=True
            )[:MAX_POSTS_PER_DAY]
        else:
            existing[d] = posts

    if not dry_run:
        SOC_SRC.mkdir(parents=True, exist_ok=True)
        with open(soc_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"    → {soc_file.name}")

    return True


# ══════════════════════════════════════════════════════════════
# 4. Forward Fill
# ══════════════════════════════════════════════════════════════

def _apply_fill(src_file: Path, dest_file: Path, ticker: str, label: str, dry_run: bool) -> None:
    if not src_file.exists():
        print(f"  [{ticker}] {label}: 원본 없음 → 스킵")
        return

    original: dict = json.loads(src_file.read_text(encoding="utf-8"))
    in_range = {d: v for d, v in original.items() if START_DATE <= d <= END_DATE}

    # 윈도우 시작 전 마지막 데이터를 seed로 사용하되, 너무 오래된 seed는 버린다.
    # → July 1에 데이터가 없어도 June 말 데이터로 fill 가능하지만,
    #   March 데이터로 Q3 전체를 채우는 식의 stale fill은 방지한다.
    pre_dates = sorted(d for d in original if d < START_DATE)
    if pre_dates:
        seed_date = pre_dates[-1]
        seed_gap = (date.fromisoformat(START_DATE) - date.fromisoformat(seed_date)).days
        if seed_gap <= MAX_FILL_SEED_GAP_DAYS:
            last_data: list = original[seed_date]
            last_data_date = seed_date
            seed_source = f"(seed: {seed_date})"
        else:
            last_data = []
            last_data_date = None
            seed_source = (
                f"(seed {seed_date} 무시: {seed_gap}일 전, "
                f"max={MAX_FILL_SEED_GAP_DAYS}일)"
            )
    else:
        last_data = []
        last_data_date = None
        seed_source = "(seed 없음 — 윈도우 선행 데이터 없음)"

    filled: dict   = {}
    filled_days    = 0
    no_data_days   = 0

    for day in ALL_DAYS:
        if day in in_range:
            last_data = in_range[day]
            last_data_date = day
            filled[day] = last_data
        elif last_data:
            filled_items = []
            for item in last_data:
                i = dict(item)
                i["filled_date"] = day
                i.setdefault("filled_from_date", last_data_date)
                filled_items.append(i)
            filled[day] = filled_items
            filled_days += 1
        else:
            # seed도 없고 실제 데이터도 없는 경우 (윈도우 맨 앞부분)
            no_data_days += 1

    real_days  = len(in_range)
    total_days = len(filled)
    print(
        f"  [{ticker}] {label}: 실제 {real_days:2d}일 + fill {filled_days:2d}일 = {total_days:2d}일 커버  "
        f"{seed_source}"
        + (f"  ⚠ 선행 데이터 없어 누락 {no_data_days}일" if no_data_days else "")
    )

    if not dry_run:
        merged = {k: v for k, v in original.items() if not (START_DATE <= k <= END_DATE)}
        merged.update(filled)
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_forward_fill(dry_run: bool) -> None:
    print(f"\n[Step 4] Forward Fill ({START_DATE} ~ {END_DATE})")
    print("  [뉴스]")
    for ticker in ALL_TICKERS:
        _apply_fill(NEWS_SRC / f"{ticker}_news.json", NEWS_DEST / f"{ticker}_news.json",
                    ticker, "뉴스", dry_run)
    print("  [소셜]")
    for ticker in ALL_TICKERS:
        _apply_fill(SOC_SRC / f"{ticker}_social.json", SOC_DEST / f"{ticker}_social.json",
                    ticker, "소셜", dry_run)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Q3 2025 데이터 수집 (2025-07-01 ~ 2025-09-30)")
    parser.add_argument("--skip-news",   action="store_true")
    parser.add_argument("--skip-social", action="store_true")
    parser.add_argument("--dry-run",     action="store_true", help="저장 없이 동작 확인")
    args = parser.parse_args()

    print("=" * 65)
    print(f"Q3 2025 데이터 수집  ({START_DATE} ~ {END_DATE})")
    print(f"뉴스 날짜별 상한: {MAX_ARTICLES_PER_DAY}건/일")
    print(f"소셜 날짜별 상한: {MAX_POSTS_PER_DAY}개/일")
    if args.dry_run:
        print("[DRY-RUN — 저장 없음]")
    print("=" * 65)

    summary: dict = {t: {} for t in ALL_TICKERS}

    # Step 1: BTC-USD 기술지표
    print(f"\n[Step 1] 기술지표 (BTC-USD)")
    for ticker in IND_TICKERS:
        summary[ticker]["indicators"] = fetch_indicators(ticker, args.dry_run)

    # Step 2: 뉴스
    if not args.skip_news:
        print(f"\n[Step 2] 뉴스 수집")

        # 2a. Finnhub — AAPL (무료, 일 제한 없음)
        print(f"  2a. Finnhub → {NEWS_FINNHUB}")
        if not os.getenv("FINNHUB_API_KEY"):
            print("  WARN: FINNHUB_API_KEY 없음 → 스킵")
            for t in NEWS_FINNHUB:
                summary[t]["news"] = False
        else:
            for ticker in NEWS_FINNHUB:
                summary[ticker]["news"] = fetch_news_finnhub(ticker, args.dry_run)

        # 2b. Alpha Vantage — BTC-USD (Finnhub 유료 전용이므로 Alpha Vantage 불가피)
        #     7 req 필요 / 무료 25 req/day 한도 → 28% 사용
        print(f"  2b. Alpha Vantage → {NEWS_ALPHAVANTAGE}")
        if not os.getenv("ALPHA_VANTAGE_API_KEY"):
            print("  WARN: ALPHA_VANTAGE_API_KEY 없음 → 스킵")
            for t in NEWS_ALPHAVANTAGE:
                summary[t]["news"] = False
        else:
            for ticker in NEWS_ALPHAVANTAGE:
                summary[ticker]["news"] = fetch_news_av(ticker, args.dry_run)
    else:
        print(f"\n[Step 2] 뉴스 스킵 (--skip-news)")

    # Step 3: 소셜
    if not args.skip_social:
        print(f"\n[Step 3] 소셜 (ArcticShift) — 전 티커")
        for ticker in ALL_TICKERS:
            summary[ticker]["social"] = fetch_social(ticker, args.dry_run)
    else:
        print(f"\n[Step 3] 소셜 스킵 (--skip-social)")

    # Step 4: Forward Fill
    if not args.dry_run:
        apply_forward_fill(dry_run=False)
    else:
        print(f"\n[Step 4] Forward Fill 스킵 (dry-run)")

    # 요약
    print("\n" + "=" * 65)
    print("수집 결과 요약")
    print("=" * 65)
    icon = {True: "✓", False: "✗", None: "—"}
    for ticker, res in summary.items():
        parts = "  ".join(f"{k}:{icon.get(v,'?')}" for k, v in res.items())
        print(f"  {ticker:10s}  {parts}")

    print(f"\n검증 명령:")
    print(
        "  .venv/bin/python3 -c \""
        "import json; "
        "[print(t, 'Q3:', len([x for x in json.load(open("
        "'tradingagents/dataflows/local_data/social_data_filled/'+t+'_social.json'))"
        " if \\'2025-07-01\\'<=x<=\\'2025-09-30\\']))"
        " for t in ['AAPL','MSFT','NVDA','TSLA','BTC-USD']]\""
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
