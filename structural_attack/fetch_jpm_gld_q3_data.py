#!/usr/bin/env python3
"""
JPM / GLD  Q3 2025 데이터 수집 (2025-07-01 ~ 2025-09-30)

  뉴스  : Finnhub  (JPM, GLD 모두 지원)
  소셜  : ArcticShift Reddit 아카이브
            JPM → r/banking + r/finance
            GLD → r/gold + r/silverbugs
  지표  : stockstats (가격 파일 이미 존재)
  Fill  : forward fill → *_filled/ 에 저장

Usage:
    .venv/bin/python fetch_jpm_gld_q3_data.py
    .venv/bin/python fetch_jpm_gld_q3_data.py --skip-news
    .venv/bin/python fetch_jpm_gld_q3_data.py --skip-social
    .venv/bin/python fetch_jpm_gld_q3_data.py --dry-run
"""

import argparse, json, os, time, statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

START_DATE = "2025-07-01"
END_DATE   = "2025-09-30"

start_dt_utc = datetime(2025, 7,  1,  0,  0,  0, tzinfo=timezone.utc)
end_dt_utc   = datetime(2025, 9, 30, 23, 59, 59, tzinfo=timezone.utc)

ALL_DAYS = [
    (date.fromisoformat(START_DATE) + timedelta(days=i)).isoformat()
    for i in range((date.fromisoformat(END_DATE) - date.fromisoformat(START_DATE)).days + 1)
]

BASE_DIR  = Path(__file__).parent / "tradingagents" / "dataflows" / "local_data"
PRICE_DIR = BASE_DIR / "market_data" / "price_data"
IND_DIR   = BASE_DIR / "indicators"
NEWS_SRC  = BASE_DIR / "news_data" / "alpha_vantage"
NEWS_DEST = BASE_DIR / "news_data" / "alpha_vantage_filled"
SOC_SRC   = BASE_DIR / "social_data"
SOC_DEST  = BASE_DIR / "social_data_filled"

TICKERS = ["JPM", "GLD"]

TICKER_CONFIG = {
    "JPM": {"subreddits": ["banking", "finance"]},
    "GLD": {"subreddits": ["gold", "silverbugs"]},
}

MAX_ARTICLES_PER_DAY = 30
MAX_POSTS_PER_DAY    = 20
MAX_FILL_SEED_GAP_DAYS = 7

ARCTICSHIFT_BASE  = "https://arctic-shift.photon-reddit.com/api/posts/search"
REDDIT_HEADERS    = {"User-Agent": "financial-social-collector/1.0 (academic research)"}
ARCTICSHIFT_DELAY = 0.5

INDICATORS = [
    "close_50_sma", "close_200_sma", "close_10_ema",
    "macd", "macds", "macdh", "rsi",
    "boll", "boll_ub", "boll_lb",
    "atr", "vwma", "mfi",
]


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _cap_by_day(by_date, max_per_day, score_key):
    capped = {}
    for d, items in by_date.items():
        if len(items) <= max_per_day:
            capped[d] = items
        else:
            capped[d] = sorted(items, key=lambda x: float(x.get(score_key) or 0),
                               reverse=True)[:max_per_day]
    return capped


def _print_stats(label, by_date):
    if not by_date:
        print(f"    [{label}] 데이터 없음")
        return
    ppd = [len(v) for v in by_date.values()]
    print(f"    [{label}] {len(ppd)}일 커버, "
          f"min={min(ppd)} max={max(ppd)} avg={statistics.mean(ppd):.1f}/일")


def _build_weekly_windows():
    windows = []
    cur = date.fromisoformat(START_DATE)
    end = date.fromisoformat(END_DATE)
    while cur <= end:
        w_end = min(cur + timedelta(days=6), end)
        windows.append((cur.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))
        cur = w_end + timedelta(days=1)
    return windows


# ── 1. 기술지표 ───────────────────────────────────────────────────────────────

def fetch_indicators(ticker, dry_run):
    print(f"  [지표] {ticker} ...")
    try:
        import pandas as pd
        from stockstats import wrap

        price_files = sorted(PRICE_DIR.glob(f"{ticker}-YFin-data-*2025-12-31.csv"))
        if not price_files:
            price_files = sorted(PRICE_DIR.glob(f"{ticker}-YFin-data-*.csv"))
        if not price_files:
            print(f"    WARN: 가격 파일 없음"); return False

        df_raw = pd.read_csv(price_files[-1])
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df = wrap(df_raw.copy())
        for ind in INDICATORS:
            try: df[ind]
            except Exception: pass

        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        new_data = {}
        for _, row in df.iterrows():
            d = row["Date"]
            if not (START_DATE <= d <= END_DATE): continue
            new_data[d] = {ind: (None if ind not in df.columns or
                                 str(row.get(ind,"nan"))=="nan" else float(row[ind]))
                           for ind in INDICATORS if ind in df.columns}

        if not new_data:
            print("    WARN: Q3 2025 범위 없음"); return False

        ind_file = IND_DIR / f"{ticker}_indicators.json"
        existing = json.loads(ind_file.read_text()) if ind_file.exists() else {}
        existing.update(new_data)
        if not dry_run:
            IND_DIR.mkdir(parents=True, exist_ok=True)
            ind_file.write_text(json.dumps(existing, indent=2))
        print(f"    ✓ {len(new_data)}일 저장")
        return True
    except Exception as e:
        print(f"    ERROR: {e}"); return False


# ── 2. 뉴스 (Finnhub) ────────────────────────────────────────────────────────

def _finnhub_request(symbol, from_date, to_date):
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key: raise ValueError("FINNHUB_API_KEY 없음")
    resp = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={"symbol": symbol, "from": from_date, "to": to_date, "token": api_key},
        timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Finnhub error: {data['error']}")
    return data if isinstance(data, list) else []


def fetch_news_finnhub(ticker, dry_run):
    print(f"  [뉴스/Finnhub] {ticker} ...")
    try:
        windows = _build_weekly_windows()
        new_by_date = defaultdict(list)

        for w_start, w_end in windows:
            articles = _finnhub_request(ticker, w_start, w_end)
            print(f"    {w_start}~{w_end}: {len(articles)}건")
            time.sleep(1.1)
            for art in articles:
                ts = art.get("datetime", 0)
                d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                if not (START_DATE <= d <= END_DATE): continue
                new_by_date[d].append({
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
                    "relevance_score":         1.0,
                    "fetch_source":            "finnhub",
                })

        # 상한 적용
        new_by_date = _cap_by_day(new_by_date, MAX_ARTICLES_PER_DAY, "relevance_score")
        _print_stats("뉴스/Finnhub", new_by_date)

        if not new_by_date:
            print("    WARN: Q3 2025 기사 없음"); return False

        news_file = NEWS_SRC / f"{ticker}_news.json"
        existing = json.loads(news_file.read_text()) if news_file.exists() else {}
        added = sum(1 for d in new_by_date if d not in existing)
        for d, arts in new_by_date.items():
            if d not in existing:
                existing[d] = arts
        print(f"    merge: {added}일 신규 추가")
        if not dry_run:
            NEWS_SRC.mkdir(parents=True, exist_ok=True)
            news_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
            print(f"    → {news_file.name}")
        return True
    except Exception as e:
        print(f"    ERROR: {e}"); return False


# ── 3. 소셜 (ArcticShift) ────────────────────────────────────────────────────

def _arcticshift_page(subreddit, before_ts):
    params = {"subreddit": subreddit, "after": START_DATE,
              "before": before_ts, "limit": 100, "sort": "desc"}
    try:
        resp = requests.get(ARCTICSHIFT_BASE, headers=REDDIT_HEADERS,
                            params=params, timeout=20)
        resp.raise_for_status()
        return resp.json().get("data") or []
    except Exception as e:
        print(f"      WARN r/{subreddit}: {e}"); return []


def _collect_subreddit(subreddit):
    all_posts = []
    before_ts = END_DATE + "T23:59:59"
    page = 0
    while True:
        posts = _arcticshift_page(subreddit, before_ts)
        time.sleep(ARCTICSHIFT_DELAY)
        if not posts: break
        all_posts.extend(posts)
        page += 1
        oldest = min(int(p.get("created_utc", 0)) for p in posts)
        oldest_date = datetime.fromtimestamp(oldest, tz=timezone.utc).strftime("%Y-%m-%d")
        before_ts = datetime.fromtimestamp(oldest, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        print(f"      page {page:3d} | 누적 {len(all_posts):5d}개 | {oldest_date}", end="\r", flush=True)
        if len(posts) < 100: break
    print(f"      page {page:3d} | 누적 {len(all_posts):5d}개 | 완료{' '*20}")
    return all_posts


def fetch_social(ticker, dry_run):
    print(f"  [소셜/ArcticShift] {ticker} ...")
    cfg = TICKER_CONFIG.get(ticker)
    if not cfg:
        print("    WARN: 설정 없음"); return False

    results, seen_urls = [], set()
    for sub in cfg["subreddits"]:
        posts = _collect_subreddit(sub)
        new_posts = []
        for p in posts:
            url = p.get("url", p.get("permalink", ""))
            if url and url not in seen_urls:
                seen_urls.add(url); new_posts.append(p)
        results.extend(new_posts)
        print(f"    r/{sub:20s}: {len(new_posts):4d}건")

    by_date = defaultdict(list)
    for p in results:
        ts = int(p.get("created_utc", 0))
        d  = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        if not (START_DATE <= d <= END_DATE): continue
        title = (p.get("title") or "").strip()
        body  = (p.get("selftext") or "").strip()
        if body.lower() in {"[removed]", "[deleted]"}: body = ""
        if not title and not body: continue
        by_date[d].append({
            "platform":     "reddit",
            "title":        title,
            "text":         body,
            "author":       p.get("author"),
            "score":        p.get("score", 0),
            "num_comments": p.get("num_comments", 0),
            "url":          p.get("url") or f"https://reddit.com{p.get('permalink','')}",
            "subreddit":    p.get("subreddit", ""),
            "created_at":   datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })

    if not by_date:
        print("    WARN: Q3 2025 게시물 없음"); return False

    before_cap = sum(len(v) for v in by_date.values())
    by_date = _cap_by_day(by_date, MAX_POSTS_PER_DAY, "score")
    after_cap = sum(len(v) for v in by_date.values())
    if before_cap != after_cap:
        print(f"    상한 적용: {before_cap} → {after_cap}건")
    _print_stats("소셜", by_date)

    soc_file = SOC_SRC / f"{ticker}_social.json"
    existing = {}
    if soc_file.exists():
        try: existing = json.loads(soc_file.read_text())
        except: pass
    for d, posts in by_date.items():
        if d in existing:
            ex_urls = {p.get("url") for p in existing[d]}
            existing[d].extend(p for p in posts if p.get("url") not in ex_urls)
            existing[d] = sorted(existing[d], key=lambda x: float(x.get("score") or 0),
                                 reverse=True)[:MAX_POSTS_PER_DAY]
        else:
            existing[d] = posts

    if not dry_run:
        SOC_SRC.mkdir(parents=True, exist_ok=True)
        soc_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
        print(f"    → {soc_file.name}")
    return True


# ── 4. Forward Fill ───────────────────────────────────────────────────────────

def _apply_fill(src_file, dest_file, ticker, label, dry_run):
    if not src_file.exists():
        print(f"  [{ticker}] {label}: 원본 없음 → 스킵"); return

    original = json.loads(src_file.read_text(encoding="utf-8"))
    in_range  = {d: v for d, v in original.items() if START_DATE <= d <= END_DATE}

    pre_dates = sorted(d for d in original if d < START_DATE)
    if pre_dates:
        seed_date = pre_dates[-1]
        gap = (date.fromisoformat(START_DATE) - date.fromisoformat(seed_date)).days
        last_data = original[seed_date] if gap <= MAX_FILL_SEED_GAP_DAYS else []
        last_data_date = seed_date if gap <= MAX_FILL_SEED_GAP_DAYS else None
        seed_src = f"(seed: {seed_date})" if gap <= MAX_FILL_SEED_GAP_DAYS \
                   else f"(seed {seed_date} 무시: {gap}일 전)"
    else:
        last_data, last_data_date, seed_src = [], None, "(seed 없음)"

    filled, filled_days, no_data_days = {}, 0, 0
    for day in ALL_DAYS:
        if day in in_range:
            last_data = in_range[day]; last_data_date = day; filled[day] = last_data
        elif last_data:
            items = []
            for item in last_data:
                i = dict(item); i["filled_date"] = day
                i.setdefault("filled_from_date", last_data_date)
                items.append(i)
            filled[day] = items; filled_days += 1
        else:
            no_data_days += 1

    print(f"  [{ticker}] {label}: 실제 {len(in_range):2d}일 + fill {filled_days:2d}일 = "
          f"{len(filled):2d}일  {seed_src}"
          + (f"  ⚠ 누락 {no_data_days}일" if no_data_days else ""))

    if not dry_run:
        merged = {k: v for k, v in original.items() if not (START_DATE <= k <= END_DATE)}
        merged.update(filled)
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2))


def apply_forward_fill(dry_run):
    print(f"\n[Step 4] Forward Fill")
    print("  [뉴스]")
    for t in TICKERS:
        _apply_fill(NEWS_SRC / f"{t}_news.json", NEWS_DEST / f"{t}_news.json", t, "뉴스", dry_run)
    print("  [소셜]")
    for t in TICKERS:
        _apply_fill(SOC_SRC / f"{t}_social.json", SOC_DEST / f"{t}_social.json", t, "소셜", dry_run)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-news",   action="store_true")
    parser.add_argument("--skip-social", action="store_true")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"JPM / GLD  Q3 2025 데이터 수집  ({START_DATE} ~ {END_DATE})")
    if args.dry_run: print("[DRY-RUN]")
    print("=" * 60)

    summary = {t: {} for t in TICKERS}

    print(f"\n[Step 1] 기술지표")
    for t in TICKERS:
        summary[t]["indicators"] = fetch_indicators(t, args.dry_run)

    if not args.skip_news:
        print(f"\n[Step 2] 뉴스 (Finnhub)")
        if not os.getenv("FINNHUB_API_KEY"):
            print("  WARN: FINNHUB_API_KEY 없음 → 스킵")
        else:
            for t in TICKERS:
                summary[t]["news"] = fetch_news_finnhub(t, args.dry_run)

    if not args.skip_social:
        print(f"\n[Step 3] 소셜 (ArcticShift)")
        for t in TICKERS:
            summary[t]["social"] = fetch_social(t, args.dry_run)

    if not args.dry_run:
        apply_forward_fill(dry_run=False)

    print("\n" + "=" * 60)
    icon = {True: "✓", False: "✗", None: "—"}
    for t, res in summary.items():
        parts = "  ".join(f"{k}:{icon.get(v,'?')}" for k, v in res.items())
        print(f"  {t:6s}  {parts}")
    print("\nDone!")


if __name__ == "__main__":
    main()
