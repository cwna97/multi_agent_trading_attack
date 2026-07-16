#!/usr/bin/env python3
"""
Forward Fill 적용 — 2026-01-01 ~ 2026-03-31

기사/포스트 없는 날은 직전 날 데이터를 재사용해 날짜 커버리지를 최대화.
원본 파일은 유지하고, forward fill 결과를 별도 디렉토리에 저장.

Output:
  news_data/alpha_vantage_filled/   (뉴스)
  social_data_filled/               (소셜)

Usage:
    .venv/bin/python3 apply_forward_fill_2026.py
    .venv/bin/python3 apply_forward_fill_2026.py --preview
"""

import json
import argparse
from datetime import date, timedelta
from pathlib import Path

START_DATE = "2026-01-01"
END_DATE   = "2026-03-31"
TICKERS    = ["MSFT", "NVDA", "TSLA", "AAPL", "BTC-USD"]

LOCAL_DIR  = Path(__file__).parent / "tradingagents" / "dataflows" / "local_data"
NEWS_SRC   = LOCAL_DIR / "news_data" / "alpha_vantage"
NEWS_DEST  = LOCAL_DIR / "news_data" / "alpha_vantage_filled"
SOC_SRC    = LOCAL_DIR / "social_data"
SOC_DEST   = LOCAL_DIR / "social_data_filled"

start_dt = date.fromisoformat(START_DATE)
end_dt   = date.fromisoformat(END_DATE)
ALL_DAYS = [(start_dt + timedelta(days=i)).isoformat()
            for i in range((end_dt - start_dt).days + 1)]


def apply_fill(src_file: Path, dest_file: Path, ticker: str, label: str, preview: bool) -> None:
    if not src_file.exists():
        print(f"  [{ticker}] {label}: 원본 파일 없음 — 건너뜀")
        return

    original: dict = json.loads(src_file.read_text(encoding="utf-8"))
    in_range = {d: v for d, v in original.items() if START_DATE <= d <= END_DATE}

    filled: dict = {}
    last_data: list = []
    filled_days = 0

    for day in ALL_DAYS:
        if day in in_range:
            last_data = in_range[day]
            filled[day] = last_data
        elif last_data:
            filled_items = []
            for item in last_data:
                i = dict(item)
                i["filled_date"] = day
                filled_items.append(i)
            filled[day] = filled_items
            filled_days += 1

    real_days    = len(in_range)
    total_days   = len(filled)
    no_data_days = len(ALL_DAYS) - total_days

    print(f"  [{ticker}] {label}: 실제 {real_days:2d}일 + fill {filled_days:2d}일 = {total_days:2d}일 커버"
          + (f"  ({no_data_days}일 선행 데이터 없음)" if no_data_days else ""))

    if not preview:
        merged = {k: v for k, v in original.items() if not (START_DATE <= k <= END_DATE)}
        merged.update(filled)
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="저장 없이 커버리지만 출력")
    args = parser.parse_args()

    mode = "미리보기" if args.preview else "저장"
    print("=" * 60)
    print(f"Forward Fill ({START_DATE} ~ {END_DATE})  [{mode}]")
    print("=" * 60)

    print("\n[뉴스]")
    for ticker in TICKERS:
        apply_fill(
            NEWS_SRC  / f"{ticker}_news.json",
            NEWS_DEST / f"{ticker}_news.json",
            ticker, "뉴스", args.preview,
        )

    print("\n[소셜]")
    for ticker in TICKERS:
        apply_fill(
            SOC_SRC  / f"{ticker}_social.json",
            SOC_DEST / f"{ticker}_social.json",
            ticker, "소셜", args.preview,
        )

    if not args.preview:
        print(f"\n완료!")
        print(f"  뉴스  → {NEWS_DEST}")
        print(f"  소셜  → {SOC_DEST}")


if __name__ == "__main__":
    main()
