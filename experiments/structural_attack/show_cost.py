"""
현재까지 누적된 API 비용을 출력합니다.

Usage:
    .venv/bin/python3 show_cost.py               # OpenAI
    .venv/bin/python3 show_cost.py --gemini      # Gemini
    .venv/bin/python3 show_cost.py --watch        # 10초마다 자동 갱신
    .venv/bin/python3 show_cost.py --budget 300   # 예산 대비 진행률 표시
"""

import argparse
import time
from pathlib import Path
from tradingagents.utils.cost_tracker import CostTracker, _load

_LOG_DIR = Path(__file__).parent / "logs"
LOG_PATH        = _LOG_DIR / "cost_log.json"
LOG_PATH_GEMINI = _LOG_DIR / "cost_log_gemini.json"


def display(budget: float = 0.0, gemini: bool = False):
    log_path = LOG_PATH_GEMINI if gemini else LOG_PATH
    data   = _load(log_path)
    total  = data.get("total_cost_usd", 0.0)
    exps   = data.get("experiments", {})

    print("\033[2J\033[H", end="")   # 화면 지우기
    print("=" * 65)
    provider = "Gemini" if gemini else "OpenAI"
    print(f"  {provider} API 비용 현황")
    print("=" * 65)

    if not exps:
        print("  (아직 기록된 실험이 없습니다)")
        print("=" * 65)
        return

    # 실험별 상세
    for name, info in sorted(exps.items()):
        cost  = info.get("cost_usd", 0.0)
        days  = info.get("ticker_days_completed", 0)
        ptok  = info.get("prompt_tokens", 0)
        ctok  = info.get("completion_tokens", 0)
        upd   = info.get("updated_at", "-")
        last  = info.get("last_entry", {})
        print(f"\n  {name}")
        print(f"    비용     : ${cost:.4f}   ({days} ticker-days)")
        print(f"    토큰     : in={ptok:>12,}  out={ctok:>10,}")
        if last:
            print(f"    마지막   : {last.get('ticker','')} {last.get('date','')}  ${last.get('cost_usd',0):.4f}")
        print(f"    갱신     : {upd}")

    # 합계
    print("\n" + "-" * 65)
    print(f"  누적 총 비용 : ${total:.4f}")
    if budget > 0:
        pct  = min(total / budget * 100, 100)
        used = int(40 * pct / 100)
        bar  = "█" * used + "░" * (40 - used)
        color = "\033[91m" if pct >= 90 else "\033[93m" if pct >= 70 else "\033[92m"
        print(f"  예산 진행률 : {color}[{bar}] {pct:.1f}%  (${total:.4f} / ${budget:.2f})\033[0m")
        print(f"  잔여 예산   : ${max(budget - total, 0):.4f}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=0.0, metavar="USD",
                        help="예산 한도 (USD) — 진행률 바 표시용")
    parser.add_argument("--watch",  action="store_true",
                        help="10초마다 자동 갱신 (Ctrl+C로 종료)")
    parser.add_argument("--gemini", action="store_true",
                        help="Gemini 비용 로그 표시 (cost_log_gemini.json)")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                display(args.budget, gemini=args.gemini)
                print("  (10초마다 갱신 — Ctrl+C 종료)")
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n종료")
    else:
        display(args.budget, gemini=args.gemini)


if __name__ == "__main__":
    main()
