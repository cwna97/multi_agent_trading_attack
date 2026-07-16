"""
Q1 2026 Clean Baseline Runner — Gemini API
  (gemini-2.5-pro / gemini-2.5-flash)

기간: 2026-01-01 ~ 2026-03-31
티커: BTC-USD, MSFT, NVDA, TSLA, AAPL (5개)
구조: linear, centralized, decentralized, hybrid (4개)
공격: 없음 (clean baseline)

Usage:
    tmux new-session -s q1_clean_gemini
    .venv/bin/python3 run_q1_clean_gemini.py

    .venv/bin/python3 run_q1_clean_gemini.py --resume
    .venv/bin/python3 run_q1_clean_gemini.py --structure centralized
    .venv/bin/python3 run_q1_clean_gemini.py --dry-run
"""

import subprocess
import os
import sys
import json
import argparse
import threading
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).parent.resolve()

DEEP_THINK_LLM  = "gemini-2.5-pro"
QUICK_THINK_LLM = "gemini-2.5-flash"
LLM_PROVIDER    = "google"

START_DATE        = "2026-01-01"
END_DATE          = "2026-03-31"
TICKERS           = "BTC-USD,MSFT,NVDA,TSLA,AAPL"
SELECTED_ANALYSTS = "market,social,news"
BASE_NEWS_SOURCE  = "alpha_vantage_filled"

STRUCTURES = ["linear", "centralized", "decentralized", "hybrid"]


def model_name(structure: str) -> str:
    return f"gemini_q1_{structure}"


def check_google_key() -> bool:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        env_file = SCRIPT_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("GOOGLE_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    os.environ["GOOGLE_API_KEY"] = key
                    break
    return bool(key)


def run_experiment(structure: str, dry_run: bool = False, resume: bool = False) -> tuple[str, int]:
    name = model_name(structure)

    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        name,
        "DEEP_THINK_LLM":    DEEP_THINK_LLM,
        "QUICK_THINK_LLM":   QUICK_THINK_LLM,
        "LLM_PROVIDER":      LLM_PROVIDER,
        "GRAPH_STRUCTURE":   structure,
        "SELECTED_ANALYSTS": SELECTED_ANALYSTS,
        "BASE_NEWS_SOURCE":  BASE_NEWS_SOURCE,
        "START_DATE":        START_DATE,
        "END_DATE":          END_DATE,
        "TICKERS":           TICKERS,
        "ATTACK_ENABLED":    "false",
        "RESUME":            "true" if resume else "false",
        "MAX_BUDGET_USD":    "0",
    })

    log_path = SCRIPT_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [{name}] structure={structure}  {'DRY RUN' if dry_run else 'started'}")

    if dry_run:
        return name, 0

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT_DIR / "main.py")],
            env=env,
            cwd=str(SCRIPT_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        proc.wait()

    rc = proc.returncode
    print(f"  [{name}] {'✓ done' if rc == 0 else f'✗ failed (rc={rc})'}")
    return name, rc


def monitor_progress(stop_event: threading.Event, start_time: datetime, names: list[str]):
    log_dir = SCRIPT_DIR / "logs"

    while not stop_event.is_set():
        lines = ["\033[2J\033[H"]
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        lines.append("=" * 70)
        lines.append(
            f"Q1 2026 GEMINI CLEAN BASELINE  |  {datetime.now().strftime('%H:%M:%S')}  "
            f"elapsed={elapsed:.1f}h"
        )
        lines.append(
            f"Period: {START_DATE} ~ {END_DATE}  |  "
            f"deep={DEEP_THINK_LLM}  quick={QUICK_THINK_LLM}"
        )
        lines.append("=" * 70)

        for name in names:
            pfile = log_dir / f".progress_{name}.json"
            if not pfile.exists():
                lines.append(f"\n[{name}] ⏳ 시작 대기 중...")
                continue
            try:
                data = json.loads(pfile.read_text())
            except Exception:
                lines.append(f"\n[{name}] ⏳ 읽기 오류")
                continue

            total_dates = data.get("total_dates", 1)
            tickers_d   = data.get("tickers", {})
            total_done  = sum(v.get("current", 0) for v in tickers_d.values())
            total_all   = total_dates * len(tickers_d)
            overall_pct = total_done / total_all * 100 if total_all else 0

            lines.append(f"\n[{name}]  updated={data.get('updated_at', '')}")
            for ticker, info in tickers_d.items():
                cur    = info.get("current", 0)
                total  = info.get("total", total_dates)
                status = info.get("status", "waiting")
                last   = info.get("last_date", "")
                pct    = cur / total * 100 if total else 0
                filled = int(20 * cur / total) if total else 0
                bar    = "█" * filled + "░" * (20 - filled)
                if status == "completed":
                    color, icon = "\033[92m", "✓"
                elif status == "running":
                    color, icon = "\033[93m", "⏳"
                else:
                    color, icon = "\033[90m", "○"
                lines.append(
                    f"  {color}{icon} {ticker:7s} [{bar}] {cur:3d}/{total:3d} "
                    f"({pct:5.1f}%) {last}\033[0m"
                )

            bar_w = int(30 * total_done / total_all) if total_all else 0
            lines.append(
                f"  Overall [{'█'*bar_w}{'░'*(30-bar_w)}] "
                f"{total_done}/{total_all} ({overall_pct:.1f}%)"
            )

        lines.append(f"\n{'='*70}")
        lines.append(f"로그: tail -f logs/gemini_q1_<structure>.log")
        print("\n".join(lines), flush=True)
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Q1 2026 Clean Baseline Runner — Gemini API")
    parser.add_argument("--structure", choices=STRUCTURES + ["all"], default="all",
                        help="실행할 구조 (default: all)")
    parser.add_argument("--resume",  action="store_true", help="기존 성공 날짜 건너뜀")
    parser.add_argument("--dry-run", action="store_true", help="실제 실행 없이 설정만 출력")
    args = parser.parse_args()

    structures = STRUCTURES if args.structure == "all" else [args.structure]

    print("=" * 70)
    print("Q1 2026 CLEAN BASELINE RUNNER — Gemini API")
    print(f"Period:     {START_DATE} ~ {END_DATE}")
    print(f"Model:      deep={DEEP_THINK_LLM}  quick={QUICK_THINK_LLM}")
    print(f"Tickers:    {TICKERS}")
    print(f"Structures: {structures}  → {len(structures)}개 동시 실행")
    if args.resume:
        print("Mode: RESUME (기존 성공 날짜 건너뜀)")
    print("=" * 70)

    if not args.dry_run:
        print("\n[Step 1] Checking Google API key...")
        if not check_google_key():
            print("  ERROR: GOOGLE_API_KEY not found in env or .env file")
            sys.exit(1)
        print("  GOOGLE_API_KEY ✓")

    names = [model_name(s) for s in structures]
    start_time = datetime.now()
    results: dict[str, int] = {}

    if not args.dry_run:
        stop_monitor = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_progress, args=(stop_monitor, start_time, names), daemon=True
        )
        monitor_thread.start()

    print(f"\n[Step 2] Launching {len(structures)} experiments in parallel...")
    with ThreadPoolExecutor(max_workers=len(structures)) as pool:
        futures = {
            pool.submit(run_experiment, s, args.dry_run, args.resume): s
            for s in structures
        }
        for future in as_completed(futures):
            name, rc = future.result()
            results[name] = rc

    if not args.dry_run:
        stop_monitor.set()

    elapsed = (datetime.now() - start_time).total_seconds() / 3600
    print("\n" + "=" * 70)
    print(f"All experiments finished.  Total elapsed: {elapsed:.1f}h")
    print(f"Results: trading_results/{START_DATE}_{END_DATE}/")
    for name, rc in results.items():
        print(f"  {'✓' if rc == 0 else f'✗ (rc={rc})'} {name}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
