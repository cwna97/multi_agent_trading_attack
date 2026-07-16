"""
Q3 2025 Clean Experiment Runner — GPT API (gpt-4.1 / gpt-4.1-mini)

기간: 2025-07-01 ~ 2025-09-30
티커: BTC-USD, MSFT, NVDA, TSLA, AAPL (5개)
구조: linear, centralized, decentralized, moa, hybrid (5개)
→ 5개 실험을 동시에 병렬 실행

Usage:
    tmux new-session -s q3_clean
    .venv/bin/python3 run_q3_clean_gpt.py
"""

import subprocess
import os
import sys
import json
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

OPENAI_API_URL = "https://api.openai.com/v1"

START_DATE = "2025-07-01"
END_DATE   = "2025-09-30"
TICKERS    = "BTC-USD,MSFT,NVDA,TSLA,AAPL"

STRUCTURES = ["linear", "centralized", "decentralized", "hybrid"]

EXPERIMENTS = [
    {
        "MODEL_NAME":        f"gpt-4.1_q3_{s}",
        "GRAPH_STRUCTURE":   s,
        "DEEP_THINK_LLM":    "gpt-4.1",
        "QUICK_THINK_LLM":   "gpt-4.1-mini",
        "BACKEND_URL":       OPENAI_API_URL,
        "LLM_PROVIDER":      "openai",
        "SELECTED_ANALYSTS": "market,social,news",
        "TICKERS":           TICKERS,
    }
    for s in STRUCTURES
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def check_openai_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        env_file = os.path.join(SCRIPT_DIR, ".env")
        if os.path.exists(env_file):
            for line in open(env_file):
                if line.startswith("OPENAI_API_KEY="):
                    key = line.strip().split("=", 1)[1]
                    os.environ["OPENAI_API_KEY"] = key
                    break
    return bool(key)


def run_experiment(exp: dict, resume: bool = False) -> tuple[str, int]:
    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        exp["MODEL_NAME"],
        "DEEP_THINK_LLM":    exp["DEEP_THINK_LLM"],
        "QUICK_THINK_LLM":   exp["QUICK_THINK_LLM"],
        "BACKEND_URL":       exp["BACKEND_URL"],
        "LLM_PROVIDER":      exp.get("LLM_PROVIDER", "openai"),
        "GRAPH_STRUCTURE":   exp.get("GRAPH_STRUCTURE", "linear"),
        "SELECTED_ANALYSTS": exp.get("SELECTED_ANALYSTS", "market,social,news"),
        "TICKERS":           exp.get("TICKERS", TICKERS),
        "START_DATE":        START_DATE,
        "END_DATE":          END_DATE,
        "RESUME":            "true" if resume else "false",
    })
    log_path = os.path.join(SCRIPT_DIR, f"logs/{exp['MODEL_NAME']}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"  [{exp['MODEL_NAME']}] started → log: {log_path}")
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, os.path.join(SCRIPT_DIR, "main.py")],
            env=env,
            cwd=SCRIPT_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        proc.wait()

    rc = proc.returncode
    status = "✓ done" if rc == 0 else f"✗ failed (rc={rc})"
    print(f"  [{exp['MODEL_NAME']}] {status}")
    return exp["MODEL_NAME"], rc


def monitor_progress(stop_event: threading.Event, start_time: datetime, experiments_list=None):
    model_names = [e["MODEL_NAME"] for e in (experiments_list or EXPERIMENTS)]

    while not stop_event.is_set():
        lines = ["\033[2J\033[H"]
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        lines.append(f"{'='*70}")
        lines.append(f"Q3 2025 CLEAN  {datetime.now().strftime('%H:%M:%S')}  elapsed={elapsed:.1f}h  (ALL PARALLEL)")
        lines.append(f"Period: {START_DATE} ~ {END_DATE}  Tickers: {TICKERS}")
        lines.append(f"{'='*70}")

        for name in model_names:
            pfile = os.path.join(SCRIPT_DIR, f"logs/.progress_{name}.json")
            if not os.path.exists(pfile):
                lines.append(f"\n[{name}] ⏳ 시작 대기 중...")
                continue
            try:
                with open(pfile) as f:
                    data = json.load(f)
            except Exception:
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
                color  = "\033[92m" if status == "completed" else "\033[93m" if status == "running" else "\033[90m"
                icon   = "✓" if status == "completed" else "⏳" if status == "running" else "○"
                lines.append(f"  {color}{icon} {ticker:7s} [{bar}] {cur:3d}/{total:3d} ({pct:5.1f}%) {last}\033[0m")

            bar_w = int(30 * total_done / total_all) if total_all else 0
            lines.append(f"  Overall [{'█'*bar_w}{'░'*(30-bar_w)}] {total_done}/{total_all} ({overall_pct:.1f}%)")

        lines.append(f"\n{'='*70}\n로그: tail -f logs/gpt-4.1_q3_<structure>.log")
        print("\n".join(lines), flush=True)
        time.sleep(5)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Q3 2025 Clean Experiment Runner — GPT API")
    parser.add_argument("--resume", action="store_true", help="Skip already-succeeded dates")
    parser.add_argument("--structure", choices=STRUCTURES + ["all"], default="all")
    args = parser.parse_args()

    experiments = EXPERIMENTS if args.structure == "all" else [e for e in EXPERIMENTS if e["GRAPH_STRUCTURE"] == args.structure]

    print("=" * 60)
    print("Q3 2025 Clean Experiment Runner — GPT API (ALL PARALLEL)")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print(f"Tickers: {TICKERS}")
    print(f"Structures: {[e['GRAPH_STRUCTURE'] for e in experiments]}  → {len(experiments)} 개 동시 실행")
    if args.resume:
        print("Mode: RESUME (기존 성공 날짜 건너뜀)")
    print("=" * 60)

    print("\n[Step 1] Checking OpenAI API key...")
    if not check_openai_key():
        print("  ERROR: OPENAI_API_KEY not found in env or .env file")
        sys.exit(1)
    print("  OPENAI_API_KEY ✓")

    start_time = datetime.now()
    results    = {}

    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_progress, args=(stop_monitor, start_time, experiments), daemon=True
    )
    monitor_thread.start()

    print(f"\n[Step 2] Launching {len(experiments)} experiments in parallel...")
    with ThreadPoolExecutor(max_workers=len(experiments)) as pool:
        futures = {pool.submit(run_experiment, exp, args.resume): exp["MODEL_NAME"] for exp in experiments}
        for future in as_completed(futures):
            name, rc = future.result()
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            results[name] = rc

    stop_monitor.set()

    elapsed = (datetime.now() - start_time).total_seconds() / 3600
    print("\n" + "=" * 60)
    print(f"All Q3 clean experiments finished.  Total elapsed: {elapsed:.1f}h")
    print(f"Results: trading_results/{START_DATE}_{END_DATE}/")
    for name, rc in results.items():
        print(f"  {'✓' if rc == 0 else f'✗ (rc={rc})'} {name}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
