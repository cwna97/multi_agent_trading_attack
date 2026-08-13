"""
4-Analyst Clean Experiment Runner (GPT-4.1)

기존 실험(3 analysts: market, social, news)에 Fundamentals Analyst를 추가한
clean(no attack) 실험. BTC-USD는 재무제표가 없으므로 제외.

대상:
  - 구조: linear, centralized, decentralized, hybrid (4개, MoA 제외)
  - 티커: MSFT, NVDA, TSLA, AAPL (4개, BTC-USD 제외)
  - 기간: 2026-01-01 ~ 2026-03-31
  - Analysts: market, social, news, fundamentals

결과 저장:
  trading_results/2026-01-01_2026-03-31/gpt-4.1_4a_{structure}/

Usage:
    .venv/bin/python3 run_4analyst_clean_gpt.py
    .venv/bin/python3 run_4analyst_clean_gpt.py --resume   # 중단 후 재시작
"""

import subprocess
import os
import sys
import json
import threading
import time
import argparse
from datetime import datetime

OPENAI_API_URL = "https://api.openai.com/v1"

START_DATE = "2026-01-01"
END_DATE   = "2026-03-31"
TICKERS    = "MSFT,NVDA,TSLA,AAPL"   # BTC-USD 제외

STRUCTURES = ["linear", "centralized", "decentralized", "hybrid"]

EXPERIMENTS = [
    {
        "MODEL_NAME":        f"gpt-4.1_4a_fund_{s}",
        "GRAPH_STRUCTURE":   s,
        "DEEP_THINK_LLM":    "gpt-4.1",
        "QUICK_THINK_LLM":   "gpt-4.1-mini",
        "BACKEND_URL":       OPENAI_API_URL,
        "LLM_PROVIDER":      "openai",
        "SELECTED_ANALYSTS": "market,social,news,fundamentals",
        "TICKERS":           TICKERS,
    }
    for s in STRUCTURES
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_experiment(exp: dict, resume: bool = False) -> tuple:
    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        exp["MODEL_NAME"],
        "DEEP_THINK_LLM":    exp["DEEP_THINK_LLM"],
        "QUICK_THINK_LLM":   exp["QUICK_THINK_LLM"],
        "BACKEND_URL":       exp["BACKEND_URL"],
        "LLM_PROVIDER":      exp["LLM_PROVIDER"],
        "GRAPH_STRUCTURE":   exp["GRAPH_STRUCTURE"],
        "SELECTED_ANALYSTS": exp["SELECTED_ANALYSTS"],
        "TICKERS":           exp["TICKERS"],
        "START_DATE":        START_DATE,
        "END_DATE":          END_DATE,
        "RESUME":            "true" if resume else "false",
        "MAX_BUDGET_USD":    "0",
    })
    log_path = os.path.join(SCRIPT_DIR, f"logs/{exp['MODEL_NAME']}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a" if resume else "w")
    print(f"  [{exp['MODEL_NAME']}] → OpenAI API  (resume={resume})")
    print(f"  [{exp['MODEL_NAME']}] log: {log_path}")
    proc = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPT_DIR, "main.py")],
        env=env,
        cwd=SCRIPT_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc, log_file


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


def monitor_progress(stop_event: threading.Event, start_time: datetime):
    model_names = [e["MODEL_NAME"] for e in EXPERIMENTS]

    while not stop_event.is_set():
        lines = []
        lines.append("\033[2J\033[H")
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        lines.append(f"{'='*70}")
        lines.append(f"4-ANALYST CLEAN EXPERIMENT  {datetime.now().strftime('%H:%M:%S')}  elapsed={elapsed:.1f}h")
        lines.append(f"Period: {START_DATE} ~ {END_DATE}  |  Tickers: {TICKERS}")
        lines.append(f"Analysts: market, social, news, fundamentals")
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

            structure   = data.get("structure", "")
            updated_at  = data.get("updated_at", "")
            total_dates = data.get("total_dates", 1)
            tickers_d   = data.get("tickers", {})

            total_done = sum(v.get("current", 0) for v in tickers_d.values())
            total_all  = total_dates * len(tickers_d)
            overall_pct = total_done / total_all * 100 if total_all else 0

            lines.append(f"\n[{name}] ({structure})  updated={updated_at}")
            for ticker, info in tickers_d.items():
                cur    = info.get("current", 0)
                total  = info.get("total", total_dates)
                status = info.get("status", "waiting")
                last   = info.get("last_date", "")
                pct    = cur / total * 100 if total else 0
                filled = int(20 * cur / total) if total else 0
                bar    = "█" * filled + "░" * (20 - filled)
                if status == "completed":
                    icon, color = "✓", "\033[92m"
                elif status == "running":
                    icon, color = "⏳", "\033[93m"
                else:
                    icon, color = "○", "\033[90m"
                lines.append(
                    f"  {color}{icon} {ticker:7s} [{bar}] {cur:3d}/{total:3d} ({pct:5.1f}%) {last}\033[0m"
                )

            bar_w = int(30 * total_done / total_all) if total_all else 0
            o_bar = "█" * bar_w + "░" * (30 - bar_w)
            lines.append(f"  Overall [{o_bar}] {total_done}/{total_all} ({overall_pct:.1f}%)")

        lines.append(f"\n{'='*70}")
        lines.append("로그: tail -f logs/gpt-4.1_4a_<structure>.log")
        print("\n".join(lines), flush=True)
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="4-Analyst Clean Experiment Runner")
    parser.add_argument("--resume", action="store_true", help="중단된 실험 재시작")
    args = parser.parse_args()

    print("=" * 60)
    print("4-Analyst Clean Experiment Runner (GPT-4.1)")
    print(f"Period : {START_DATE} ~ {END_DATE}")
    print(f"Tickers: {TICKERS}")
    print(f"Structs: {STRUCTURES}")
    print(f"Analysts: market, social, news, fundamentals")
    print(f"Resume : {args.resume}")
    print("=" * 60)

    print("\n[Step 1] Checking OpenAI API key...")
    if not check_openai_key():
        print("  ERROR: OPENAI_API_KEY not found in env or .env file")
        sys.exit(1)
    print("  OPENAI_API_KEY ✓")

    print(f"\n[Step 2] Starting experiments (순차 실행)...")
    for e in EXPERIMENTS:
        print(f"  {e['MODEL_NAME']} ({e['GRAPH_STRUCTURE']})")

    start_time = datetime.now()
    results    = {}

    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_progress,
        args=(stop_monitor, start_time),
        daemon=True,
    )
    monitor_thread.start()

    for exp in EXPERIMENTS:
        proc, log = run_experiment(exp, resume=args.resume)
        print(f"\n  [{exp['MODEL_NAME']}] PID={proc.pid} started")
        proc.wait()
        log.close()
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        rc = proc.returncode
        status = "✓ done" if rc == 0 else f"✗ failed (rc={rc})"
        print(f"  [{exp['MODEL_NAME']}] {status}  elapsed={elapsed:.1f}h")
        results[exp["MODEL_NAME"]] = rc

    stop_monitor.set()

    elapsed = (datetime.now() - start_time).total_seconds() / 3600
    print("\n" + "=" * 60)
    print(f"All experiments finished.  Total elapsed: {elapsed:.1f}h")
    print(f"Results: trading_results/{START_DATE}_{END_DATE}/")
    for name, rc in results.items():
        status = "✓" if rc == 0 else f"✗ (rc={rc})"
        print(f"  {status} {name}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
