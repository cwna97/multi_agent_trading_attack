"""
GPT API experiment runner — Fundamentals Analyst 포함 버전 (MoA 제외)

기존 run_experiments_gpt.py 와 동일한 구조이나:
  - SELECTED_ANALYSTS: market,social,news,fundamentals (4개)
  - MODEL_NAME prefix: gpt-4.1-fund_<structure>  (기존 결과 덮어쓰기 방지)
  - structures: linear, centralized, decentralized, hybrid  (moa 제외)

BTC-USD는 fundamental data가 없으므로 main.py:69 의 자동 제외 로직이 처리.

Usage:
    tmux new-session -s gpt_fund
    uv run python3 run_experiments_gpt_fund.py
"""

import subprocess
import os
import sys
import json
import threading
import time
from datetime import datetime

OPENAI_API_URL = "https://api.openai.com/v1"

START_DATE = "2026-01-01"
END_DATE   = "2026-03-31"

###############################################################################
# 실험 세트: 4구조 (MoA 제외) × fundamentals 포함
###############################################################################
STRUCTURES = ["linear", "centralized", "decentralized", "hybrid"]

EXPERIMENTS = [
    {
        "MODEL_NAME":        f"gpt-4.1-fund_{s}",
        "DEEP_THINK_LLM":    "gpt-4.1",
        "QUICK_THINK_LLM":   "gpt-4.1-mini",
        "BACKEND_URL":       OPENAI_API_URL,
        "LLM_PROVIDER":      "openai",
        "GRAPH_STRUCTURE":   s,
        "SELECTED_ANALYSTS": "market,social,news,fundamentals",
    }
    for s in STRUCTURES
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_experiment(exp: dict):
    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        exp["MODEL_NAME"],
        "DEEP_THINK_LLM":    exp["DEEP_THINK_LLM"],
        "QUICK_THINK_LLM":   exp["QUICK_THINK_LLM"],
        "BACKEND_URL":       exp["BACKEND_URL"],
        "LLM_PROVIDER":      exp.get("LLM_PROVIDER", "openai"),
        "GRAPH_STRUCTURE":   exp.get("GRAPH_STRUCTURE", "linear"),
        "SELECTED_ANALYSTS": exp.get("SELECTED_ANALYSTS", "market,social,news,fundamentals"),
        "TICKERS":           "MSFT,NVDA,TSLA,AAPL",  # BTC-USD 제외
        "START_DATE":        START_DATE,
        "END_DATE":          END_DATE,
    })
    log_path = os.path.join(SCRIPT_DIR, f"logs/{exp['MODEL_NAME']}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    print(f"  [{exp['MODEL_NAME']}] → OpenAI API  analysts={exp['SELECTED_ANALYSTS']}")
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
        lines.append(f"GPT-FUND EXPERIMENT PROGRESS  {datetime.now().strftime('%H:%M:%S')}  elapsed={elapsed:.1f}h")
        lines.append(f"Period: {START_DATE} ~ {END_DATE}  |  Analysts: market,social,news,fundamentals")
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
            tickers     = data.get("tickers", {})

            total_done = sum(v.get("current", 0) for v in tickers.values())
            total_all  = total_dates * len(tickers)
            overall_pct = total_done / total_all * 100 if total_all else 0

            lines.append(f"\n[{name}] ({structure})  updated={updated_at}")
            for ticker, info in tickers.items():
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
                lines.append(f"  {color}{icon} {ticker:7s} [{bar}] {cur:3d}/{total:3d} ({pct:5.1f}%) {last}\033[0m")

            bar_w = int(30 * total_done / total_all) if total_all else 0
            o_bar = "█" * bar_w + "░" * (30 - bar_w)
            lines.append(f"  Overall [{o_bar}] {total_done}/{total_all} ({overall_pct:.1f}%)")

        lines.append(f"\n{'='*70}")
        lines.append("로그 확인: tail -f logs/gpt-4.1-fund_<structure>.log")
        print("\n".join(lines), flush=True)
        time.sleep(5)


def main():
    print("=" * 70)
    print("GPT API Experiment Runner — Fundamentals Analyst 포함 (MoA 제외)")
    print(f"Period : {START_DATE} ~ {END_DATE}")
    print(f"Structures : {STRUCTURES}")
    print(f"Tickers    : MSFT, NVDA, TSLA, AAPL  (BTC-USD 제외)")
    print(f"Analysts   : market, social, news, fundamentals")
    print(f"Results → trading_results/{START_DATE}_{END_DATE}/gpt-4.1-fund_<structure>/")
    print("=" * 70)

    print("\n[Step 1] Checking OpenAI API key...")
    if not check_openai_key():
        print("  ERROR: OPENAI_API_KEY not found in env or .env file")
        sys.exit(1)
    print("  OPENAI_API_KEY ✓")

    # 기존 결과 충돌 체크
    print("\n[Step 2] Checking for existing results...")
    results_base = os.path.join(SCRIPT_DIR, "trading_results", f"{START_DATE}_{END_DATE}")
    conflict = False
    for exp in EXPERIMENTS:
        out_dir = os.path.join(results_base, exp["MODEL_NAME"])
        if os.path.exists(out_dir):
            print(f"  ⚠️  Already exists: {out_dir}")
            conflict = True
    if conflict:
        ans = input("\n  일부 결과 폴더가 이미 존재합니다. 덮어쓸까요? [y/N]: ").strip().lower()
        if ans != "y":
            print("  중단합니다.")
            sys.exit(0)
    else:
        print("  충돌 없음 ✓")

    print(f"\n[Step 3] Starting {len(EXPERIMENTS)} experiments (순차 실행)...")
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
        proc, log = run_experiment(exp)
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
    print("\n" + "=" * 70)
    print(f"All experiments finished.  Total elapsed: {elapsed:.1f}h")
    print(f"Results saved to: trading_results/{START_DATE}_{END_DATE}/")
    for name, rc in results.items():
        status = "✓" if rc == 0 else f"✗ (rc={rc})"
        print(f"  {status} {name}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
