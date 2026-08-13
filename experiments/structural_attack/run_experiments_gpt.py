"""
GPT API experiment runner (gpt-4.1 / gpt-4.1-mini)

Ollama 실험(run_experiments.py)과 동시에 tmux 별도 세션에서 실행.
GPU 사용 없이 OpenAI API로 요청 → 리소스 충돌 없음.

Usage:
    tmux new-session -s gpt
    .venv/bin/python3 run_experiments_gpt.py
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
# GPT 실험 세트: 4구조 × 2모델 순차 실행
###############################################################################
def make_exps(model_name, deep, quick):
    structures = ["linear", "centralized", "decentralized", "moa", "hybrid"]
    return [
        {
            "MODEL_NAME":        f"{model_name}_{s}",
            "DEEP_THINK_LLM":    deep,
            "QUICK_THINK_LLM":   quick,
            "BACKEND_URL":       OPENAI_API_URL,
            "LLM_PROVIDER":      "openai",
            "GRAPH_STRUCTURE":   s,
            "SELECTED_ANALYSTS": "market,social,news",
        }
        for s in structures
    ]

STRUCTURES  = ["linear", "centralized", "decentralized", "moa", "hybrid"]
EXPERIMENTS = make_exps("gpt-4.1", "gpt-4.1", "gpt-4.1-mini")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_experiment(exp: dict):
    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        exp["MODEL_NAME"],
        "DEEP_THINK_LLM":    exp["DEEP_THINK_LLM"],
        "QUICK_THINK_LLM":   exp["QUICK_THINK_LLM"],
        "BACKEND_URL":       exp["BACKEND_URL"],
        "LLM_PROVIDER":      exp.get("LLM_PROVIDER", "openai"),
        "GRAPH_STRUCTURE":   exp.get("GRAPH_STRUCTURE", "debate"),
        "SELECTED_ANALYSTS": exp.get("SELECTED_ANALYSTS", "market,social,news"),
        "START_DATE":        START_DATE,
        "END_DATE":          END_DATE,
        "MAX_BUDGET_USD":    "0",
    })
    log_path = os.path.join(SCRIPT_DIR, f"logs/{exp['MODEL_NAME']}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    print(f"  [{exp['MODEL_NAME']}] → OpenAI API")
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
        # .env에서 직접 읽기
        env_file = os.path.join(SCRIPT_DIR, ".env")
        if os.path.exists(env_file):
            for line in open(env_file):
                if line.startswith("OPENAI_API_KEY="):
                    key = line.strip().split("=", 1)[1]
                    os.environ["OPENAI_API_KEY"] = key
                    break
    return bool(key)


def monitor_progress(stop_event: threading.Event, start_time: datetime, experiments_list=None):
    """진행률을 터미널에 표시."""
    model_names = [e["MODEL_NAME"] for e in (experiments_list or EXPERIMENTS)]

    while not stop_event.is_set():
        lines = []
        lines.append("\033[2J\033[H")
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        lines.append(f"{'='*70}")
        lines.append(f"GPT EXPERIMENT PROGRESS  {datetime.now().strftime('%H:%M:%S')}  elapsed={elapsed:.1f}h")
        lines.append(f"Period: {START_DATE} ~ {END_DATE}")
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

            total_done  = sum(v.get("current", 0) for v in tickers.values())
            total_all   = total_dates * len(tickers)
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
        lines.append("로그: tail -f logs/<model>.log")
        print("\n".join(lines), flush=True)
        time.sleep(5)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPT API Experiment Runner (clean baseline)")
    parser.add_argument("--structure", choices=STRUCTURES + ["all"], default="all",
                        help="실행할 구조 (default: all)")
    args = parser.parse_args()

    experiments = EXPERIMENTS if args.structure == "all" else [
        e for e in EXPERIMENTS if e["GRAPH_STRUCTURE"] == args.structure
    ]

    print("=" * 60)
    print("GPT API Experiment Runner")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print(f"Models: {[e['MODEL_NAME'] for e in experiments]}")
    print("=" * 60)

    # API 키 확인
    print("\n[Step 1] Checking OpenAI API key...")
    if not check_openai_key():
        print("  ERROR: OPENAI_API_KEY not found in env or .env file")
        sys.exit(1)
    print("  OPENAI_API_KEY ✓")

    print(f"\n[Step 2] Starting experiments (순차 실행)...")
    for e in experiments:
        print(f"  {e['MODEL_NAME']} ({e['GRAPH_STRUCTURE']})")

    start_time = datetime.now()
    results    = {}

    # 터미널 진행률 모니터링
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_progress,
        args=(stop_monitor, start_time, experiments),
        daemon=True,
    )
    monitor_thread.start()

    # gpt-4.1 → gpt-4.1-mini 순차 실행
    for exp in experiments:
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
    print("\n" + "=" * 60)
    print(f"All GPT experiments finished.  Total elapsed: {elapsed:.1f}h")
    print(f"Results: trading_results/{START_DATE}_{END_DATE}/")
    for name, rc in results.items():
        status = "✓" if rc == 0 else f"✗ (rc={rc})"
        print(f"  {status} {name}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
