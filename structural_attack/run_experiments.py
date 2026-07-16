"""
Multi-model experiment runner using 2 DGX Sparks.

Spark 1 (localhost:11434)   → linear      → centralized   (순차)
Spark 2 (169.254.30.132)    → decentralized → moa          (순차)
두 Spark는 서로 병렬 실행.
"""

import subprocess
import os
import sys
import json
import threading
import time
from datetime import datetime

SPARK2_HOST = "169.254.30.132"
SPARK2_OLLAMA_URL = f"http://{SPARK2_HOST}:11434/v1"
SPARK1_OLLAMA_URL = "http://localhost:11434/v1"

START_DATE = "2026-01-01"
END_DATE   = "2026-03-31"

###############################################################################
# Experiment Set A: Model Backbone Variation (same structure = debate)
###############################################################################
MODEL_EXPERIMENTS = [
    {
        "MODEL_NAME":      "qwen2.5_72b",
        "DEEP_THINK_LLM":  "qwen2.5:72b",
        "QUICK_THINK_LLM": "qwen2.5:14b",
        "BACKEND_URL":     SPARK1_OLLAMA_URL,
        "GRAPH_STRUCTURE": "debate",
        "spark":           1,
    },
    {
        "MODEL_NAME":      "deepseek_r1_70b",
        "DEEP_THINK_LLM":  "deepseek-r1:70b",
        "QUICK_THINK_LLM": "qwen2.5:14b",
        "BACKEND_URL":     SPARK2_OLLAMA_URL,
        "GRAPH_STRUCTURE": "debate",
        "spark":           2,
    },
    {
        "MODEL_NAME":      "llama3.3_70b",
        "DEEP_THINK_LLM":  "llama3.3:70b",
        "QUICK_THINK_LLM": "qwen2.5:14b",
        "BACKEND_URL":     SPARK2_OLLAMA_URL,
        "GRAPH_STRUCTURE": "debate",
        "spark":           2,
    },
]

###############################################################################
# Experiment Set B: Graph Structure Variation (same model = qwen2.5:72b)
# Tickers: BTC-USD, NVDA, TSLA, AAPL, MSFT
# Analysts: market, social, news (BTC-USD 호환을 위해 fundamentals 제외)
###############################################################################
STRUCTURE_EXPERIMENTS = [
    {
        "MODEL_NAME":        "linear",
        "DEEP_THINK_LLM":    "qwen2.5:72b",
        "QUICK_THINK_LLM":   "qwen2.5:14b",
        "BACKEND_URL":       SPARK1_OLLAMA_URL,
        "GRAPH_STRUCTURE":   "linear",
        "SELECTED_ANALYSTS": "market,social,news",
        "spark":             1,
    },
    {
        "MODEL_NAME":        "centralized",
        "DEEP_THINK_LLM":    "qwen2.5:72b",
        "QUICK_THINK_LLM":   "qwen2.5:14b",
        "BACKEND_URL":       SPARK1_OLLAMA_URL,
        "GRAPH_STRUCTURE":   "centralized",
        "SELECTED_ANALYSTS": "market,social,news",
        "spark":             1,
    },
    {
        "MODEL_NAME":        "decentralized",
        "DEEP_THINK_LLM":    "qwen2.5:72b",
        "QUICK_THINK_LLM":   "qwen2.5:14b",
        "BACKEND_URL":       SPARK2_OLLAMA_URL,
        "GRAPH_STRUCTURE":   "decentralized",
        "SELECTED_ANALYSTS": "market,social,news",
        "spark":             2,
    },
    {
        "MODEL_NAME":        "moa",
        "DEEP_THINK_LLM":    "qwen2.5:72b",
        "QUICK_THINK_LLM":   "qwen2.5:14b",
        "BACKEND_URL":       SPARK2_OLLAMA_URL,
        "GRAPH_STRUCTURE":   "moa",
        "SELECTED_ANALYSTS": "market,social,news",
        "spark":             2,
    },
]

# Active experiment set — change this to switch
EXPERIMENTS = STRUCTURE_EXPERIMENTS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_experiment(exp: dict) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        exp["MODEL_NAME"],
        "DEEP_THINK_LLM":    exp["DEEP_THINK_LLM"],
        "QUICK_THINK_LLM":   exp["QUICK_THINK_LLM"],
        "BACKEND_URL":       exp["BACKEND_URL"],
        "GRAPH_STRUCTURE":   exp.get("GRAPH_STRUCTURE", "linear"),
        "SELECTED_ANALYSTS": exp.get("SELECTED_ANALYSTS", "market,social,news"),
        "START_DATE":        START_DATE,
        "END_DATE":          END_DATE,
    })
    log_path = os.path.join(SCRIPT_DIR, f"logs/{exp['MODEL_NAME']}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    print(f"  [{exp['MODEL_NAME']}] → Spark {exp['spark']} ({exp['BACKEND_URL']})")
    print(f"  [{exp['MODEL_NAME']}] log: {log_path}")
    proc = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPT_DIR, "main.py")],
        env=env,
        cwd=SCRIPT_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc, log_file


def check_ollama(url: str) -> bool:
    import urllib.request
    try:
        req = urllib.request.Request(url.replace("/v1", "/api/tags"))
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def monitor_progress(experiments: list, stop_event: threading.Event, start_time: datetime):
    """모든 실험의 진행률을 터미널에 통합 표시."""
    model_names = [e["MODEL_NAME"] for e in experiments]

    while not stop_event.is_set():
        lines = []
        lines.append(f"\033[2J\033[H")  # clear screen
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        lines.append(f"{'='*70}")
        lines.append(f"EXPERIMENT PROGRESS  {datetime.now().strftime('%H:%M:%S')}  elapsed={elapsed:.1f}h")
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

            structure  = data.get("structure", "")
            updated_at = data.get("updated_at", "")
            total_dates = data.get("total_dates", 1)
            tickers = data.get("tickers", {})

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

            bar_w  = int(30 * total_done / total_all) if total_all else 0
            o_bar  = "█" * bar_w + "░" * (30 - bar_w)
            lines.append(f"  Overall [{o_bar}] {total_done}/{total_all} ({overall_pct:.1f}%)")

        lines.append(f"\n{'='*70}")
        lines.append("로그: tail -f logs/<model>.log")
        print("\n".join(lines), flush=True)
        time.sleep(5)


def run_spark_sequential(exps: list, spark_id: int, start_time: datetime, results: dict):
    """한 Spark에 할당된 실험들을 순차 실행. 별도 스레드에서 호출."""
    for exp in exps:
        proc, log = run_experiment(exp)
        print(f"  [{exp['MODEL_NAME']}] PID={proc.pid} started (Spark {spark_id})\n")
        proc.wait()
        log.close()
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        rc = proc.returncode
        status = "✓ done" if rc == 0 else f"✗ failed (rc={rc})"
        print(f"\n  [Spark {spark_id}] [{exp['MODEL_NAME']}] {status}  elapsed={elapsed:.1f}h")
        results[exp["MODEL_NAME"]] = rc


def main():
    print("=" * 60)
    print(f"Multi-model Experiment Runner")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print(f"Experiments: {[e['MODEL_NAME'] for e in EXPERIMENTS]}")
    print("=" * 60)

    # Ollama 연결 확인
    print("\n[Step 1] Checking Ollama connections...")
    if not check_ollama(SPARK1_OLLAMA_URL):
        print(f"  ERROR: Spark 1 Ollama not reachable at {SPARK1_OLLAMA_URL}")
        sys.exit(1)
    print(f"  Spark 1 ({SPARK1_OLLAMA_URL}) ✓")

    if not check_ollama(SPARK2_OLLAMA_URL):
        print(f"  ERROR: Spark 2 Ollama not reachable at {SPARK2_OLLAMA_URL}")
        print(f"  Hint: SSH to {SPARK2_HOST} and run:")
        print(f"    OLLAMA_HOST=0.0.0.0 OLLAMA_MAX_LOADED_MODELS=1 ollama serve &")
        sys.exit(1)
    print(f"  Spark 2 ({SPARK2_OLLAMA_URL}) ✓")

    # Spark별 실험 분리
    spark1_exps = [e for e in EXPERIMENTS if e["spark"] == 1]
    spark2_exps = [e for e in EXPERIMENTS if e["spark"] == 2]

    print(f"\n[Step 2] Starting experiments...")
    print(f"  Spark 1: {[e['MODEL_NAME'] for e in spark1_exps]} (순차)")
    print(f"  Spark 2: {[e['MODEL_NAME'] for e in spark2_exps]} (순차)")
    print(f"  두 Spark는 병렬 실행\n")

    start_time = datetime.now()
    results = {}

    # 터미널 진행률 모니터링 스레드
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_progress,
        args=(EXPERIMENTS, stop_monitor, start_time),
        daemon=True,
    )
    monitor_thread.start()

    # 각 Spark를 별도 스레드에서 순차 실행 → 두 Spark는 병렬
    t1 = threading.Thread(target=run_spark_sequential, args=(spark1_exps, 1, start_time, results))
    t2 = threading.Thread(target=run_spark_sequential, args=(spark2_exps, 2, start_time, results))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

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
