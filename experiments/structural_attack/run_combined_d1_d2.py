"""
Combined D1+D2 Attack Runner — News Content Poisoning + Social Indirect
Injection applied SIMULTANEOUSLY, multi-provider (gemini / gpt / qwen).

D1 (news_text_poison) and D2 (social_inject) are independent, orthogonal,
config-flag-driven mechanisms that each touch a different data source
(news_source vs. social poisoning flags) - see main_adversarial.py's D1/D2
blocks. Passing ATTACK_TYPE="news_text_poison,social_inject" (now supported
by main_adversarial.py) simply turns both blocks on in the same run, so the
News Analyst reads poisoned/fabricated articles AND the Social Analyst reads
posts with a hidden adversarial instruction injected, at the same time, in
the same trading-day decision.

Attack ID: D1D2. Structures: the 4 real topologies (decentralized, hybrid,
centralized, linear) - no null/single-agent variant, since the point of this
combo is to see how structure aggregates TWO simultaneous compromised data
sources, not to isolate one analyst. Signals: BUY and SELL (both are genuine
same-quarter campaigns, matching every other combined/attack table in this
project - see table_struct_main_2026Q1.tex).

Usage:
    .venv/bin/python3 run_combined_d1_d2.py --provider gpt  --all
    .venv/bin/python3 run_combined_d1_d2.py --provider qwen --all --parallel 4
    .venv/bin/python3 run_combined_d1_d2.py --provider gpt  --signal SELL --structure linear
    .venv/bin/python3 run_combined_d1_d2.py --provider gpt  --all --dry-run
    .venv/bin/python3 run_combined_d1_d2.py --provider gpt  --all --resume
"""

import subprocess
import os
import sys
import argparse
import threading
import time
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tradingagents.agents.attacks.attack_news_text_poisoning import prepare_text_poisoned_news

SCRIPT_DIR = Path(__file__).parent.resolve()

START_DATE        = "2026-01-01"
END_DATE          = "2026-03-31"
BASE_NEWS_SOURCE  = "alpha_vantage_filled"
SELECTED_ANALYSTS = "market,social,news"
TICKERS           = ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]

STRUCTURES = ["decentralized", "hybrid", "centralized", "linear"]
SIGNALS    = ["BUY", "SELL"]

ATTACK_ID   = "D1D2"
ATTACK_TYPE = "news_text_poison,social_inject"
POISON_RATE = "0.1"   # shared by D1 and D2

PROVIDER_CONFIGS = {
    "gemini": {
        "prefix":       "gemini",
        "deep":         "google/gemini-2.5-flash",
        "quick":        "google/gemini-2.5-flash-lite",
        "llm_provider": "openrouter",
        "backend_url":  "https://openrouter.ai/api/v1",
        "key_env":      "OPENROUTER_API_KEY",
    },
    "gpt": {
        "prefix":       "gpt",
        "deep":         "gpt-4.1",
        "quick":        "gpt-4.1-mini",
        "llm_provider": "openai",
        "backend_url":  "https://api.openai.com/v1",
        "key_env":      "OPENAI_API_KEY",
    },
    "qwen": {
        "prefix":       "qwen",
        "deep":         "qwen/qwen3-235b-a22b-2507",
        "quick":        "qwen/qwen3-30b-a3b-instruct-2507",
        "llm_provider": "openrouter",
        "backend_url":  "https://openrouter.ai/api/v1",
        "key_env":      "OPENROUTER_API_KEY",
    },
}


def check_api_key(cfg: dict) -> bool:
    key_env = cfg["key_env"]
    key = os.environ.get(key_env, "")
    if not key:
        env_file = SCRIPT_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith(f"{key_env}="):
                    key = line.split("=", 1)[1].strip()
                    os.environ[key_env] = key
                    break
    return bool(key)


def pregen_news(signal: str, poison_rate: float = 1.0) -> str:
    """D1's half of the combo needs a poisoned-news file generated once up
    front (shared across all tickers/workers) - same as every other D1 run
    in this project. D2 needs no file; it poisons social data live per-query."""
    local_data_dir = str(SCRIPT_DIR / "tradingagents/dataflows/local_data")
    print(f"  [Pregen] signal={signal} -> generating poisoned news (D1 half of D1D2)...")
    src = prepare_text_poisoned_news(
        tickers=TICKERS,
        target_signal=signal,
        poison_mode="content",
        poison_rate=poison_rate,
        src_news_source=BASE_NEWS_SOURCE,
        local_data_dir=local_data_dir,
    )
    print(f"  [Pregen] done -> {src}")
    return src


def experiment_name(prefix: str, signal: str, structure: str) -> str:
    return f"adv_{prefix}_{ATTACK_ID}_{signal.lower()}_{structure}"


def run_single_experiment(
    provider_cfg: dict, signal: str, structure: str,
    dry_run: bool = False, resume: bool = False,
) -> tuple[str, int]:
    name = experiment_name(provider_cfg["prefix"], signal, structure)

    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        name,
        "DEEP_THINK_LLM":    provider_cfg["deep"],
        "QUICK_THINK_LLM":   provider_cfg["quick"],
        "LLM_PROVIDER":      provider_cfg["llm_provider"],
        "GRAPH_STRUCTURE":   structure,
        "SELECTED_ANALYSTS": SELECTED_ANALYSTS,
        "START_DATE":        START_DATE,
        "END_DATE":          END_DATE,
        "TICKERS":           ",".join(TICKERS),
        "ATTACK_ENABLED":    "true",
        "ATTACK_TYPE":       ATTACK_TYPE,
        "ATTACK_SIGNAL":     signal,
        "ATTACK_ID":         ATTACK_ID,
        "ATTACK_POISON_MODE": "content",
        "ATTACK_POISON_RATE": POISON_RATE,
        "SKIP_NEWS_REGEN":   "true",   # pre-generated once in main() before dispatch
        "RESUME":            "true" if resume else "false",
        "MAX_BUDGET_USD":    "0",
    })
    if provider_cfg["backend_url"]:
        env["BACKEND_URL"] = provider_cfg["backend_url"]

    log_path = SCRIPT_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [{name}] D1+D2 combined  target={signal}  structure={structure}")

    if dry_run:
        print(f"  [{name}] DRY RUN")
        return name, 0

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT_DIR / "main_adversarial.py")],
            env=env,
            cwd=str(SCRIPT_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        proc.wait()

    rc = proc.returncode
    print(f"  [{name}] {'✓ done' if rc == 0 else f'✗ failed (rc={rc})'}")
    return name, rc


def monitor_progress(stop_event: threading.Event, start_time: datetime,
                      experiment_names: list[str], results: dict):
    log_dir = SCRIPT_DIR / "logs"
    while not stop_event.is_set():
        lines = ["\033[2J\033[H"]
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        done, total = len(results), len(experiment_names)

        lines.append("=" * 72)
        lines.append(f"D1+D2 COMBINED RUNNER  |  {datetime.now().strftime('%H:%M:%S')}  elapsed={elapsed:.1f}h")
        lines.append(f"완료: {done}/{total}")
        lines.append("=" * 72)

        for name in experiment_names:
            rc = results.get(name)
            if rc is not None:
                lines.append(f"  {'✓' if rc == 0 else '✗'} {name}")
                continue
            pfile = log_dir / f".progress_{name}.json"
            if not pfile.exists():
                lines.append(f"  ⏳ {name}  (대기 중)")
                continue
            try:
                data = json.loads(pfile.read_text())
            except Exception:
                lines.append(f"  ⏳ {name}  (읽기 오류)")
                continue
            tickers     = data.get("tickers", {})
            total_dates = data.get("total_dates", 1)
            total_done  = sum(v.get("current", 0) for v in tickers.values())
            total_all   = total_dates * len(tickers)
            pct         = total_done / total_all * 100 if total_all else 0
            bar_w       = int(20 * total_done / total_all) if total_all else 0
            lines.append(f"  ⏳ {name}  [{'█'*bar_w+'░'*(20-bar_w)}] {total_done}/{total_all} ({pct:.0f}%)")

        ok  = sum(1 for rc in results.values() if rc == 0)
        err = sum(1 for rc in results.values() if rc != 0)
        lines.append("=" * 72)
        lines.append(f"완료 ✓{ok}  실패 ✗{err}  진행 중 {total - done}")
        lines.append("로그: logs/adv_<provider>_D1D2_<signal>_<structure>.log")
        print("\n".join(lines), flush=True)
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Combined D1+D2 (news poisoning + social injection) runner")
    parser.add_argument("--provider",  choices=list(PROVIDER_CONFIGS.keys()), required=True)
    parser.add_argument("--all",       action="store_true", help="Run all 8 experiments (4 structures x 2 signals)")
    parser.add_argument("--signal",    choices=SIGNALS, help="Run only this signal (both structures unless --structure also given)")
    parser.add_argument("--structure", choices=STRUCTURES, help="Run only this structure")
    parser.add_argument("--parallel",  type=int, default=3, help="동시 실행 수 (default: 3)")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--resume",    action="store_true")
    args = parser.parse_args()

    provider_cfg = PROVIDER_CONFIGS[args.provider]

    if args.all:
        experiments = [(s, t) for s in SIGNALS for t in STRUCTURES]
    elif args.signal or args.structure:
        signals    = [args.signal] if args.signal else SIGNALS
        structures = [args.structure] if args.structure else STRUCTURES
        experiments = [(s, t) for s in signals for t in structures]
    else:
        parser.print_help()
        sys.exit(0)

    print("=" * 70)
    print(f"D1+D2 COMBINED RUNNER — {args.provider}")
    print(f"Period:     {START_DATE} ~ {END_DATE}")
    print(f"Model:      deep={provider_cfg['deep']}  quick={provider_cfg['quick']}")
    print(f"Analysts:   {SELECTED_ANALYSTS}")
    print(f"Attack:     {ATTACK_ID} = {ATTACK_TYPE}  (poison_rate={POISON_RATE}, shared)")
    print(f"Total:      {len(experiments)} experiments")
    print("=" * 70)

    if not args.dry_run:
        print(f"\n[Step 1] Checking {provider_cfg['key_env']}...")
        if not check_api_key(provider_cfg):
            print(f"  ERROR: {provider_cfg['key_env']} not found in env or .env file")
            sys.exit(1)
        print(f"  {provider_cfg['key_env']} ✓")

    signals_in_plan = sorted({s for s, _ in experiments})
    if not args.dry_run:
        print(f"\n[Step 2] Pre-generating poisoned news files (D1 half) for signal(s) {signals_in_plan}...")
        for s in signals_in_plan:
            pregen_news(s, float(POISON_RATE))
        print("  Poisoned news ready ✓\n")
    else:
        print("\n[Step 2] Skipped (dry-run)\n")

    print(f"[Step 3] Launching {len(experiments)} experiments ({args.parallel} parallel)...")
    start_time = datetime.now()
    results: dict[str, int] = {}
    exp_names = [experiment_name(provider_cfg["prefix"], s, t) for s, t in experiments]

    stop_monitor = threading.Event()
    if not args.dry_run:
        monitor_thread = threading.Thread(
            target=monitor_progress, args=(stop_monitor, start_time, exp_names, results), daemon=True,
        )
        monitor_thread.start()

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                run_single_experiment,
                provider_cfg, signal, structure,
                dry_run=args.dry_run,
                resume=args.resume,
            ): experiment_name(provider_cfg["prefix"], signal, structure)
            for signal, structure in experiments
        }
        for future in as_completed(futures):
            name, rc = future.result()
            results[name] = rc

    if not args.dry_run:
        stop_monitor.set()

    total_elapsed = (datetime.now() - start_time).total_seconds() / 3600
    print("\n" + "=" * 70)
    print(f"D1+D2 combined runner finished ({args.provider}).  Total: {total_elapsed:.1f}h")
    print(f"Results: trading_results/{START_DATE}_{END_DATE}/adv_{provider_cfg['prefix']}_D1D2_<signal>_<structure>/")
    failed = [(n, rc) for n, rc in results.items() if rc != 0]
    print(f"  ✓ {len(results) - len(failed)} succeeded  ✗ {len(failed)} failed")
    for name, rc in failed:
        print(f"    ✗ {name} (rc={rc})")
    print("=" * 70)


if __name__ == "__main__":
    main()
