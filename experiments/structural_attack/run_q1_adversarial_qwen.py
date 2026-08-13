"""
Q1 2026 Adversarial Experiment Runner — Qwen3 (OpenRouter)
  (qwen/qwen3-235b-a22b-2507 / qwen/qwen3-30b-a3b-instruct-2507)

기간: 2026-01-01 ~ 2026-03-31
티커: BTC-USD, MSFT, NVDA, TSLA, AAPL (main_adversarial.py 하드코딩)
구조: linear, centralized, decentralized, hybrid (4개)
공격: D1(News content poisoning), D2(Social indirect injection), A1, A2s × BUY → 총 16 실험 (DEFAULT_ATTACKS 기준)

(D2는 원래 news_data에 숨긴 지시문을 삽입하는 방식이었는데, 그건 사실 News
Analyst 대상 data poisoning이지 진짜 프롬프트 인젝션이 아니었다. 지금은 social_data
(Reddit 포스트)에 실제 텍스트는 그대로 두고 "AI SYSTEM INSTRUCTION" 블록을 숨겨
넣는 방식으로 바뀌었고, config 플래그만 세팅하면 되는 query-time 방식이라 사전
파일 생성이 필요 없다 - graph_structure 어떤 걸 쓰든 동일하게 동작한다.)

Usage:
    tmux new-session -s q1_adv_qwen
    .venv/bin/python3 run_q1_adversarial_qwen.py --all

    .venv/bin/python3 run_q1_adversarial_qwen.py --all --parallel 4
    .venv/bin/python3 run_q1_adversarial_qwen.py --attack D1 --structure linear
    .venv/bin/python3 run_q1_adversarial_qwen.py --all --dry-run
    .venv/bin/python3 run_q1_adversarial_qwen.py --all --resume
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

DEEP_THINK_LLM  = "qwen/qwen3-235b-a22b-2507"
QUICK_THINK_LLM = "qwen/qwen3-30b-a3b-instruct-2507"
LLM_PROVIDER    = "openrouter"
BACKEND_URL     = "https://openrouter.ai/api/v1"

START_DATE        = "2026-01-01"
END_DATE          = "2026-03-31"
SELECTED_ANALYSTS = "market,social,news"
BASE_NEWS_SOURCE  = "alpha_vantage_filled"
TICKERS           = ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]

STRUCTURES      = ["linear", "centralized", "decentralized", "hybrid"]
SIGNALS         = ["BUY"]
DEFAULT_ATTACKS = ["D1", "D2", "A1", "A2s"]

ATTACK_CONFIGS = {
    "D1": {
        "attack_type": "news_text_poison",
        "poison_mode": "content",
        "poison_rate": "0.1",
        "description": "News Content Poisoning",
    },
    "D2": {
        "attack_type": "social_inject",
        "poison_rate": "0.1",
        "description": "Social Indirect Injection",
    },
    "A1": {
        "attack_type":     "analyst_compromise",
        "bias_style":      "subtle",
        "num_compromised": "1",
        "description":     "Single Analyst Compromise (subtle)",
    },
    "A1o": {
        "attack_type":     "analyst_compromise",
        "bias_style":      "overt",
        "num_compromised": "1",
        "description":     "Single Analyst Compromise (overt)",
    },
    "A2": {
        "attack_type":     "analyst_compromise",
        "bias_style":      "overt",
        "num_compromised": "2",
        "description":     "Multi-Analyst Compromise (overt)",
    },
    "A2s": {
        "attack_type":     "analyst_compromise",
        "bias_style":      "subtle",
        "num_compromised": "2",
        "description":     "Multi-Analyst Compromise (subtle)",
    },
}

NEWS_ATTACKS = {"D1"}  # D2 is now social_inject - config-flag driven, no pregen needed


def check_openrouter_key() -> bool:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        env_file = SCRIPT_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    os.environ["OPENROUTER_API_KEY"] = key
                    break
    return bool(key)


def pregen_news(signal: str, mode: str, poison_rate: float = 1.0):
    local_data_dir = str(SCRIPT_DIR / "tradingagents/dataflows/local_data")
    print(f"  [Pregen] signal={signal} mode={mode} → generating poisoned news...")
    src = prepare_text_poisoned_news(
        tickers=TICKERS,
        target_signal=signal,
        poison_mode=mode,
        poison_rate=poison_rate,
        src_news_source=BASE_NEWS_SOURCE,
        local_data_dir=local_data_dir,
    )
    print(f"  [Pregen] done → {src}")
    return src


def experiment_name(attack_id: str, signal: str, structure: str, poison_rate: str = None) -> str:
    base = f"adv_qwen_q1_{attack_id}_{signal.lower()}_{structure}"
    default_rate = ATTACK_CONFIGS.get(attack_id, {}).get("poison_rate")
    if poison_rate is not None and poison_rate != default_rate:
        base += f"_ratio{int(round(float(poison_rate) * 100))}"
    return base


def run_single_experiment(
    attack_id: str, signal: str, structure: str,
    dry_run: bool = False, resume: bool = False,
    skip_news_regen: bool = False, poison_rate: str = None,
) -> tuple[str, int]:
    name = experiment_name(attack_id, signal, structure, poison_rate)
    cfg  = ATTACK_CONFIGS[attack_id]

    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        name,
        "DEEP_THINK_LLM":    DEEP_THINK_LLM,
        "QUICK_THINK_LLM":   QUICK_THINK_LLM,
        "LLM_PROVIDER":      LLM_PROVIDER,
        "BACKEND_URL":       BACKEND_URL,
        "GRAPH_STRUCTURE":   structure,
        "SELECTED_ANALYSTS": SELECTED_ANALYSTS,
        "START_DATE":        START_DATE,
        "END_DATE":          END_DATE,
        "TICKERS":           ",".join(TICKERS),
        "ATTACK_ENABLED":    "true",
        "ATTACK_TYPE":       cfg["attack_type"],
        "ATTACK_SIGNAL":     signal,
        "ATTACK_ID":         attack_id,
        "RESUME":            "true" if resume else "false",
        "SKIP_NEWS_REGEN":   "true" if skip_news_regen else "false",
        "MAX_BUDGET_USD":    "0",
    })

    if "poison_mode" in cfg:
        env["ATTACK_POISON_MODE"] = cfg["poison_mode"]
    if "poison_rate" in cfg:
        env["ATTACK_POISON_RATE"] = poison_rate if poison_rate is not None else cfg["poison_rate"]
    if "bias_style" in cfg:
        env["ATTACK_BIAS_STYLE"]      = cfg["bias_style"]
        env["ATTACK_NUM_COMPROMISED"] = cfg.get("num_compromised", "1")

    log_path = SCRIPT_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [{name}] {cfg['description']} target={signal} structure={structure}")

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
        lines = ["\033[2J\033[H"]  # 화면 지우기
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        done    = len(results)
        total   = len(experiment_names)

        lines.append("=" * 72)
        lines.append(f"Q1 2026 QWEN3 ADVERSARIAL  |  {datetime.now().strftime('%H:%M:%S')}  elapsed={elapsed:.1f}h")
        lines.append(f"완료: {done}/{total}  |  deep={DEEP_THINK_LLM}  quick={QUICK_THINK_LLM}")
        lines.append("=" * 72)

        for name in experiment_names:
            rc = results.get(name)
            if rc is not None:
                icon = "✓" if rc == 0 else "✗"
                lines.append(f"  {icon} {name}")
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
            updated_at  = data.get("updated_at", "")

            total_done = sum(v.get("current", 0) for v in tickers.values())
            total_all  = total_dates * len(tickers)
            pct        = total_done / total_all * 100 if total_all else 0
            bar_w      = int(20 * total_done / total_all) if total_all else 0
            bar        = "█" * bar_w + "░" * (20 - bar_w)

            lines.append(f"  ⏳ {name}  [{bar}] {total_done}/{total_all} ({pct:.0f}%)  {updated_at}")
            for ticker, info in tickers.items():
                cur    = info.get("current", 0)
                tot    = info.get("total", total_dates)
                status = info.get("status", "waiting")
                last   = info.get("last_date", "")
                t_pct  = cur / tot * 100 if tot else 0
                t_bar  = int(15 * cur / tot) if tot else 0
                if status == "completed":
                    icon, color = "✓", "\033[92m"
                elif status == "running":
                    icon, color = "⏳", "\033[93m"
                else:
                    icon, color = "○", "\033[90m"
                lines.append(f"     {color}{icon} {ticker:7s} [{'█'*t_bar+'░'*(15-t_bar)}] {cur:3d}/{tot}  ({t_pct:5.1f}%)  {last}\033[0m")

        # 완료 요약
        ok  = sum(1 for rc in results.values() if rc == 0)
        err = sum(1 for rc in results.values() if rc != 0)
        lines.append("=" * 72)
        lines.append(f"완료 ✓{ok}  실패 ✗{err}  진행 중 {total - done}")
        lines.append("로그: logs/adv_qwen_q1_<attack>_<signal>_<struct>.log")

        print("\n".join(lines), flush=True)
        time.sleep(5)


def build_experiment_list(attacks=None, signals=None, structures=None):
    attacks    = attacks    or DEFAULT_ATTACKS
    signals    = signals    or SIGNALS
    structures = structures or STRUCTURES
    return [
        (attack, signal, structure)
        for attack    in attacks
        for signal    in signals
        for structure in structures
    ]


def main():
    parser = argparse.ArgumentParser(description="Q1 2026 Adversarial Experiment Runner (Qwen3 / OpenRouter)")
    parser.add_argument("--all",       action="store_true", help="Run all 16 experiments (4 attacks × 4 structures)")
    parser.add_argument("--attack",    choices=list(ATTACK_CONFIGS.keys()), help="Attack ID")
    parser.add_argument("--signal",    choices=["BUY", "SELL"], default="BUY")
    parser.add_argument("--structure", choices=STRUCTURES + ["all"], default="all")
    parser.add_argument("--parallel",  type=int, default=6, help="동시 실행 수 (default: 6)")
    parser.add_argument("--poison-rate", type=str, default=None,
                         help="D1/D2 poison_rate override (0-1, e.g. 0.4). Only affects D1/D2 - "
                              "ignored for A1/A1o/A2/A2s. Output folder gets a _ratioNN suffix when "
                              "this differs from the attack's own default, so it never collides with "
                              "existing adv_qwen_q1_D1_.../adv_qwen_q1_D2_... results.")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--resume",    action="store_true", help="Skip already-succeeded dates")
    args = parser.parse_args()

    structures = STRUCTURES if args.structure == "all" else [args.structure]

    if args.all:
        experiments = build_experiment_list(structures=structures, signals=[args.signal])
    elif args.attack:
        experiments = [(args.attack, args.signal, s) for s in structures]
    else:
        parser.print_help()
        sys.exit(0)

    print("=" * 70)
    print("Q1 2026 ADVERSARIAL EXPERIMENT RUNNER — Qwen3 (OpenRouter)")
    print(f"Period:   {START_DATE} ~ {END_DATE}")
    print(f"Model:    deep={DEEP_THINK_LLM}  quick={QUICK_THINK_LLM}")
    print(f"Attacks:  {list(ATTACK_CONFIGS.keys())}  Signal: {args.signal}")
    print(f"Parallel: {args.parallel} 개 동시 실행")
    print(f"Total:    {len(experiments)} experiments")
    print("=" * 70)

    if not args.dry_run:
        print("\n[Step 1] Checking OpenRouter API key...")
        if not check_openrouter_key():
            print("  ERROR: OPENROUTER_API_KEY not found in env or .env file")
            sys.exit(1)
        print("  OPENROUTER_API_KEY ✓")

    news_attacks_in_plan = {atk for atk, _, _ in experiments if atk in NEWS_ATTACKS}
    if news_attacks_in_plan and not args.dry_run:
        print("\n[Step 2] Pre-generating poisoned news files (1회)...")
        if "D1" in news_attacks_in_plan:
            # Bug fixed: this used to always pregen at the function's own
            # default (1.0) regardless of ATTACK_CONFIGS["D1"]["poison_rate"]
            # or --poison-rate, since neither was ever passed through - so
            # SKIP_NEWS_REGEN runs were silently reading a 100%-poisoned file
            # no matter what rate was configured. Now uses the same override
            # precedence as run_single_experiment (CLI > ATTACK_CONFIGS).
            d1_rate = float(args.poison_rate if args.poison_rate is not None
                            else ATTACK_CONFIGS["D1"]["poison_rate"])
            pregen_news(args.signal, "content", poison_rate=d1_rate)
        print("  Poisoned news ready ✓\n")
    else:
        print("\n[Step 2] Skipped (dry-run or no D1)\n")

    print(f"[Step 3] Launching {len(experiments)} experiments ({args.parallel} parallel)...")
    start_time     = datetime.now()
    results        = {}
    exp_names      = [experiment_name(a, s, st, args.poison_rate) for a, s, st in experiments]

    stop_monitor = threading.Event()
    if not args.dry_run:
        monitor_thread = threading.Thread(
            target=monitor_progress,
            args=(stop_monitor, start_time, exp_names, results),
            daemon=True,
        )
        monitor_thread.start()

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                run_single_experiment,
                attack, signal, structure,
                dry_run=args.dry_run,
                resume=args.resume,
                skip_news_regen=(attack in NEWS_ATTACKS),
                poison_rate=args.poison_rate,
            ): experiment_name(attack, signal, structure, args.poison_rate)
            for attack, signal, structure in experiments
        }
        for future in as_completed(futures):
            name, rc = future.result()
            results[name] = rc

    if not args.dry_run:
        stop_monitor.set()

    total_elapsed = (datetime.now() - start_time).total_seconds() / 3600
    print("\n" + "=" * 70)
    print(f"All Q1 Qwen3 adversarial experiments finished.  Total: {total_elapsed:.1f}h")
    print(f"Results: trading_results/{START_DATE}_{END_DATE}/")
    failed = [(n, rc) for n, rc in results.items() if rc != 0]
    print(f"  ✓ {len(results) - len(failed)} succeeded  ✗ {len(failed)} failed")
    for name, rc in failed:
        print(f"    ✗ {name} (rc={rc})")
    print("=" * 70)


if __name__ == "__main__":
    main()
