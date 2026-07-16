"""
Q3 2025 Adversarial Experiment Runner — JPM / GLD  ×  GPT API
  (gpt-4.1 / gpt-4.1-mini)

기간: 2025-07-01 ~ 2025-09-30
티커: JPM, GLD
구조: linear, centralized, decentralized, hybrid  (4개)
공격: D1, D2, A1, A1o, A2, A2s  ×  SELL  →  총 24 실험

  D1  : News Content Poisoning
  D2  : Indirect Prompt Injection
  A1  : Single Analyst Compromise (subtle)
  A1o : Single Analyst Compromise (overt)
  A2  : Multi-Analyst Compromise (overt,  2개)
  A2s : Multi-Analyst Compromise (subtle, 2개)

Usage:
    tmux new-session -s q3_jpmgld
    .venv/bin/python run_q3_jpmgld_gpt.py --all

    .venv/bin/python run_q3_jpmgld_gpt.py --all --parallel 4
    .venv/bin/python run_q3_jpmgld_gpt.py --attack D1 --structure linear
    .venv/bin/python run_q3_jpmgld_gpt.py --all --dry-run
    .venv/bin/python run_q3_jpmgld_gpt.py --all --resume
"""

import subprocess
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tradingagents.agents.attacks.attack_news_text_poisoning import prepare_text_poisoned_news

SCRIPT_DIR = Path(__file__).parent.resolve()

OPENAI_API_URL  = "https://api.openai.com/v1"
DEEP_THINK_LLM  = "gpt-4.1"
QUICK_THINK_LLM = "gpt-4.1-mini"
LLM_PROVIDER    = "openai"

START_DATE        = "2025-07-01"
END_DATE          = "2025-09-30"
SELECTED_ANALYSTS = "market,social,news"
BASE_NEWS_SOURCE  = "alpha_vantage_filled"
TICKERS           = ["JPM", "GLD"]

STRUCTURES = ["linear", "centralized", "decentralized", "hybrid"]
SIGNALS    = ["SELL"]

ATTACK_CONFIGS = {
    "D1": {
        "attack_type": "news_text_poison",
        "poison_mode": "content",
        "poison_rate": "1.0",
        "description": "News Content Poisoning",
    },
    "D2": {
        "attack_type": "news_inject",
        "poison_mode": "inject",
        "poison_rate": "1.0",
        "description": "Indirect Prompt Injection",
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

NEWS_ATTACKS = {"D1", "D2"}


def check_openai_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        env_file = SCRIPT_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    os.environ["OPENAI_API_KEY"] = key
                    break
    return bool(key)


def pregen_news(signal: str, mode: str, poison_rate: float = 1.0):
    local_data_dir = str(SCRIPT_DIR / "tradingagents/dataflows/local_data")
    print(f"  [Pregen] signal={signal} mode={mode} → generating poisoned news ...")
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


def experiment_name(attack_id: str, signal: str, structure: str) -> str:
    return f"adv_gpt_q3_jpmgld_{attack_id}_{signal.lower()}_{structure}"


def run_single_experiment(
    attack_id: str, signal: str, structure: str,
    dry_run: bool = False, resume: bool = False,
    skip_news_regen: bool = False,
) -> tuple[str, int]:
    name = experiment_name(attack_id, signal, structure)
    cfg  = ATTACK_CONFIGS[attack_id]

    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        name,
        "DEEP_THINK_LLM":    DEEP_THINK_LLM,
        "QUICK_THINK_LLM":   QUICK_THINK_LLM,
        "BACKEND_URL":       OPENAI_API_URL,
        "LLM_PROVIDER":      LLM_PROVIDER,
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
        env["ATTACK_POISON_RATE"] = cfg.get("poison_rate", "1.0")
    if "bias_style" in cfg:
        env["ATTACK_BIAS_STYLE"]      = cfg["bias_style"]
        env["ATTACK_NUM_COMPROMISED"] = cfg.get("num_compromised", "1")

    log_path = SCRIPT_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [{name}] {cfg['description']}  target={signal}  structure={structure}")

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


def build_experiment_list(attacks=None, signals=None, structures=None):
    attacks    = attacks    or list(ATTACK_CONFIGS.keys())
    signals    = signals    or SIGNALS
    structures = structures or STRUCTURES
    return [
        (attack, signal, structure)
        for attack    in attacks
        for signal    in signals
        for structure in structures
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Q3 2025 Adversarial Runner — JPM/GLD × GPT"
    )
    parser.add_argument("--all",       action="store_true", help="Run all 24 experiments")
    parser.add_argument("--attack",    choices=list(ATTACK_CONFIGS.keys()),
                        nargs="+", help="Attack ID (복수 지정 가능, e.g. --attack D1 A1o A2)")
    parser.add_argument("--signal",    choices=["SELL"], default="SELL")
    parser.add_argument("--structure", choices=STRUCTURES + ["all"], default="all")
    parser.add_argument("--parallel",  type=int, default=4,
                        help="동시 실행 수 (default: 4)")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--resume",    action="store_true", help="Skip already-succeeded dates")
    args = parser.parse_args()

    structures = STRUCTURES if args.structure == "all" else [args.structure]

    if args.all:
        experiments = build_experiment_list(structures=structures)
    elif args.attack:
        experiments = build_experiment_list(attacks=args.attack, structures=structures)
    else:
        parser.print_help()
        sys.exit(0)

    print("=" * 70)
    print("Q3 2025 ADVERSARIAL RUNNER — JPM / GLD  ×  GPT")
    print(f"Period  : {START_DATE} ~ {END_DATE}")
    print(f"Tickers : {TICKERS}")
    print(f"Model   : deep={DEEP_THINK_LLM}  quick={QUICK_THINK_LLM}")
    print(f"Attacks : {list(ATTACK_CONFIGS.keys())}  Signal: SELL")
    print(f"Parallel: {args.parallel} 개 동시 실행")
    print(f"Total   : {len(experiments)} experiments")
    print("=" * 70)

    if not args.dry_run:
        print("\n[Step 1] Checking OpenAI API key ...")
        if not check_openai_key():
            print("  ERROR: OPENAI_API_KEY not found in env or .env")
            sys.exit(1)
        print("  OPENAI_API_KEY ✓")

    news_attacks_in_plan = {atk for atk, _, _ in experiments if atk in NEWS_ATTACKS}
    if news_attacks_in_plan and not args.dry_run:
        print("\n[Step 2] Pre-generating poisoned news (1회) ...")
        if "D1" in news_attacks_in_plan:
            pregen_news("SELL", "content")
        if "D2" in news_attacks_in_plan:
            pregen_news("SELL", "inject")
        print("  Poisoned news ready ✓\n")
    else:
        print("\n[Step 2] Skipped (dry-run or no D1/D2)\n")

    print(f"[Step 3] Launching {len(experiments)} experiments ({args.parallel} parallel) ...")
    start_time = datetime.now()
    results    = {}

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                run_single_experiment,
                attack, signal, structure,
                dry_run=args.dry_run,
                resume=args.resume,
                skip_news_regen=(attack in NEWS_ATTACKS),
            ): experiment_name(attack, signal, structure)
            for attack, signal, structure in experiments
        }
        for future in as_completed(futures):
            name, rc = future.result()
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            results[name] = rc
            done = len(results)
            print(f"  [{done}/{len(experiments)}] {name} {'✓' if rc == 0 else '✗'}  "
                  f"elapsed={elapsed:.1f}h")

    total_elapsed = (datetime.now() - start_time).total_seconds() / 3600
    failed = [(n, rc) for n, rc in results.items() if rc != 0]
    print("\n" + "=" * 70)
    print(f"Done.  Total: {total_elapsed:.1f}h")
    print(f"Results: trading_results/{START_DATE}_{END_DATE}/")
    print(f"  ✓ {len(results) - len(failed)} succeeded  ✗ {len(failed)} failed")
    for name, rc in failed:
        print(f"    ✗ {name} (rc={rc})")
    print("=" * 70)


if __name__ == "__main__":
    main()
