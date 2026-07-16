"""
Adversarial Experiment Runner — GPT API (gpt-4.1 / gpt-4.1-mini)

구조: linear | centralized | decentralized | hybrid  (moa, debate 제외)
공격: D1 | D2 | A1 | A2 | A3  × 방향 BUY | SELL  → 총 40 실험

Usage:
    # 전체 40 실험 순차 실행 (tmux 권장)
    tmux new-session -s adv_gpt
    .venv/bin/python3 run_adversarial_experiments_gpt.py --all

    # 특정 공격 하나 실행
    .venv/bin/python3 run_adversarial_experiments_gpt.py --attack D1 --signal SELL --structure linear

    # 특정 구조만
    .venv/bin/python3 run_adversarial_experiments_gpt.py --all --structure linear

    # dry-run (실행 없이 계획만 출력)
    .venv/bin/python3 run_adversarial_experiments_gpt.py --all --dry-run
"""

import subprocess
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

OPENAI_API_URL  = "https://api.openai.com/v1"
DEEP_THINK_LLM  = "gpt-4.1"
QUICK_THINK_LLM = "gpt-4.1-mini"

START_DATE        = "2026-01-01"
END_DATE          = "2026-03-31"
SELECTED_ANALYSTS = "market,social,news"

# ---------------------------------------------------------------------------
# 공격 구성 정의 (run_adversarial_experiments.py와 동일)
# ---------------------------------------------------------------------------

STRUCTURES = ["linear", "centralized", "decentralized", "hybrid"]
SIGNALS    = ["BUY", "SELL"]

ATTACK_CONFIGS = {
    # D1: News content replacement (headline/summary/content 교체)
    "D1": {
        "attack_type": "news_text_poison",
        "poison_mode": "content",
        "poison_rate": "1.0",
        "description": "News Content Poisoning",
    },
    # D2: Indirect prompt injection (기사 본문 끝에 LLM instruction 삽입)
    "D2": {
        "attack_type": "news_inject",
        "poison_mode": "inject",
        "poison_rate": "1.0",
        "description": "Indirect Prompt Injection",
    },
    # A1: Single analyst compromise (subtle, 1개 analyst)
    "A1": {
        "attack_type":      "analyst_compromise",
        "bias_style":       "subtle",
        "num_compromised":  "1",
        "description":      "Single Analyst Compromise (subtle)",
    },
    # A2: Multi-analyst compromise (overt, 2개 analyst)
    "A2": {
        "attack_type":      "analyst_compromise",
        "bias_style":       "overt",
        "num_compromised":  "2",
        "description":      "Multi-Analyst Compromise (overt)",
    },
    # A3: Decision-maker hijacking (forced)
    "A3": {
        "attack_type":  "decision_maker_hijacking",
        "hijack_mode":  "forced",
        "description":  "Decision-Maker Hijacking (forced)",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def experiment_name(attack_id: str, signal: str, structure: str) -> str:
    return f"adv_gpt_{attack_id}_{signal.lower()}_{structure}"


def run_single_experiment(attack_id: str, signal: str, structure: str, dry_run: bool = False, resume: bool = False) -> int:
    name = experiment_name(attack_id, signal, structure)
    cfg  = ATTACK_CONFIGS[attack_id]

    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        name,
        "DEEP_THINK_LLM":    DEEP_THINK_LLM,
        "QUICK_THINK_LLM":   QUICK_THINK_LLM,
        "BACKEND_URL":       OPENAI_API_URL,
        "LLM_PROVIDER":      "openai",
        "GRAPH_STRUCTURE":   structure,
        "SELECTED_ANALYSTS": SELECTED_ANALYSTS,
        "START_DATE":        START_DATE,
        "END_DATE":          END_DATE,
        # Attack config
        "ATTACK_ENABLED":    "true",
        "ATTACK_TYPE":       cfg["attack_type"],
        "ATTACK_SIGNAL":     signal,
        "ATTACK_ID":         attack_id,
        "RESUME":            "true" if resume else "false",
    })

    if "poison_mode" in cfg:
        env["ATTACK_POISON_MODE"] = cfg["poison_mode"]
        env["ATTACK_POISON_RATE"] = cfg.get("poison_rate", "1.0")
    if "bias_style" in cfg:
        env["ATTACK_BIAS_STYLE"]        = cfg["bias_style"]
        env["ATTACK_NUM_COMPROMISED"]   = cfg.get("num_compromised", "1")
    if "hijack_mode" in cfg:
        env["ATTACK_HIJACK_MODE"] = cfg["hijack_mode"]

    log_path = SCRIPT_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [{name}] {cfg['description']} target={signal} structure={structure}")
    print(f"  [{name}] log: {log_path}")

    if dry_run:
        print(f"  [{name}] DRY RUN — skipping actual execution")
        return 0

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
    status = "✓ done" if rc == 0 else f"✗ failed (rc={rc})"
    print(f"  [{name}] {status}")
    return rc


def build_experiment_list(
    attacks: list = None,
    signals: list = None,
    structures: list = None,
) -> list:
    attacks    = attacks    or list(ATTACK_CONFIGS.keys())
    signals    = signals    or SIGNALS
    structures = structures or STRUCTURES
    return [
        (attack, signal, structure)
        for attack    in attacks
        for signal    in signals
        for structure in structures
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Adversarial Experiment Runner (GPT API)")
    parser.add_argument("--all",       action="store_true", help="Run all 40 experiments")
    parser.add_argument("--attack",    choices=list(ATTACK_CONFIGS.keys()), help="Attack ID")
    parser.add_argument("--signal",    choices=["BUY", "SELL"], default="SELL")
    parser.add_argument("--structure", choices=STRUCTURES + ["all"], default="all", help="Structure to run (moa excluded)")
    parser.add_argument("--dry-run",   action="store_true", help="Print plan without executing")
    parser.add_argument("--resume",    action="store_true", help="Skip already-succeeded dates (preserves existing results)")
    args = parser.parse_args()

    structures = STRUCTURES if args.structure == "all" else [args.structure]

    if args.all:
        experiments = build_experiment_list(structures=structures)
    elif args.attack:
        experiments = [(args.attack, args.signal, s) for s in structures]
    else:
        parser.print_help()
        sys.exit(0)

    print("=" * 70)
    print("ADVERSARIAL EXPERIMENT RUNNER — GPT API")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print(f"Model:  deep={DEEP_THINK_LLM}  quick={QUICK_THINK_LLM}")
    print(f"Total experiments: {len(experiments)}")
    print("=" * 70)

    if not args.dry_run:
        print("\n[Step 1] Checking OpenAI API key...")
        if not check_openai_key():
            print("  ERROR: OPENAI_API_KEY not found in env or .env file")
            sys.exit(1)
        print("  OPENAI_API_KEY ✓\n")

    start_time = datetime.now()
    results    = {}

    for i, (attack, signal, structure) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Attack={attack}  Signal={signal}  Structure={structure}")
        t0 = datetime.now()
        rc = run_single_experiment(attack, signal, structure, dry_run=args.dry_run, resume=args.resume)
        elapsed = (datetime.now() - t0).total_seconds() / 3600
        print(f"  Elapsed: {elapsed:.1f}h  rc={rc}")
        results[experiment_name(attack, signal, structure)] = rc

    total_elapsed = (datetime.now() - start_time).total_seconds() / 3600
    print("\n" + "=" * 70)
    print(f"All adversarial GPT experiments finished.  Total elapsed: {total_elapsed:.1f}h")
    print(f"Results: trading_results/{START_DATE}_{END_DATE}/")
    for name, rc in results.items():
        status = "✓" if rc == 0 else f"✗ (rc={rc})"
        print(f"  {status} {name}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
