"""
Adversarial Experiment Runner
구조: linear | centralized | decentralized | moa  (debate 제외)
공격: D1 | D2 | A1 | A2 | A3  × 방향 BUY | SELL

Usage:
    # 특정 공격 하나 실행
    .venv/bin/python3 run_adversarial_experiments.py --attack D1 --signal SELL --structure linear

    # 전체 44 실험 순차 실행 (tmux 권장)
    .venv/bin/python3 run_adversarial_experiments.py --all

    # 특정 구조만
    .venv/bin/python3 run_adversarial_experiments.py --all --structure linear

환경 변수 (run_experiments.py와 동일):
    DEEP_THINK_LLM, QUICK_THINK_LLM, BACKEND_URL, LLM_PROVIDER
    START_DATE, END_DATE, SELECTED_ANALYSTS
"""

import subprocess
import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# 공격 구성 정의
# ---------------------------------------------------------------------------

STRUCTURES = ["linear", "centralized", "decentralized", "moa"]
SIGNALS    = ["BUY", "SELL"]

ATTACK_CONFIGS = {
    # D1: News content replacement (headline/summary/content 교체)
    "D1": {
        "attack_type":  "news_text_poison",
        "poison_mode":  "content",
        "poison_rate":  "1.0",
        "description":  "News Content Poisoning",
    },
    # D2: Indirect prompt injection (기사 본문 끝에 LLM instruction 삽입)
    "D2": {
        "attack_type":  "news_inject",
        "poison_mode":  "inject",
        "poison_rate":  "1.0",
        "description":  "Indirect Prompt Injection",
    },
    # A1: Single analyst compromise (subtle, 1개 analyst)
    "A1": {
        "attack_type":  "analyst_compromise",
        "bias_style":   "subtle",
        "num_compromised": "1",
        "description":  "Single Analyst Compromise (subtle)",
    },
    # A2: Multi-analyst compromise (overt, 2개 analyst)
    "A2": {
        "attack_type":  "analyst_compromise",
        "bias_style":   "overt",
        "num_compromised": "2",
        "description":  "Multi-Analyst Compromise (overt)",
    },
    # A3: Decision-maker hijacking (forced)
    "A3": {
        "attack_type":  "decision_maker_hijacking",
        "hijack_mode":  "forced",
        "description":  "Decision-Maker Hijacking (forced)",
    },
}

# ---------------------------------------------------------------------------
# 기본 LLM / 실험 기간 설정 (환경변수로 오버라이드 가능)
# ---------------------------------------------------------------------------

START_DATE = os.environ.get("START_DATE", "2026-01-01")
END_DATE   = os.environ.get("END_DATE",   "2026-03-31")
SELECTED_ANALYSTS = os.environ.get("SELECTED_ANALYSTS", "market,social,news")

LLM_PROVIDER   = os.environ.get("LLM_PROVIDER",    "ollama")
DEEP_THINK_LLM = os.environ.get("DEEP_THINK_LLM",  "qwen2.5:72b")
QUICK_THINK_LLM= os.environ.get("QUICK_THINK_LLM", "qwen2.5:14b")
BACKEND_URL    = os.environ.get("BACKEND_URL",      "http://localhost:11434/v1")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def experiment_name(attack_id: str, signal: str, structure: str) -> str:
    return f"adv_{attack_id}_{signal.lower()}_{structure}"


def run_single_experiment(attack_id: str, signal: str, structure: str, dry_run: bool = False):
    """단일 실험 실행. main_adversarial.py를 서브프로세스로 호출."""
    name = experiment_name(attack_id, signal, structure)
    cfg  = ATTACK_CONFIGS[attack_id]

    env = os.environ.copy()
    env.update({
        "MODEL_NAME":          name,
        "DEEP_THINK_LLM":      DEEP_THINK_LLM,
        "QUICK_THINK_LLM":     QUICK_THINK_LLM,
        "BACKEND_URL":         BACKEND_URL,
        "LLM_PROVIDER":        LLM_PROVIDER,
        "GRAPH_STRUCTURE":     structure,
        "SELECTED_ANALYSTS":   SELECTED_ANALYSTS,
        "START_DATE":          START_DATE,
        "END_DATE":            END_DATE,
        # Attack config (main_adversarial.py가 읽음)
        "ATTACK_ENABLED":      "true",
        "ATTACK_TYPE":         cfg["attack_type"],
        "ATTACK_SIGNAL":       signal,
        "ATTACK_ID":           attack_id,
    })

    # attack-specific env vars
    if "poison_mode" in cfg:
        env["ATTACK_POISON_MODE"]  = cfg["poison_mode"]
        env["ATTACK_POISON_RATE"]  = cfg.get("poison_rate", "1.0")
    if "bias_style" in cfg:
        env["ATTACK_BIAS_STYLE"]   = cfg["bias_style"]
        env["ATTACK_NUM_COMPROMISED"] = cfg.get("num_compromised", "1")
    if "hijack_mode" in cfg:
        env["ATTACK_HIJACK_MODE"]  = cfg["hijack_mode"]

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
    exps = []
    for attack in attacks:
        for signal in signals:
            for structure in structures:
                exps.append((attack, signal, structure))
    return exps


def main():
    parser = argparse.ArgumentParser(description="Adversarial Experiment Runner")
    parser.add_argument("--all",       action="store_true", help="Run all 44 experiments")
    parser.add_argument("--attack",    choices=list(ATTACK_CONFIGS.keys()), help="Attack ID")
    parser.add_argument("--signal",    choices=["BUY", "SELL"], default="SELL")
    parser.add_argument("--structure", choices=STRUCTURES + ["all"], default="all")
    parser.add_argument("--dry-run",   action="store_true", help="Print plan without executing")
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
    print("ADVERSARIAL EXPERIMENT RUNNER")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print(f"Model:  deep={DEEP_THINK_LLM}  quick={QUICK_THINK_LLM}")
    print(f"Total experiments: {len(experiments)}")
    print("=" * 70)

    for i, (attack, signal, structure) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Attack={attack}  Signal={signal}  Structure={structure}")
        start = datetime.now()
        rc = run_single_experiment(attack, signal, structure, dry_run=args.dry_run)
        elapsed = (datetime.now() - start).total_seconds() / 3600
        print(f"  Elapsed: {elapsed:.1f}h  rc={rc}")

    print("\n" + "=" * 70)
    print("All adversarial experiments finished.")
    print(f"Results: trading_results/{START_DATE}_{END_DATE}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
