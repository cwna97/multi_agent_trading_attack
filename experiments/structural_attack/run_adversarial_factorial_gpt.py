"""
Adversarial Factorial Experiment — A1-overt & A2-subtle (GPT API)

기존 실험(A1=subtle/1명, A2=overt/2명)의 교란 변수 분리를 위한 추가 실험:
  - A1-overt : analyst 1개 타협 + overt bias  (수=1, 스타일=강함)
  - A2-subtle : analyst 2개 타협 + subtle bias (수=2, 스타일=은밀)

구조: linear | centralized | decentralized | hybrid  (moa 제외)
방향: BUY | SELL
총 실험: 2 attacks × 1 signal (BUY) × 4 structures = 8

결과 폴더: adv_gpt_A1overt_{signal}_{structure}/
           adv_gpt_A2subtle_{signal}_{structure}/

Usage:
    uv run python3 run_adversarial_factorial_gpt.py --all
    uv run python3 run_adversarial_factorial_gpt.py --all --dry-run
    uv run python3 run_adversarial_factorial_gpt.py --attack A1overt --signal BUY
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

STRUCTURES = ["linear", "centralized", "decentralized", "hybrid"]
SIGNALS    = ["BUY"]

# ---------------------------------------------------------------------------
# 추가 공격 설정 (quantity × style 교차)
# ---------------------------------------------------------------------------
ATTACK_CONFIGS = {
    # A1-overt: 1명 타협, overt (기존 A1은 subtle)
    "A1overt": {
        "attack_type":     "analyst_compromise",
        "bias_style":      "overt",
        "num_compromised": "1",
        "description":     "Single Analyst Compromise (overt)",
    },
    # A2-subtle: 2명 타협, subtle (기존 A2는 overt)
    "A2subtle": {
        "attack_type":     "analyst_compromise",
        "bias_style":      "subtle",
        "num_compromised": "2",
        "description":     "Multi-Analyst Compromise (subtle)",
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


def run_single(attack_id: str, signal: str, structure: str,
               dry_run: bool = False, resume: bool = False) -> int:
    name = experiment_name(attack_id, signal, structure)
    cfg  = ATTACK_CONFIGS[attack_id]

    # 이미 완료된 결과가 있으면 스킵
    out_dir = SCRIPT_DIR / "trading_results" / f"{START_DATE}_{END_DATE}" / name
    if out_dir.exists() and not resume:
        csv = out_dir / "all_tickers_results.csv"
        if csv.exists():
            print(f"  [{name}] 이미 존재 — 스킵 (--resume 으로 강제 재실행 가능)")
            return 0

    env = os.environ.copy()
    env.update({
        "MODEL_NAME":             name,
        "DEEP_THINK_LLM":         DEEP_THINK_LLM,
        "QUICK_THINK_LLM":        QUICK_THINK_LLM,
        "BACKEND_URL":            OPENAI_API_URL,
        "LLM_PROVIDER":           "openai",
        "GRAPH_STRUCTURE":        structure,
        "SELECTED_ANALYSTS":      SELECTED_ANALYSTS,
        "START_DATE":             START_DATE,
        "END_DATE":               END_DATE,
        "ATTACK_ENABLED":         "true",
        "ATTACK_TYPE":            cfg["attack_type"],
        "ATTACK_SIGNAL":          signal,
        "ATTACK_ID":              attack_id,
        "ATTACK_BIAS_STYLE":      cfg["bias_style"],
        "ATTACK_NUM_COMPROMISED": cfg["num_compromised"],
        "RESUME":                 "true" if resume else "false",
        "MAX_BUDGET_USD":         "0",
    })

    log_path = SCRIPT_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [{name}] {cfg['description']}  signal={signal}  structure={structure}")
    print(f"           bias={cfg['bias_style']}  n_compromised={cfg['num_compromised']}")
    print(f"           log → {log_path}")

    if dry_run:
        print(f"  [{name}] DRY RUN — skipping")
        return 0

    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT_DIR / "main_adversarial.py")],
            env=env, cwd=str(SCRIPT_DIR),
            stdout=lf, stderr=subprocess.STDOUT,
        )
        proc.wait()

    rc = proc.returncode
    print(f"  [{name}] {'✓ done' if rc == 0 else f'✗ failed (rc={rc})'}")
    return rc


def build_experiment_list(attacks=None, signals=None, structures=None):
    attacks    = attacks    or list(ATTACK_CONFIGS.keys())
    signals    = signals    or SIGNALS
    structures = structures or STRUCTURES
    return [
        (atk, sig, struct)
        for atk    in attacks
        for sig    in signals
        for struct in structures
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Factorial adversarial experiments: A1-overt & A2-subtle"
    )
    parser.add_argument("--all",       action="store_true", help="Run all 16 experiments")
    parser.add_argument("--attack",    choices=list(ATTACK_CONFIGS.keys()), help="Attack ID")
    parser.add_argument("--signal",    choices=["BUY", "SELL"], default="BUY")
    parser.add_argument("--structure", choices=STRUCTURES + ["all"], default="all")
    parser.add_argument("--dry-run",   action="store_true", help="Print plan without executing")
    parser.add_argument("--resume",    action="store_true", help="Re-run even if results exist")
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
    print("FACTORIAL ADVERSARIAL EXPERIMENT — GPT API")
    print(f"Period    : {START_DATE} ~ {END_DATE}")
    print(f"Model     : deep={DEEP_THINK_LLM}  quick={QUICK_THINK_LLM}")
    print(f"Attacks   : A1-overt (1명/overt)  A2-subtle (2명/subtle)")
    print(f"Structures: {STRUCTURES}")
    print(f"Total     : {len(experiments)} experiments")
    print("=" * 70)

    if not args.dry_run:
        print("\n[Step 1] Checking OpenAI API key...")
        if not check_openai_key():
            print("  ERROR: OPENAI_API_KEY not found")
            sys.exit(1)
        print("  OPENAI_API_KEY ✓\n")

    print(f"[Step 2] Experiment plan:")
    for i, (atk, sig, struct) in enumerate(experiments, 1):
        name = experiment_name(atk, sig, struct)
        print(f"  {i:2d}. {name}")

    if not args.dry_run:
        print()
        input("  Enter를 눌러 시작하거나 Ctrl+C로 취소: ")

    start_time = datetime.now()
    results    = {}

    for i, (atk, sig, struct) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {atk}  signal={sig}  structure={struct}")
        t0 = datetime.now()
        rc = run_single(atk, sig, struct, dry_run=args.dry_run, resume=args.resume)
        elapsed = (datetime.now() - t0).total_seconds() / 60
        results[experiment_name(atk, sig, struct)] = rc
        print(f"  elapsed: {elapsed:.1f}min")

    total_elapsed = (datetime.now() - start_time).total_seconds() / 3600
    print("\n" + "=" * 70)
    print(f"Done.  Total elapsed: {total_elapsed:.1f}h")
    ok  = sum(1 for rc in results.values() if rc == 0)
    err = sum(1 for rc in results.values() if rc != 0)
    print(f"Results: {ok} success, {err} failed")
    for name, rc in results.items():
        print(f"  {'✓' if rc == 0 else '✗'} {name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
