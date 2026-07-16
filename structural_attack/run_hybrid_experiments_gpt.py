"""
Hybrid Structure Experiment Runner — GPT API (gpt-4.1 / gpt-4.1-mini)

Clean 베이스라인 1개 + Adversarial 10개(D1/D2/A1/A2/A3 × BUY/SELL) 를
hybrid 구조에 대해서만 실행합니다.

Usage:
    # 전체 실행 (clean + adversarial 11개)
    .venv/bin/python3 run_hybrid_experiments_gpt.py --all

    # clean 베이스라인만
    .venv/bin/python3 run_hybrid_experiments_gpt.py --clean

    # adversarial 전체 (10개)
    .venv/bin/python3 run_hybrid_experiments_gpt.py --adv

    # 특정 공격 하나
    .venv/bin/python3 run_hybrid_experiments_gpt.py --attack D1 --signal SELL

    # 계획만 확인 (실제 실행 없음)
    .venv/bin/python3 run_hybrid_experiments_gpt.py --all --dry-run

    # 이전에 성공한 날짜 건너뛰기
    .venv/bin/python3 run_hybrid_experiments_gpt.py --all --resume
"""

import subprocess
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from tradingagents.utils.cost_tracker import CostTracker

SCRIPT_DIR      = Path(__file__).parent.resolve()
STRUCTURE       = "hybrid"
OPENAI_API_URL  = "https://api.openai.com/v1"
DEEP_THINK_LLM  = "gpt-4.1"
QUICK_THINK_LLM = "gpt-4.1-mini"
START_DATE      = "2026-01-01"
END_DATE        = "2026-03-31"
SELECTED_ANALYSTS = "market,social,news"

# ── 공격 구성 정의 ──────────────────────────────────────────────────────────────
SIGNALS = ["BUY", "SELL"]

ATTACK_CONFIGS = {
    "D1": {
        "attack_type":  "news_text_poison",
        "poison_mode":  "content",
        "poison_rate":  "1.0",
        "description":  "News Content Poisoning",
    },
    "D2": {
        "attack_type":  "news_inject",
        "poison_mode":  "inject",
        "poison_rate":  "1.0",
        "description":  "Indirect Prompt Injection",
    },
    "A1": {
        "attack_type":       "analyst_compromise",
        "bias_style":        "subtle",
        "num_compromised":   "1",
        "description":       "Single Analyst Compromise (subtle)",
    },
    "A2": {
        "attack_type":       "analyst_compromise",
        "bias_style":        "overt",
        "num_compromised":   "2",
        "description":       "Multi-Analyst Compromise (overt)",
    },
    "A3": {
        "attack_type":  "decision_maker_hijacking",
        "hijack_mode":  "forced",
        "description":  "Decision-Maker Hijacking (forced)",
    },
}


# ── OpenAI API 키 확인 ──────────────────────────────────────────────────────────
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


# ── 실험 이름 ────────────────────────────────────────────────────────────────────
def clean_name() -> str:
    return f"gpt-4.1_{STRUCTURE}"


def adv_name(attack_id: str, signal: str) -> str:
    return f"adv_gpt_{attack_id}_{signal.lower()}_{STRUCTURE}"


# ── Clean 베이스라인 실행 ─────────────────────────────────────────────────────────
def run_clean(dry_run: bool = False, resume: bool = False, max_budget_usd: float = 0.0) -> int:
    name = clean_name()
    env  = os.environ.copy()
    env.update({
        "MODEL_NAME":        name,
        "DEEP_THINK_LLM":    DEEP_THINK_LLM,
        "QUICK_THINK_LLM":   QUICK_THINK_LLM,
        "BACKEND_URL":       OPENAI_API_URL,
        "LLM_PROVIDER":      "openai",
        "GRAPH_STRUCTURE":   STRUCTURE,
        "SELECTED_ANALYSTS": SELECTED_ANALYSTS,
        "START_DATE":        START_DATE,
        "END_DATE":          END_DATE,
        "ATTACK_ENABLED":    "false",
        "RESUME":            "true" if resume else "false",
        "MAX_BUDGET_USD":    str(max_budget_usd),
    })

    log_path = SCRIPT_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [{name}] Clean baseline  structure={STRUCTURE}")
    print(f"  [{name}] log: {log_path}")

    if dry_run:
        print(f"  [{name}] DRY RUN — skipping")
        return 0

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT_DIR / "main_adversarial.py")],
            env=env, cwd=str(SCRIPT_DIR),
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        proc.wait()

    rc = proc.returncode
    print(f"  [{name}] {'✓ done' if rc == 0 else f'✗ failed (rc={rc})'}")
    return rc


# ── Adversarial 실험 실행 ─────────────────────────────────────────────────────────
def run_adversarial(attack_id: str, signal: str, dry_run: bool = False, resume: bool = False, max_budget_usd: float = 0.0) -> int:
    name = adv_name(attack_id, signal)
    cfg  = ATTACK_CONFIGS[attack_id]

    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        name,
        "DEEP_THINK_LLM":    DEEP_THINK_LLM,
        "QUICK_THINK_LLM":   QUICK_THINK_LLM,
        "BACKEND_URL":       OPENAI_API_URL,
        "LLM_PROVIDER":      "openai",
        "GRAPH_STRUCTURE":   STRUCTURE,
        "SELECTED_ANALYSTS": SELECTED_ANALYSTS,
        "START_DATE":        START_DATE,
        "END_DATE":          END_DATE,
        "ATTACK_ENABLED":    "true",
        "ATTACK_TYPE":       cfg["attack_type"],
        "ATTACK_SIGNAL":     signal,
        "ATTACK_ID":         attack_id,
        "RESUME":            "true" if resume else "false",
        "MAX_BUDGET_USD":    str(max_budget_usd),
    })

    if "poison_mode" in cfg:
        env["ATTACK_POISON_MODE"] = cfg["poison_mode"]
        env["ATTACK_POISON_RATE"] = cfg.get("poison_rate", "1.0")
    if "bias_style" in cfg:
        env["ATTACK_BIAS_STYLE"]       = cfg["bias_style"]
        env["ATTACK_NUM_COMPROMISED"]  = cfg.get("num_compromised", "1")
    if "hijack_mode" in cfg:
        env["ATTACK_HIJACK_MODE"] = cfg["hijack_mode"]

    log_path = SCRIPT_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [{name}] {cfg['description']}  target={signal}")
    print(f"  [{name}] log: {log_path}")

    if dry_run:
        print(f"  [{name}] DRY RUN — skipping")
        return 0

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT_DIR / "main_adversarial.py")],
            env=env, cwd=str(SCRIPT_DIR),
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        proc.wait()

    rc = proc.returncode
    print(f"  [{name}] {'✓ done' if rc == 0 else f'✗ failed (rc={rc})'}")
    return rc


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Structure Experiment Runner (GPT API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all",   action="store_true", help="Clean + 전체 adversarial (11개)")
    group.add_argument("--clean", action="store_true", help="Clean 베이스라인만 실행")
    group.add_argument("--adv",   action="store_true", help="Adversarial 전체 10개 실행")
    group.add_argument("--attack", choices=list(ATTACK_CONFIGS.keys()), help="특정 공격 하나")
    parser.add_argument("--signal",  choices=["BUY", "SELL"], default=None,
                        help="공격 방향 (--attack과 함께 사용; 미지정 시 BUY+SELL 모두 실행)")
    parser.add_argument("--dry-run", action="store_true", help="계획만 출력, 실제 실행 없음")
    parser.add_argument("--resume",  action="store_true", help="이전 성공 날짜 건너뛰기")
    parser.add_argument("--budget",  type=float, default=0.0,
                        metavar="USD",
                        help="총 누적 비용 한도 (USD). 초과 시 실험 중단. 기본: 무제한")
    args = parser.parse_args()

    if not any([args.all, args.clean, args.adv, args.attack]):
        parser.print_help()
        sys.exit(0)

    # 실행 목록 구성
    tasks = []   # list of ("clean", None, None) | ("adv", attack_id, signal)

    if args.all or args.clean:
        tasks.append(("clean", None, None))

    if args.all or args.adv:
        for attack_id in ATTACK_CONFIGS:
            for signal in SIGNALS:
                tasks.append(("adv", attack_id, signal))

    if args.attack:
        signals = [args.signal] if args.signal else SIGNALS
        for signal in signals:
            tasks.append(("adv", args.attack, signal))

    budget = args.budget

    # 헤더 출력
    print("=" * 70)
    print(f"HYBRID STRUCTURE EXPERIMENT RUNNER — GPT API")
    print(f"Structure : {STRUCTURE}")
    print(f"Period    : {START_DATE} ~ {END_DATE}")
    print(f"Model     : deep={DEEP_THINK_LLM}  quick={QUICK_THINK_LLM}")
    print(f"Total     : {len(tasks)} experiments")
    if budget > 0:
        current_spent = CostTracker.get_total_cost()
        print(f"Budget    : ${budget:.2f}  (현재 누적: ${current_spent:.4f}  잔여: ${budget - current_spent:.4f})")
    if args.resume:
        print("Mode      : RESUME (이전 성공 날짜 건너뜀)")
    print("=" * 70)

    # 실험 목록 미리 표시
    print("\n실험 목록:")
    for kind, attack_id, signal in tasks:
        if kind == "clean":
            print(f"  ○ {clean_name():<40} [Clean baseline]")
        else:
            name = adv_name(attack_id, signal)
            desc = ATTACK_CONFIGS[attack_id]["description"]
            print(f"  ○ {name:<40} [{desc} → {signal}]")
    print()

    if not args.dry_run:
        print("[Step 1] Checking OpenAI API key...")
        if not check_openai_key():
            print("  ERROR: OPENAI_API_KEY not found in env or .env file")
            sys.exit(1)
        print("  OPENAI_API_KEY ✓\n")

    # 순차 실행
    start_time = datetime.now()
    results = {}

    for i, (kind, attack_id, signal) in enumerate(tasks, 1):
        if kind == "clean":
            label = clean_name()
        else:
            label = adv_name(attack_id, signal)

        # ── 실험 시작 전 예산 확인 ──────────────────────────────────────────
        if budget > 0 and not args.dry_run:
            current_spent = CostTracker.get_total_cost()
            remaining = budget - current_spent
            if current_spent >= budget:
                print(f"\n[Budget] ⛔ 총 예산 ${budget:.2f} 초과 (현재 ${current_spent:.4f})")
                print(f"[Budget] 남은 실험 {len(tasks) - i + 1}개를 건너뜁니다.")
                for _, aid, sig in tasks[i - 1:]:
                    lbl = clean_name() if _ == "clean" else adv_name(aid, sig)
                    results[lbl] = -99   # budget skip 표시
                break
            print(f"  [Budget] 누적 ${current_spent:.4f} / ${budget:.2f}  (잔여 ${remaining:.4f})")

        print(f"\n[{i}/{len(tasks)}] {label}")
        t0 = datetime.now()

        if kind == "clean":
            rc = run_clean(dry_run=args.dry_run, resume=args.resume, max_budget_usd=budget)
        else:
            rc = run_adversarial(attack_id, signal, dry_run=args.dry_run, resume=args.resume, max_budget_usd=budget)

        elapsed = (datetime.now() - t0).total_seconds() / 3600
        print(f"  Elapsed: {elapsed:.2f}h  rc={rc}")
        results[label] = rc

    # 최종 요약
    total_elapsed = (datetime.now() - start_time).total_seconds() / 3600
    print("\n" + "=" * 70)
    print(f"HYBRID EXPERIMENTS COMPLETE  (Total elapsed: {total_elapsed:.1f}h)")
    print(f"Results: trading_results/{START_DATE}_{END_DATE}/")
    success  = sum(1 for rc in results.values() if rc == 0)
    skipped  = sum(1 for rc in results.values() if rc == -99)
    failed   = len(results) - success - skipped
    for name, rc in results.items():
        if rc == 0:       status = "✓"
        elif rc == -99:   status = "⛔ budget skip"
        else:             status = f"✗ (rc={rc})"
        print(f"  {status} {name}/")
    summary = f"  {success}/{len(results)} succeeded"
    if skipped: summary += f", {skipped} budget-skipped"
    if failed:  summary += f", {failed} failed"
    print(summary)
    if not args.dry_run:
        print()
        CostTracker.print_summary()
    print("=" * 70)


if __name__ == "__main__":
    main()
