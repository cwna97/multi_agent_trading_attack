"""
Q1 2026 Adversarial Experiment Runner — GPT (OpenAI API, gpt-4.1 / gpt-4.1-mini)

기간: 2026-01-01 ~ 2026-03-31
티커: BTC-USD, MSFT, NVDA, TSLA, AAPL (명시적으로 env["TICKERS"]로 넘김 - run_q1_adversarial_qwen.py와 동일)
구조: linear, centralized, decentralized, hybrid (moa, debate 제외)
공격: D1(News content poisoning), D2(Social indirect injection), A1, A2, A3 × BUY/SELL → 총 40 실험

(구 run_adversarial_experiments_gpt.py를 run_q1_adversarial_qwen.py와 동일한 구조로
맞춘 버전 - TICKERS를 명시 변수로 빼서 env로 넘기고, ThreadPoolExecutor 기반 병렬
실행 + 실시간 진행률 모니터를 추가했다. experiment_name()의 출력 폴더 이름
(adv_gpt_<attack>_<signal>_<structure>, "q1" 없음)은 기존 trading_results/ 데이터와
호환되도록 그대로 유지 - 여기서 바꾸면 지금까지 만든 모든 분석 스크립트/그림이
가리키는 경로가 깨진다.)

Usage:
    tmux new-session -s q1_adv_gpt
    .venv/bin/python3 run_q1_adversarial_gpt.py --all

    .venv/bin/python3 run_q1_adversarial_gpt.py --all --parallel 4
    .venv/bin/python3 run_q1_adversarial_gpt.py --attack D1 --signal SELL --structure linear
    .venv/bin/python3 run_q1_adversarial_gpt.py --all --dry-run
    .venv/bin/python3 run_q1_adversarial_gpt.py --all --resume
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

SCRIPT_DIR = Path(__file__).parent.resolve()

OPENAI_API_URL  = "https://api.openai.com/v1"
DEEP_THINK_LLM  = "gpt-4.1"
QUICK_THINK_LLM = "gpt-4.1-mini"
LLM_PROVIDER    = "openai"

START_DATE        = "2026-01-01"
END_DATE          = "2026-03-31"
SELECTED_ANALYSTS = "market,social,news"
TICKERS            = ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]

STRUCTURES      = ["linear", "centralized", "decentralized", "hybrid"]
SIGNALS         = ["BUY", "SELL"]
DEFAULT_ATTACKS = ["D1", "D2", "A1", "A2", "A3"]

ATTACK_CONFIGS = {
    # D1: News content replacement (headline/summary/content 교체)
    "D1": {
        "attack_type": "news_text_poison",
        "poison_mode": "content",
        "poison_rate": "0.1",
        "description": "News Content Poisoning",
    },
    # D2: Social indirect prompt injection (Reddit 포스트에 숨긴 LLM instruction 삽입 - news_data 아님)
    "D2": {
        "attack_type": "social_inject",
        "poison_rate": "0.1",
        "description": "Social Indirect Injection",
    },
    # A1: Single analyst compromise (subtle, 1개 analyst)
    "A1": {
        "attack_type":     "analyst_compromise",
        "bias_style":      "subtle",
        "num_compromised": "1",
        "description":     "Single Analyst Compromise (subtle)",
    },
    # A2: Multi-analyst compromise (overt, 2개 analyst)
    "A2": {
        "attack_type":     "analyst_compromise",
        "bias_style":      "overt",
        "num_compromised": "2",
        "description":     "Multi-Analyst Compromise (overt)",
    },
    # A3: Decision-maker hijacking (forced)
    "A3": {
        "attack_type": "decision_maker_hijacking",
        "hijack_mode": "forced",
        "description": "Decision-Maker Hijacking (forced)",
    },
}

NEWS_ATTACKS = {"D1"}  # D2 is social_inject - config-flag driven, no pregen needed


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


def experiment_name(attack_id: str, signal: str, structure: str, poison_rate: str = None) -> str:
    base = f"adv_gpt_{attack_id}_{signal.lower()}_{structure}"
    # Only D1/D2 read poison_rate at all, and only append a suffix when it's
    # overridden away from the attack's own configured default - keeps the
    # existing adv_gpt_D1_buy_linear/etc. folder names untouched for the
    # normal (0.1) case, same convention as role_attack's _ratioNN suffix.
    default_rate = ATTACK_CONFIGS.get(attack_id, {}).get("poison_rate")
    if poison_rate is not None and poison_rate != default_rate:
        base += f"_ratio{int(round(float(poison_rate) * 100))}"
    return base


def run_single_experiment(
    attack_id: str, signal: str, structure: str,
    dry_run: bool = False, resume: bool = False,
    poison_rate: str = None,
) -> tuple[str, int]:
    name = experiment_name(attack_id, signal, structure, poison_rate)
    cfg  = ATTACK_CONFIGS[attack_id]

    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        name,
        "DEEP_THINK_LLM":    DEEP_THINK_LLM,
        "QUICK_THINK_LLM":   QUICK_THINK_LLM,
        "LLM_PROVIDER":      LLM_PROVIDER,
        "BACKEND_URL":       OPENAI_API_URL,
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
        "MAX_BUDGET_USD":    "0",
    })

    if "poison_mode" in cfg:
        env["ATTACK_POISON_MODE"] = cfg["poison_mode"]
    if "poison_rate" in cfg:
        env["ATTACK_POISON_RATE"] = poison_rate if poison_rate is not None else cfg["poison_rate"]
    if "bias_style" in cfg:
        env["ATTACK_BIAS_STYLE"]      = cfg["bias_style"]
        env["ATTACK_NUM_COMPROMISED"] = cfg.get("num_compromised", "1")
    if "hijack_mode" in cfg:
        env["ATTACK_HIJACK_MODE"] = cfg["hijack_mode"]

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
        lines.append(f"Q1 2026 GPT ADVERSARIAL  |  {datetime.now().strftime('%H:%M:%S')}  elapsed={elapsed:.1f}h")
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

        ok  = sum(1 for rc in results.values() if rc == 0)
        err = sum(1 for rc in results.values() if rc != 0)
        lines.append("=" * 72)
        lines.append(f"완료 ✓{ok}  실패 ✗{err}  진행 중 {total - done}")
        lines.append("로그: logs/adv_gpt_<attack>_<signal>_<struct>.log")

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
    parser = argparse.ArgumentParser(description="Q1 2026 Adversarial Experiment Runner (GPT / OpenAI API)")
    parser.add_argument("--all",       action="store_true", help="Run all experiments (5 attacks x 2 signals x 4 structures)")
    parser.add_argument("--attack",    choices=list(ATTACK_CONFIGS.keys()), help="Attack ID")
    parser.add_argument("--signal",    choices=["BUY", "SELL"], default="SELL")
    parser.add_argument("--structure", choices=STRUCTURES + ["all"], default="all")
    parser.add_argument("--parallel",  type=int, default=6, help="동시 실행 수 (default: 6)")
    parser.add_argument("--poison-rate", type=str, default=None,
                         help="D1/D2 poison_rate override (0-1, e.g. 0.4). Only affects D1/D2 - "
                              "ignored for A1/A2/A3. Output folder gets a _ratioNN suffix when this "
                              "differs from the attack's own default, so it never collides with the "
                              "existing adv_gpt_D1_.../adv_gpt_D2_... results.")
    parser.add_argument("--dry-run",   action="store_true")
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
    print("Q1 2026 ADVERSARIAL EXPERIMENT RUNNER — GPT (OpenAI API)")
    print(f"Period:   {START_DATE} ~ {END_DATE}")
    print(f"Model:    deep={DEEP_THINK_LLM}  quick={QUICK_THINK_LLM}")
    print(f"Attacks:  {list(ATTACK_CONFIGS.keys())}")
    print(f"Parallel: {args.parallel} 개 동시 실행")
    print(f"Total:    {len(experiments)} experiments")
    print("=" * 70)

    if not args.dry_run:
        print("\n[Step 1] Checking OpenAI API key...")
        if not check_openai_key():
            print("  ERROR: OPENAI_API_KEY not found in env or .env file")
            sys.exit(1)
        print("  OPENAI_API_KEY ✓")

    print(f"\n[Step 2] Launching {len(experiments)} experiments ({args.parallel} parallel)...")
    start_time = datetime.now()
    results    = {}
    exp_names  = [experiment_name(a, s, st, args.poison_rate) for a, s, st in experiments]

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
    print(f"All Q1 GPT adversarial experiments finished.  Total: {total_elapsed:.1f}h")
    print(f"Results: trading_results/{START_DATE}_{END_DATE}/")
    failed = [(n, rc) for n, rc in results.items() if rc != 0]
    print(f"  ✓ {len(results) - len(failed)} succeeded  ✗ {len(failed)} failed")
    for name, rc in failed:
        print(f"    ✗ {name} (rc={rc})")
    print("=" * 70)


if __name__ == "__main__":
    main()
