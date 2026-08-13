"""
Null Structure Experiment Runner — Compliance-Control Baseline (multi-provider)

목적: 구조(Linear/Centralized/Decentralized/Hybrid)가 만드는 aggregation 효과를
완전히 제거한 "solo, no-communication" 베이스라인을 측정. market/social/news
analyst 각각을 독립적으로(서로 통신 없이) 그래프에 단 하나만 넣고 곧바로
결정을 내리게 한 뒤(Null 구조, graph_structure="null"), 여기에 D1/D2/A1
공격을 적용해 "모델 자체의 raw compliance"를 구조 효과와 분리해서 측정한다.

기간: 2026-01-01 ~ 2026-03-31 (기존 Q1 실험과 동일 — 직접 비교 가능)
티커: BTC-USD, MSFT, NVDA, TSLA, AAPL
Analyst: market, social, news (fundamentals 미사용 — 기존 파이프라인과 동일)
공격: D1(news content poison), D2(news inject), A1(1-analyst compromise, subtle)
      + clean baseline (공격 없음)

D1/D2는 news 데이터 자체를 오염시키므로 news analyst에는 직접 영향을 주고,
market/social solo에는 영향이 없어야 정상 — 이 결과는 attack specificity에
대한 negative control로 사용한다. A1은 selected_analysts가 단일 analyst이므로
자동으로 "그 analyst 자신"을 compromise한다 (attack_analyst_compromise.py:
targets = explicit_list or selected_analysts[:num_compromised]).

Usage:
    .venv/bin/python3 run_null_structure.py --provider gemini --all
    .venv/bin/python3 run_null_structure.py --provider gpt    --all
    .venv/bin/python3 run_null_structure.py --provider qwen   --all --parallel 3

    .venv/bin/python3 run_null_structure.py --provider gemini --analyst news --attack D1
    .venv/bin/python3 run_null_structure.py --provider gemini --all --dry-run
    .venv/bin/python3 run_null_structure.py --provider gemini --all --resume
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

START_DATE       = "2026-01-01"
END_DATE         = "2026-03-31"
BASE_NEWS_SOURCE = "alpha_vantage_filled"
TICKERS          = ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]

ANALYSTS = ["market", "social", "news"]
DEFAULT_SIGNAL = "BUY"

PROVIDER_CONFIGS = {
    "gemini": {
        "prefix":      "gemini",
        "deep":        "google/gemini-2.5-flash",
        "quick":       "google/gemini-2.5-flash-lite",
        "llm_provider": "openrouter",
        "backend_url": "https://openrouter.ai/api/v1",
        "key_env":     "OPENROUTER_API_KEY",
    },
    "gpt": {
        "prefix":      "gpt",
        "deep":        "gpt-4.1",
        "quick":       "gpt-4.1-mini",
        "llm_provider": "openai",
        "backend_url": "https://api.openai.com/v1",
        "key_env":     "OPENAI_API_KEY",
    },
    "qwen": {
        "prefix":      "qwen",
        "deep":        "qwen/qwen3-235b-a22b-2507",
        "quick":       "qwen/qwen3-30b-a3b-instruct-2507",
        "llm_provider": "openrouter",
        "backend_url": "https://openrouter.ai/api/v1",
        "key_env":     "OPENROUTER_API_KEY",
    },
}

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
        "description":     "Single Analyst Compromise (subtle) — targets itself",
    },
}
NEWS_ATTACKS = {"D1"}  # D2 is now social_inject - config-flag driven, no pregen needed
DEFAULT_ATTACKS = ["clean", "D1", "D2", "A1"]   # "clean" = no attack


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


def pregen_news(mode: str, signal: str, poison_rate: float = 1.0):
    local_data_dir = str(SCRIPT_DIR / "tradingagents/dataflows/local_data")
    print(f"  [Pregen] mode={mode} signal={signal} → generating poisoned news...")
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


def experiment_name(prefix: str, attack_id: str, analyst: str, poison_rate: str = None,
                    signal: str = DEFAULT_SIGNAL) -> str:
    base = f"null_{prefix}_{attack_id}_{analyst}"
    # No suffix for the original default (BUY) so existing null_<provider>_D1_news/
    # etc. folders stay exactly where they are; SELL gets its own distinct name.
    if signal != DEFAULT_SIGNAL:
        base += f"_{signal.lower()}"
    default_rate = ATTACK_CONFIGS.get(attack_id, {}).get("poison_rate")
    if poison_rate is not None and poison_rate != default_rate:
        base += f"_ratio{int(round(float(poison_rate) * 100))}"
    return base


def run_single_experiment(
    provider_cfg: dict, attack_id: str, analyst: str,
    dry_run: bool = False, resume: bool = False,
    skip_news_regen: bool = False, poison_rate: str = None,
    signal: str = DEFAULT_SIGNAL,
) -> tuple[str, int]:
    name = experiment_name(provider_cfg["prefix"], attack_id, analyst, poison_rate, signal)
    is_clean = attack_id == "clean"

    env = os.environ.copy()
    env.update({
        "MODEL_NAME":        name,
        "DEEP_THINK_LLM":    provider_cfg["deep"],
        "QUICK_THINK_LLM":   provider_cfg["quick"],
        "LLM_PROVIDER":      provider_cfg["llm_provider"],
        "GRAPH_STRUCTURE":   "null",
        "SELECTED_ANALYSTS": analyst,          # 단일 analyst만 — Null 구조 필수 조건
        "START_DATE":        START_DATE,
        "END_DATE":          END_DATE,
        "TICKERS":           ",".join(TICKERS),
        "ATTACK_ENABLED":    "false" if is_clean else "true",
        "RESUME":            "true" if resume else "false",
        "SKIP_NEWS_REGEN":   "true" if skip_news_regen else "false",
        "MAX_BUDGET_USD":    "0",
    })
    if provider_cfg["backend_url"]:
        env["BACKEND_URL"] = provider_cfg["backend_url"]

    if not is_clean:
        cfg = ATTACK_CONFIGS[attack_id]
        env.update({
            "ATTACK_TYPE":   cfg["attack_type"],
            "ATTACK_SIGNAL": signal,
            "ATTACK_ID":     attack_id,
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

    desc = "clean baseline" if is_clean else ATTACK_CONFIGS[attack_id]["description"]
    print(f"  [{name}] {desc}  analyst={analyst}")

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
        lines.append(f"NULL STRUCTURE RUNNER  |  {datetime.now().strftime('%H:%M:%S')}  elapsed={elapsed:.1f}h")
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
        lines.append("로그: logs/null_<provider>_<attack>_<analyst>.log")
        print("\n".join(lines), flush=True)
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Null Structure (solo, no-communication) compliance-control runner")
    parser.add_argument("--provider", choices=list(PROVIDER_CONFIGS.keys()), required=True)
    parser.add_argument("--all",      action="store_true", help="Run all 12 experiments (4 attacks incl. clean × 3 analysts)")
    parser.add_argument("--attack",   choices=list(ATTACK_CONFIGS.keys()) + ["clean"], help="Attack ID (or 'clean')")
    parser.add_argument("--analyst",  choices=ANALYSTS + ["all"], default="all")
    parser.add_argument("--signal",   choices=["BUY", "SELL"], default=DEFAULT_SIGNAL,
                         help="Target signal for D1/D2/A1 (default: BUY). SELL runs get a "
                              "_sell suffix on the output folder so they never collide with "
                              "the existing BUY results.")
    parser.add_argument("--parallel", type=int, default=3, help="동시 실행 수 (default: 3)")
    parser.add_argument("--poison-rate", type=str, default=None,
                         help="D1/D2 poison_rate override (0-1, e.g. 0.4). Only affects D1/D2 - "
                              "ignored for A1/clean. Output folder gets a _ratioNN suffix when this "
                              "differs from the attack's own default, so it never collides with the "
                              "existing null_<provider>_D1_.../null_<provider>_D2_... results.")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--resume",   action="store_true")
    args = parser.parse_args()

    provider_cfg = PROVIDER_CONFIGS[args.provider]
    analysts = ANALYSTS if args.analyst == "all" else [args.analyst]

    if args.all:
        experiments = [(atk, a) for atk in DEFAULT_ATTACKS for a in analysts]
    elif args.attack:
        experiments = [(args.attack, a) for a in analysts]
    else:
        parser.print_help()
        sys.exit(0)

    print("=" * 70)
    print(f"NULL STRUCTURE RUNNER — {args.provider}")
    print(f"Period:   {START_DATE} ~ {END_DATE}")
    print(f"Model:    deep={provider_cfg['deep']}  quick={provider_cfg['quick']}")
    print(f"Analysts: {analysts}  (no inter-analyst communication)")
    print(f"Attacks:  {DEFAULT_ATTACKS}")
    print(f"Total:    {len(experiments)} experiments")
    print("=" * 70)

    if not args.dry_run:
        print(f"\n[Step 1] Checking {provider_cfg['key_env']}...")
        if not check_api_key(provider_cfg):
            print(f"  ERROR: {provider_cfg['key_env']} not found in env or .env file")
            sys.exit(1)
        print(f"  {provider_cfg['key_env']} ✓")

    news_attacks_in_plan = {atk for atk, _ in experiments if atk in NEWS_ATTACKS}
    if news_attacks_in_plan and not args.dry_run:
        print("\n[Step 2] Pre-generating poisoned news files (1회)...")
        if "D1" in news_attacks_in_plan:
            # Same fix as run_q1_adversarial_qwen.py: pregen_news's own
            # poison_rate default (1.0) was never overridden by
            # ATTACK_CONFIGS["D1"]["poison_rate"] or a CLI override, so
            # SKIP_NEWS_REGEN runs were silently reading a 100%-poisoned
            # file regardless of what rate was configured.
            d1_rate = float(args.poison_rate if args.poison_rate is not None
                            else ATTACK_CONFIGS["D1"]["poison_rate"])
            pregen_news("content", args.signal, poison_rate=d1_rate)
        print("  Poisoned news ready ✓\n")
    else:
        print("\n[Step 2] Skipped (dry-run or no D1)\n")

    print(f"[Step 3] Launching {len(experiments)} experiments ({args.parallel} parallel)...")
    start_time = datetime.now()
    results: dict[str, int] = {}
    exp_names = [experiment_name(provider_cfg["prefix"], atk, a, args.poison_rate, args.signal)
                 for atk, a in experiments]

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
                provider_cfg, attack_id, analyst,
                dry_run=args.dry_run,
                resume=args.resume,
                skip_news_regen=(attack_id in NEWS_ATTACKS),
                poison_rate=args.poison_rate,
                signal=args.signal,
            ): experiment_name(provider_cfg["prefix"], attack_id, analyst, args.poison_rate, args.signal)
            for attack_id, analyst in experiments
        }
        for future in as_completed(futures):
            name, rc = future.result()
            results[name] = rc

    if not args.dry_run:
        stop_monitor.set()

    total_elapsed = (datetime.now() - start_time).total_seconds() / 3600
    print("\n" + "=" * 70)
    print(f"Null structure runner finished ({args.provider}).  Total: {total_elapsed:.1f}h")
    print(f"Results: trading_results/{START_DATE}_{END_DATE}/null_{provider_cfg['prefix']}_<attack>_<analyst>/")
    failed = [(n, rc) for n, rc in results.items() if rc != 0]
    print(f"  ✓ {len(results) - len(failed)} succeeded  ✗ {len(failed)} failed")
    for name, rc in failed:
        print(f"    ✗ {name} (rc={rc})")
    print("=" * 70)


if __name__ == "__main__":
    main()
