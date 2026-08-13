"""
Run Role-Attack Suite: clean baseline + all 5 attack scenarios, for one or
more periods, on a fixed ticker list.

Order matters: every attack script (01-05) reads the baseline decision from
eval_results/{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json
to pick its target action (the opposite of whatever the clean run decided).
So for a given period+ticker, `main.py` (clean) MUST run and produce that log
before any attack script for the same period+ticker can do anything useful.

This script therefore runs, for each period, in strict sequence:
    1. main.py                              (clean/baseline)
    2. scripts/run_attack_01_data_poisoning.py
    3. scripts/run_attack_02_indirect_injection.py
    4. scripts/run_attack_03_persuasive_adversary.py  --role bullish --strength strong
    5. scripts/run_attack_04_objective_hijacking.py
    6. scripts/run_attack_05_jailbreaking.py          --method authority_override

If any stage exits non-zero (in particular: OpenAI quota exhaustion, which
main.py and all 5 attack scripts now detect and stop on immediately - see
_is_quota_exhausted() in each of them), the whole suite stops immediately.
Nothing after the failed stage is started, so you don't burn budget running
attacks against an incomplete/partial baseline, or start a new period after
the account can't make any more calls.

Usage:
    python run_role_attack_suite.py
    python run_role_attack_suite.py --tickers BTC-USD MSFT NVDA TSLA AAPL \\
        --periods 2025-07-01:2025-09-30 2026-01-01:2026-03-31
    python run_role_attack_suite.py --workers 2 --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# When stdout isn't a TTY (piped to a file, `nohup ... &`, etc.) CPython fully
# block-buffers instead of line-buffering, so progress would only show up in
# multi-KB bursts. Force line buffering here so this script's own prints
# appear immediately regardless of how it's invoked.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass  # Python <3.7 fallback: not expected here, but don't crash on it

REPO_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable  # the interpreter currently running this script (expected: .venv/bin/python)

sys.path.insert(0, str(REPO_DIR))
from llm_provider_cli import add_llm_provider_args  # noqa: E402

# Force the same unbuffered behavior in every child stage (main.py, the 5
# attack scripts), so their own live progress bars show up immediately too
# instead of appearing in delayed bursts when this whole suite is backgrounded
# or redirected to a log file.
CHILD_ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}

DEFAULT_TICKERS = ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]
DEFAULT_PERIODS = ["2025-07-01:2025-09-30", "2026-01-01:2026-03-31"]


def parse_period(period_str: str):
    start, end = period_str.split(":")
    return start.strip(), end.strip()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run clean baseline + all 5 role_attack scenarios for one or more periods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_role_attack_suite.py
  python run_role_attack_suite.py --periods 2025-07-01:2025-09-30
  python run_role_attack_suite.py --tickers AAPL MSFT --workers 2
  python run_role_attack_suite.py --dry-run
        """,
    )
    add_llm_provider_args(parser)
    parser.add_argument(
        "--tickers", type=str, nargs="+", default=DEFAULT_TICKERS,
        help=f"Tickers to run (default: {' '.join(DEFAULT_TICKERS)})",
    )
    parser.add_argument(
        "--periods", type=str, nargs="+", default=DEFAULT_PERIODS,
        help="Periods as START:END (YYYY-MM-DD:YYYY-MM-DD), space-separated "
             f"(default: {' '.join(DEFAULT_PERIODS)})",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Parallel workers passed to every stage (default: 3)",
    )
    parser.add_argument(
        "--persuasive-role", choices=["bullish", "bearish"], default="bullish",
        help="attack_03 role variant to run (default: bullish)",
    )
    parser.add_argument(
        "--persuasive-strength", choices=["mild", "moderate", "strong"], default="strong",
        help="attack_03 persuasion strength (default: strong)",
    )
    parser.add_argument(
        "--jailbreak-method", choices=["authority_override", "hypothetical"], default="authority_override",
        help="attack_05 jailbreak method (default: authority_override)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the exact commands that would run, without executing them.",
    )
    return parser.parse_args()


def llm_provider_flags(args) -> list:
    """Reconstruct the --llm-provider/--openrouter-* flags to forward to every stage,
    so main.py and all 5 attack scripts run against the same backbone as this suite."""
    flags = ["--llm-provider", args.llm_provider]
    if args.llm_provider == "openrouter":
        flags += ["--openrouter-model", args.openrouter_model]
    if args.llm_provider in ("openrouter", "gemini", "qwen"):
        # Only forward these if the user explicitly set them - otherwise let
        # each child script fall back to its own per-model preset default
        # (see PRESET_MAX_PRICE in llm_provider_cli.py).
        if args.openrouter_max_price_prompt is not None:
            flags += ["--openrouter-max-price-prompt", str(args.openrouter_max_price_prompt)]
        if args.openrouter_max_price_completion is not None:
            flags += ["--openrouter-max-price-completion", str(args.openrouter_max_price_completion)]
        if args.openrouter_no_throughput_routing:
            flags.append("--openrouter-no-throughput-routing")
    return flags


def build_stages(tickers, start, end, workers, role, strength, method, provider_flags):
    """Return the ordered list of (label, cmd) for one period. Clean MUST be first."""
    common = [
        "--tickers", *tickers, "--start-date", start, "--end-date", end, "--workers", str(workers),
        *provider_flags,
    ]
    return [
        ("clean (baseline)", [PYTHON, "main.py", *common]),
        ("attack_01_data_poisoning", [PYTHON, "scripts/run_attack_01_data_poisoning.py", *common]),
        ("attack_02_indirect_injection", [PYTHON, "scripts/run_attack_02_indirect_injection.py", *common]),
        ("attack_03_persuasive_adversary", [
            PYTHON, "scripts/run_attack_03_persuasive_adversary.py",
            "--role", role, "--strength", strength, *common,
        ]),
        ("attack_04_objective_hijacking", [PYTHON, "scripts/run_attack_04_objective_hijacking.py", *common]),
        ("attack_05_jailbreaking", [
            PYTHON, "scripts/run_attack_05_jailbreaking.py",
            "--method", method, *common,
        ]),
    ]


def main():
    args = parse_arguments()
    periods = [parse_period(p) for p in args.periods]
    provider_flags = llm_provider_flags(args)

    manifest = {
        "started_at": datetime.now().isoformat(),
        "tickers": args.tickers,
        "periods": [f"{s}:{e}" for s, e in periods],
        "workers": args.workers,
        "persuasive_role": args.persuasive_role,
        "persuasive_strength": args.persuasive_strength,
        "jailbreak_method": args.jailbreak_method,
        "llm_provider": args.llm_provider,
        "openrouter_model": args.openrouter_model,
        "openrouter_max_price_prompt": args.openrouter_max_price_prompt,
        "openrouter_max_price_completion": args.openrouter_max_price_completion,
        "openrouter_no_throughput_routing": args.openrouter_no_throughput_routing,
        "stages": [],
    }

    print("=" * 70)
    print("ROLE_ATTACK SUITE: clean + 5 attack scenarios")
    print("=" * 70)
    print(f"Tickers: {', '.join(args.tickers)}")
    print(f"Periods: {', '.join(f'{s}..{e}' for s, e in periods)}")
    print(f"Workers per stage: {args.workers}")
    print(f"LLM provider: {args.llm_provider}"
          + (f" ({args.openrouter_model})" if args.llm_provider == "openrouter" else ""))
    print(f"attack_03 variant: {args.persuasive_role}/{args.persuasive_strength}")
    print(f"attack_05 variant: {args.jailbreak_method}")
    print(f"Stages per period: clean, attack_01, attack_02, attack_03, attack_04, attack_05")
    print("=" * 70)

    for period_idx, (start, end) in enumerate(periods, 1):
        print(f"\n{'#'*70}")
        print(f"# PERIOD {period_idx}/{len(periods)}: {start} .. {end}")
        print(f"{'#'*70}\n")

        stages = build_stages(
            args.tickers, start, end, args.workers,
            args.persuasive_role, args.persuasive_strength, args.jailbreak_method,
            provider_flags,
        )

        for stage_idx, (label, cmd) in enumerate(stages, 1):
            print(f"\n{'-'*70}")
            print(f"[{period_idx}.{stage_idx}] {label}  ({start}..{end})")
            print(f"cmd: {' '.join(cmd)}")
            print(f"{'-'*70}\n")

            if args.dry_run:
                manifest["stages"].append({
                    "period": f"{start}:{end}", "label": label, "cmd": cmd, "status": "dry-run",
                })
                continue

            t0 = time.time()
            result = subprocess.run(cmd, cwd=str(REPO_DIR), env=CHILD_ENV)
            elapsed = time.time() - t0

            stage_record = {
                "period": f"{start}:{end}",
                "label": label,
                "cmd": cmd,
                "returncode": result.returncode,
                "elapsed_sec": round(elapsed, 1),
                "status": "ok" if result.returncode == 0 else "FAILED",
            }
            manifest["stages"].append(stage_record)
            _save_manifest(manifest)

            if result.returncode != 0:
                print(f"\n{'!'*70}")
                print(f"STOPPING SUITE: [{period_idx}.{stage_idx}] {label} exited with code {result.returncode}")
                print(f"(most likely cause: OpenAI API quota exhausted - see that stage's own output above)")
                print(f"Nothing after this stage was started. Partial results are saved under")
                print(f"eval_results/, trading_results/, and attack_results/ as usual.")
                print(f"{'!'*70}\n")
                manifest["finished_at"] = datetime.now().isoformat()
                manifest["stopped_early"] = True
                _save_manifest(manifest)
                sys.exit(1)

    manifest["finished_at"] = datetime.now().isoformat()
    manifest["stopped_early"] = False
    _save_manifest(manifest)

    print("\n" + "=" * 70)
    print("ROLE_ATTACK SUITE COMPLETE - all periods, all stages finished successfully")
    print(f"Manifest: {_manifest_path()}")
    print("=" * 70)


def _manifest_path() -> Path:
    return REPO_DIR / "run_role_attack_suite_manifest.json"


def _save_manifest(manifest: dict) -> None:
    with open(_manifest_path(), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
