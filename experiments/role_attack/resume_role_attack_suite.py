"""
Resume the role_attack suite after it stopped early (e.g. OpenAI insufficient_quota).

You don't need to figure out which stage/ticker/date to restart from by hand:

- main.py and all 5 run_attack_0N_*.py scripts now skip any (ticker, date)
  they already have a *successful* saved result for (see resume_utils.py),
  and retry only dates that are missing or previously errored.
- So this script just re-reads run_role_attack_suite_manifest.json from the
  failed run and re-invokes run_role_attack_suite.py with the *exact same*
  --tickers/--periods/--workers/attack-variant flags used before.

Stages that already finished 100% (e.g. "clean" and attack_01/attack_02 for
the first period, in a run that failed partway through attack_03) come back
almost instantly instead of re-paying for OpenAI calls, because every ticker
in that stage has nothing left to do. The stage that failed partway through
picks up on exactly the (ticker, date) pairs it hadn't finished yet. Stages/
periods that never started at all just run as normal.

The previous manifest (with the FAILED stage recorded) is copied aside first
so that failure history isn't lost when run_role_attack_suite.py writes a
fresh manifest for this resumed invocation.

Usage:
    python resume_role_attack_suite.py
    python resume_role_attack_suite.py --dry-run
    python resume_role_attack_suite.py --workers 2   # override worker count for this resume
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = REPO_DIR / "run_role_attack_suite_manifest.json"


def main():
    parser = argparse.ArgumentParser(
        description="Resume the role_attack suite from its last saved manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_PATH,
        help=f"Manifest to resume from (default: {MANIFEST_PATH})",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override the worker count from the manifest for this resume (default: reuse manifest's).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the reconstructed command without running it or backing up the manifest.",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        sys.exit(f"No manifest found at {args.manifest} - nothing to resume from.")

    with open(args.manifest) as f:
        manifest = json.load(f)

    if not manifest.get("stopped_early"):
        print(f"{args.manifest} says the suite finished successfully - nothing to resume.")
        return

    failed_stages = [s for s in manifest.get("stages", []) if s.get("status") == "FAILED"]
    ok_stages = [s for s in manifest.get("stages", []) if s.get("status") == "ok"]

    workers = args.workers if args.workers is not None else manifest["workers"]
    cmd = [
        sys.executable, str(REPO_DIR / "run_role_attack_suite.py"),
        "--tickers", *manifest["tickers"],
        "--periods", *manifest["periods"],
        "--workers", str(workers),
        "--persuasive-role", manifest["persuasive_role"],
        "--persuasive-strength", manifest["persuasive_strength"],
        "--jailbreak-method", manifest["jailbreak_method"],
    ]
    # Older manifests (written before --llm-provider support) don't have these
    # keys - default to openai (gpt-4.1) so a resume of an old run keeps using
    # the same backbone it started with.
    llm_provider = manifest.get("llm_provider", "openai")
    cmd += ["--llm-provider", llm_provider]
    if llm_provider == "openrouter":
        cmd += ["--openrouter-model", manifest.get("openrouter_model", "qwen/qwen3-235b-a22b-2507")]
    if llm_provider in ("openrouter", "gemini", "qwen"):
        # Only forward if the original run explicitly set them - otherwise
        # let each child script fall back to its own per-model preset default.
        if manifest.get("openrouter_max_price_prompt") is not None:
            cmd += ["--openrouter-max-price-prompt", str(manifest["openrouter_max_price_prompt"])]
        if manifest.get("openrouter_max_price_completion") is not None:
            cmd += ["--openrouter-max-price-completion", str(manifest["openrouter_max_price_completion"])]
        if manifest.get("openrouter_no_throughput_routing"):
            cmd.append("--openrouter-no-throughput-routing")

    print("=" * 70)
    print("RESUMING ROLE_ATTACK SUITE")
    print("=" * 70)
    print(f"Previously completed stages: {len(ok_stages)}")
    if failed_stages:
        last = failed_stages[-1]
        print(f"Stopped at: {last['label']} ({last['period']})  -  will retry only its unfinished ticker/dates")
    print("Everything before that will be re-issued too, but each stage's own script")
    print("skips ticker/dates it already has a successful result for, so completed")
    print("stages return almost instantly instead of re-paying for OpenAI calls.")
    print(f"\ncmd: {' '.join(cmd)}")
    print("=" * 70)

    if args.dry_run:
        return

    backup_path = args.manifest.with_name(
        f"{args.manifest.stem}.failed_{datetime.now().strftime('%Y%m%dT%H%M%S')}{args.manifest.suffix}"
    )
    shutil.copy2(args.manifest, backup_path)
    print(f"Backed up previous (failed) manifest to {backup_path.name}\n")

    result = subprocess.run(cmd, cwd=str(REPO_DIR))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
