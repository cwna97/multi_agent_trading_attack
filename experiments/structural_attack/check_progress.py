"""
Check progress of all running/finished structural_attack experiments.

Reads logs/.progress_<MODEL_NAME>.json (written by main.py / main_adversarial.py)
and cross-checks against currently running processes (by MODEL_NAME env var)
to show: done/total, %, whether it's still alive, and last update time.

Usage:
    .venv/bin/python3 check_progress.py                # all experiments
    .venv/bin/python3 check_progress.py gemini          # only names containing "gemini"
    .venv/bin/python3 check_progress.py --sort pct       # sort by completion % (default: name)
    .venv/bin/python3 check_progress.py --watch          # refresh every 5s
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

_COLOR = sys.stdout.isatty()

SCRIPT_DIR = Path(__file__).parent.resolve()
LOG_DIR = SCRIPT_DIR / "logs"


def get_live_model_names() -> set:
    """MODEL_NAME env values of every currently-running structural_attack process."""
    live = set()
    for pid_dir in glob.glob("/proc/[0-9]*"):
        pid = os.path.basename(pid_dir)
        try:
            with open(f"{pid_dir}/cmdline", "rb") as f:
                cmdline = f.read().decode(errors="ignore")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if "structural_attack" not in cmdline:
            continue
        try:
            with open(f"{pid_dir}/environ", "rb") as f:
                environ = f.read().decode(errors="ignore")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for entry in environ.split("\0"):
            if entry.startswith("MODEL_NAME="):
                live.add(entry[len("MODEL_NAME="):])
    return live


def collect_progress(name_filter: str = "") -> list[dict]:
    live_names = get_live_model_names()
    rows = []

    for pfile in sorted(LOG_DIR.glob(".progress_*.json")):
        name = pfile.stem[len(".progress_"):]
        if name_filter and name_filter.lower() not in name.lower():
            continue
        try:
            data = json.loads(pfile.read_text())
        except Exception:
            continue

        tickers = data.get("tickers", {})
        total_dates = data.get("total_dates", 0)
        done = sum(v.get("current", 0) for v in tickers.values())
        total = total_dates * len(tickers) if tickers else 0
        pct = (done / total * 100) if total else 0.0
        errors = sum(v.get("errors", 0) for v in tickers.values())

        alive = name in live_names
        if pct >= 100 or (tickers and all(v.get("status") == "completed" for v in tickers.values())):
            status = "완료"
        elif alive:
            status = "진행중"
        else:
            status = "중단됨"  # not 100% and no live process with this MODEL_NAME

        rows.append({
            "name": name,
            "done": done,
            "total": total,
            "pct": pct,
            "errors": errors,
            "status": status,
            "updated_at": data.get("updated_at", ""),
        })
    return rows


def render(rows: list[dict]):
    if not rows:
        print("표시할 진행 상황이 없습니다.")
        return

    name_w = max(len(r["name"]) for r in rows) + 2
    header = f"{'MODEL_NAME':<{name_w}} {'상태':<6} {'진행':>13} {'%':>6}  {'에러':>4}  최종갱신"
    print(header)
    print("-" * len(header))

    done_count = sum(1 for r in rows if r["status"] == "완료")
    running_count = sum(1 for r in rows if r["status"] == "진행중")
    stalled_count = sum(1 for r in rows if r["status"] == "중단됨")

    for r in rows:
        icon_map = {"완료": "✓", "진행중": "⏳", "중단됨": "✗"}
        color_map = {"완료": "\033[92m", "진행중": "\033[93m", "중단됨": "\033[91m"}
        icon = icon_map[r["status"]]
        color = color_map[r["status"]] if _COLOR else ""
        color_reset = "\033[0m" if _COLOR else ""
        err_str = str(r["errors"]) if r["errors"] else "-"
        print(
            f"{r['name']:<{name_w}} {color}{icon} {r['status']:<4}{color_reset} "
            f"{r['done']:>5}/{r['total']:<5} {r['pct']:>5.1f}%  {err_str:>4}  {r['updated_at']}"
        )

    print("-" * len(header))
    print(f"총 {len(rows)}개  |  완료 {done_count}  진행중 {running_count}  중단됨 {stalled_count}")


def main():
    parser = argparse.ArgumentParser(description="structural_attack 실험 진행률 확인")
    parser.add_argument("filter", nargs="?", default="", help="MODEL_NAME에 포함된 문자열로 필터링 (예: gemini)")
    parser.add_argument("--sort", choices=["name", "pct"], default="name")
    parser.add_argument("--watch", action="store_true", help="5초마다 갱신")
    args = parser.parse_args()

    def show():
        rows = collect_progress(args.filter)
        if args.sort == "pct":
            rows.sort(key=lambda r: r["pct"])
        if args.watch:
            print("\033[2J\033[H", end="")
        render(rows)

    if args.watch:
        try:
            while True:
                show()
                time.sleep(5)
        except KeyboardInterrupt:
            pass
    else:
        show()


if __name__ == "__main__":
    main()
