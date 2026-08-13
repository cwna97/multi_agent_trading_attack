"""
role_attack 진행률 확인 스크립트

logs/*.log에 남은 마지막 진행 상황 스냅샷(ANSI progress bar)을 파싱하고,
실제 저장된 결과 CSV 행수와 비교해서 보여준다.

Usage:
    .venv/bin/python3 check_progress.py                # logs/*.log 전부
    .venv/bin/python3 check_progress.py logs/resume_attack01_q3.log
"""

import glob
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# 예: "✓ BTC-USD [████...] 64/ 64 (100.0%) 2025-09-30 (2 errors)"
LINE_RE = re.compile(
    r"(?P<icon>[✓⏳○])\s+(?P<ticker>[\w.-]+)\s*\[[█░]+\]\s*"
    r"(?P<current>\d+)/\s*(?P<total>\d+)\s*\(\s*(?P<pct>[\d.]+)%\)\s*"
    r"(?P<last_date>\S*)\s*(?:\((?P<errors>\d+) errors?\))?"
)
# 시작 시 찍히는 "  MSFT: resuming - 59 already done, 5 remaining"
RESUME_RE = re.compile(
    r"^\s*(?P<ticker>[\w.-]+):\s*resuming\s*-\s*(?P<done>\d+)\s*already done,\s*(?P<remaining>\d+)\s*remaining",
    re.MULTILINE,
)
FOUND_DAYS_RE = re.compile(r"Found\s+(\d+)\s+trading days")


def parse_resume_baseline(text: str) -> dict[str, tuple[int, int]]:
    """시작 시점에 이미 완료돼 있던 개수와 이번 세션에 처리할 remaining 개수."""
    return {
        m.group("ticker"): (int(m.group("done")), int(m.group("remaining")))
        for m in RESUME_RE.finditer(text)
    }


def parse_log(path: Path) -> dict[str, dict]:
    """로그 파일에서 각 ticker의 마지막 progress 스냅샷을 추출."""
    raw = path.read_text(errors="ignore")
    text = ANSI_RE.sub("", raw)

    found_days_m = FOUND_DAYS_RE.search(text)
    total_days = int(found_days_m.group(1)) if found_days_m else None
    baseline = parse_resume_baseline(text)  # ticker -> (prior_done, remaining_at_start)

    latest: dict[str, dict] = {}
    for m in LINE_RE.finditer(text):
        ticker = m.group("ticker")
        current, bar_total = int(m.group("current")), int(m.group("total"))
        prior_done, remaining_at_start = baseline.get(ticker, (0, None))

        if remaining_at_start == 0:
            # 세션 시작 전 이미 완료 → skip된 케이스, bar 자체가 전체 기간 기준
            overall_done, overall_total = current, bar_total
        elif total_days is not None:
            # bar의 current/total은 "이번 세션에 처리할 remaining" 기준 → prior_done을 더해야 전체 진행률
            overall_done, overall_total = prior_done + current, total_days
        else:
            overall_done, overall_total = current, bar_total

        latest[ticker] = {
            "icon": m.group("icon"),
            "current": overall_done,
            "total": overall_total,
            "pct": round(overall_done / overall_total * 100, 1) if overall_total else 0.0,
            "last_date": m.group("last_date"),
            "errors": int(m.group("errors")) if m.group("errors") else 0,
        }
    return latest


def find_results_dir(log_path: Path) -> Path | None:
    """로그 내용에서 attack 이름과 기간을 찾아 결과 디렉터리를 추정."""
    text = log_path.read_text(errors="ignore")
    m_attack = re.search(r"ATTACK\s+(\d+):\s*([A-Z ]+)", text)
    m_period = re.search(r"Period:\s*(\d{4}-\d{2}-\d{2})\s*to\s*(\d{4}-\d{2}-\d{2})", text)
    if not m_period:
        return None
    start, end = m_period.group(1), m_period.group(2)
    period_dir = f"{start}_{end}"

    attack_dir_map = {
        "01": "attack_01_data_poisoning",
        "02": "attack_02_indirect_injection",
        "03": "attack_03_persuasive_adversary",
        "04": "attack_04_objective_hijacking",
        "05": "attack_05_jailbreaking_authority_override",
    }
    if m_attack:
        num = m_attack.group(1)
        attack_dir = attack_dir_map.get(num)
        if attack_dir:
            candidate = BASE / "attack_results" / period_dir / attack_dir
            if candidate.exists():
                return candidate

    candidate = BASE / "trading_results" / period_dir
    return candidate if candidate.exists() else None


def csv_row_count(results_dir: Path, ticker: str) -> int | None:
    for suffix in ["_attack_results.csv", "_results.csv"]:
        p = results_dir / f"{ticker}{suffix}"
        if p.exists():
            try:
                return len(pd.read_csv(p))
            except Exception:
                return None
    return None


def report(log_path: Path):
    print(f"\n{'='*70}")
    print(f"{log_path.name}")
    print("=" * 70)

    snapshot = parse_log(log_path)
    if not snapshot:
        print("  (진행 스냅샷을 아직 못 찾음 - 막 시작했거나 포맷이 다를 수 있음)")
        return

    results_dir = find_results_dir(log_path)

    total_current = total_total = 0
    for ticker, info in snapshot.items():
        current, total = info["current"], info["total"]
        total_current += current
        total_total += total

        saved = csv_row_count(results_dir, ticker) if results_dir else None
        saved_str = f"  (저장된 CSV: {saved}행)" if saved is not None else ""

        bar_w = 24
        filled = int(bar_w * current / total) if total else 0
        bar = "#" * filled + "-" * (bar_w - filled)
        err_str = f"  {info['errors']}건 에러" if info["errors"] else ""
        print(
            f"  {info['icon']} {ticker:10s} [{bar}] {current:3d}/{total:3d} "
            f"({info['pct']:5.1f}%){err_str}{saved_str}"
        )

    overall = total_current / total_total * 100 if total_total else 0
    print("-" * 70)
    print(f"  전체: {total_current}/{total_total} ({overall:.1f}%)")

    # 프로세스가 살아있는지 참고 정보
    import subprocess
    running = subprocess.run(
        ["pgrep", "-f", log_path.stem.replace("resume_", "")],
        capture_output=True, text=True
    ).stdout.strip()


def main():
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted((BASE / "logs").glob("*.log"))

    if not paths:
        print("logs/*.log 없음 - 로그 파일 경로를 직접 넘겨주세요.")
        return

    for p in paths:
        report(p)
    print()


if __name__ == "__main__":
    main()
