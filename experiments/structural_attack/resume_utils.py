"""
Shared helper so the run_attack_0N_*.py scripts can resume a previous partial
run (e.g. one that stopped after OpenAI insufficient_quota, or was killed
mid-run) without re-paying for ticker/dates that already produced a
successful result.

A (ticker, date) only counts as "done" if its row in the previous
<ticker><suffix> CSV has status == "success". Rows with status == "error"
(including the insufficient_quota row itself) are treated as not-yet-done
and get retried.
"""
import os
from typing import Dict, List, Tuple

import pandas as pd

DEFAULT_SUFFIX = "_attack_results.csv"


def load_prior_success(
    results_dir: str, ticker: str, suffix: str = DEFAULT_SUFFIX
) -> Tuple[set, List[dict]]:
    """Load already-successful rows for `ticker` from a previous run in results_dir.

    Returns (dates_already_done, prior_success_rows). Both are empty if there's
    no previous CSV for this ticker (first run, nothing to resume).
    """
    path = os.path.join(results_dir, f"{ticker}{suffix}")
    if not os.path.exists(path):
        return set(), []

    df = pd.read_csv(path)
    if "status" not in df.columns or "date" not in df.columns:
        return set(), []

    df = df[df["status"] == "success"].where(pd.notnull(df), None)
    rows = df.to_dict("records")
    dates = {str(d) for d in df["date"].tolist()}
    return dates, rows


def filter_pending_dates(
    ticker_dates: Dict[str, List[str]], results_dir: str, suffix: str = DEFAULT_SUFFIX
) -> Dict[str, List[str]]:
    """Drop dates each ticker already has a successful result for in results_dir.

    Prints a per-ticker skip count so a resumed run makes clear what it's
    skipping vs. what it's about to (re)pay OpenAI for.
    """
    pending = {}
    skipped_any = False
    for ticker, dates in ticker_dates.items():
        done_dates, _ = load_prior_success(results_dir, ticker, suffix)
        remaining = [d for d in dates if d not in done_dates]
        skipped = len(dates) - len(remaining)
        if skipped:
            skipped_any = True
            print(f"  {ticker}: resuming - {skipped} already done, {len(remaining)} remaining")
        pending[ticker] = remaining
    if skipped_any:
        print()
    return pending


def merge_with_prior(
    results_dir: str, ticker: str, new_results: List[dict], suffix: str = DEFAULT_SUFFIX
) -> List[dict]:
    """Combine previously-successful rows (skipped this run) with freshly
    produced ones, so the saved CSV reflects the union across every attempt
    instead of only the dates this particular run happened to touch.

    A date present in `new_results` always wins over a same-date prior row
    (that's only possible if the prior row was an error being retried).
    """
    _, prior_rows = load_prior_success(results_dir, ticker, suffix)
    new_dates = {r["date"] for r in new_results}
    combined = [r for r in prior_rows if r["date"] not in new_dates] + list(new_results)
    combined.sort(key=lambda r: r["date"])
    return combined
