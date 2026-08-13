"""
Financial Impact: how much CR drops under DATA-LEVEL attacks (D1, D2) vs. the
clean GPT-4.1 baseline, for Q1 2026 (2026-01-01_2026-03-31), per structure,
averaged across all 5 tickers (BTC-USD, MSFT, NVDA, TSLA, AAPL).

Reuses the same long-only BUY/SELL/HOLD simulation and local price cache as
generate_financial_impact_table_q1.py. D1/D2 are the "data poisoning" attacks
(target=BUY in this BUY-targeted quarter), read from
trading_results/2026-01-01_2026-03-31/adv_gpt_D{1,2}_buy_{structure}/.
"""

import pandas as pd
from pathlib import Path

from generate_financial_impact_table_q1 import (
    BASE_DIR, TRADING_DIR, TICKERS, STRUCTURES, STRUCT_LABEL, STRUCT_FOLDER,
    load_prices, calculate_metrics,
)

ATTACKS = ["D1", "D2"]
ATTACK_FOLDER = {(s, a): f"adv_gpt_{a}_buy_{s}" for s in STRUCTURES for a in ATTACKS}


def run_clean(structure, ticker):
    folder = TRADING_DIR / STRUCT_FOLDER[structure]
    ddf = pd.read_csv(folder / f"{ticker}_results.csv")
    ddf = ddf[ddf["status"] == "success"][["date", "decision"]]
    return calculate_metrics(ddf, load_prices(ticker))["CR%"]


def run_attack(structure, attack, ticker):
    folder = TRADING_DIR / ATTACK_FOLDER[(structure, attack)]
    ddf = pd.read_csv(folder / f"{ticker}_results.csv")
    ddf = ddf[ddf["status"] == "success"][["date", "decision"]]
    return calculate_metrics(ddf, load_prices(ticker))["CR%"]


rows = []
for structure in STRUCTURES:
    for ticker in TICKERS:
        clean_cr = run_clean(structure, ticker)
        row = {"Structure": STRUCT_LABEL[structure], "Ticker": ticker, "Clean CR (%)": clean_cr}
        for attack in ATTACKS:
            attack_cr = run_attack(structure, attack, ticker)
            row[f"{attack} CR (%)"] = attack_cr
            row[f"{attack} ΔCR (%)"] = round(attack_cr - clean_cr, 2)
        rows.append(row)

df = pd.DataFrame(rows)
out_path = BASE_DIR / "financial_impact_attack_delta_q1_2026_gpt41.csv"
df.to_csv(out_path, index=False)

print(df.to_string(index=False))
print()

# Per-structure average across all 5 tickers
summary = df.groupby("Structure").agg({
    "Clean CR (%)": "mean",
    "D1 CR (%)": "mean", "D1 ΔCR (%)": "mean",
    "D2 CR (%)": "mean", "D2 ΔCR (%)": "mean",
}).round(2).reindex(list(STRUCT_LABEL.values()))
summary["Data-level (D1+D2) ΔCR (%)"] = df.groupby("Structure").apply(
    lambda g: pd.concat([g["D1 ΔCR (%)"], g["D2 ΔCR (%)"]]).mean()
).round(2).reindex(summary.index)

print("=== Per-structure average across 5 tickers ===")
print(summary.to_string())
print(f"\nSaved per-ticker detail to {out_path}")
