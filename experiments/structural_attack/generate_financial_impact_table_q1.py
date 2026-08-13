"""
Financial Impact table for Q1 2026 (2026-01-01_2026-03-31), CLEAN GPT-4.1 baseline
runs only, across the 4 structures (decentralized, hybrid, centralized, linear)
and 5 tickers (BTC-USD, MSFT, NVDA, TSLA, AAPL).

Reuses the exact long-only BUY/SELL/HOLD simulation from calculate_metrics.py
(same entry/exit logic, $100,000 initial capital), but sources prices from the
local cache (tradingagents/dataflows/local_data/market_data/price_data/) instead
of hitting yfinance over the network - the local files already cover the needed
date range.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
TRADING_DIR = BASE_DIR / "trading_results" / "2026-01-01_2026-03-31"
PRICE_DIR = BASE_DIR / "tradingagents" / "dataflows" / "local_data" / "market_data" / "price_data"

TICKERS = ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]
STRUCTURES = ["decentralized", "hybrid", "centralized", "linear"]
STRUCT_LABEL = {"decentralized": "Decentralized", "hybrid": "Hybrid",
                "centralized": "Centralized", "linear": "Linear"}
STRUCT_FOLDER = {s: f"gpt-4.1_{s}" for s in STRUCTURES}

PRICE_FILE = {t: PRICE_DIR / f"{t}-YFin-data-2025-01-01-2026-03-31.csv" for t in TICKERS}


def parse_decision(text):
    if pd.isna(text):
        return "UNKNOWN"
    text = str(text).upper()
    if "BUY" in text:
        return "BUY"
    if "SELL" in text:
        return "SELL"
    if "HOLD" in text:
        return "HOLD"
    return "UNKNOWN"


def load_prices(ticker):
    df = pd.read_csv(PRICE_FILE[ticker])
    df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df[["date", "Close"]].rename(columns={"Close": "close"})


def calculate_metrics(decisions_df, prices_df, initial_capital=100000.0, risk_free_rate=0.02):
    merged = decisions_df.merge(prices_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if merged.empty:
        return {"CR%": 0.0, "MDD%": 0.0, "trading_days": 0}

    position = 0
    entry_price = 0.0
    invested_amount = 0.0
    capital = initial_capital
    equity_curve = [capital]

    for _, row in merged.iterrows():
        decision = parse_decision(row["decision"])
        price = row["close"]

        if decision == "BUY" and position == 0:
            position = 1
            entry_price = price
            invested_amount = capital

        elif decision == "SELL" and position == 1:
            position = 0
            trade_return = (price - entry_price) / entry_price
            capital = capital + invested_amount * trade_return
            invested_amount = 0.0

        if position == 1:
            uninvested = capital - invested_amount
            current_position_value = (invested_amount / entry_price) * price
            equity_curve.append(uninvested + current_position_value)
        else:
            equity_curve.append(capital)

    if position == 1:
        final_price = merged.iloc[-1]["close"]
        trade_return = (final_price - entry_price) / entry_price
        capital = capital + invested_amount * trade_return
        equity_curve[-1] = capital

    equity_curve = np.array(equity_curve)
    cr = (equity_curve[-1] - initial_capital) / initial_capital * 100
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    mdd = np.min(drawdown) * 100

    return {"CR%": round(cr, 2), "MDD%": round(mdd, 2), "trading_days": len(merged)}


rows = []
for structure in STRUCTURES:
    folder = TRADING_DIR / STRUCT_FOLDER[structure]
    for ticker in TICKERS:
        decisions_path = folder / f"{ticker}_results.csv"
        ddf = pd.read_csv(decisions_path)
        ddf = ddf[ddf["status"] == "success"][["date", "decision"]]
        prices = load_prices(ticker)
        m = calculate_metrics(ddf, prices)
        rows.append({"Structure": STRUCT_LABEL[structure], "Ticker": ticker,
                     "CR (%)": m["CR%"], "MDD (%)": m["MDD%"], "Trading days": m["trading_days"]})

result = pd.DataFrame(rows)
out_path = BASE_DIR / "financial_impact_q1_2026_gpt41.csv"
result.to_csv(out_path, index=False)

print(result.to_string(index=False))
print()

cr_pivot = result.pivot(index="Ticker", columns="Structure", values="CR (%)")[
    [STRUCT_LABEL[s] for s in STRUCTURES]].reindex(TICKERS)
mdd_pivot = result.pivot(index="Ticker", columns="Structure", values="MDD (%)")[
    [STRUCT_LABEL[s] for s in STRUCTURES]].reindex(TICKERS)
cr_pivot.loc["Average"] = cr_pivot.mean().round(2)
mdd_pivot.loc["Average"] = mdd_pivot.mean().round(2)

print("=== CR (%) ===")
print(cr_pivot.to_string())
print()
print("=== MDD (%) ===")
print(mdd_pivot.to_string())
print()
print(f"Saved long-form table to {out_path}")
