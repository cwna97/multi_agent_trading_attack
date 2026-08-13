"""
Qwen 결과를 baseline_reports 형식으로 저장하는 스크립트.

실행 순서:
  1. python main.py  (trading_results/2025-07-01_2025-09-30/ 에 결과 생성)
  2. python save_qwen_baseline.py  (baseline_reports/qwen2.5_72b_2025_Q3.csv 생성)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

RESULTS_CSV  = "trading_results/2025-07-01_2025-09-30/all_tickers_results.csv"
PRICE_DIR    = "tradingagents/dataflows/local_data/market_data/price_data"
OUTPUT_CSV   = "baseline_reports/qwen2.5_72b_2025_Q3.csv"
STRATEGY     = "Qwen2.5-72B"
START_DATE   = "2025-07-01"
END_DATE     = "2025-09-30"
SEED_MONEY   = 10_000.0
COMMISSION   = 0.001
SLIPPAGE     = 0.0005
RISK_FREE    = 0.0       # daily risk-free rate


def load_prices(ticker: str) -> pd.DataFrame:
    """Load local price data for a ticker."""
    for f in Path(PRICE_DIR).glob(f"{ticker}-YFin-data-*.csv"):
        df = pd.read_csv(f, parse_dates=["Date"])
        df["Date"] = df["Date"].dt.tz_localize(None)
        df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
        df = df.rename(columns={"Date": "date", "Close": "close"})
        return df[["date", "close"]].sort_values("date").reset_index(drop=True)
    raise FileNotFoundError(f"No local price file for {ticker}")


def simulate(decisions: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Run simple long-only portfolio simulation."""
    decisions = decisions.copy()
    decisions["date"] = pd.to_datetime(decisions["date"]).dt.tz_localize(None)
    merged = decisions.merge(prices, on="date", how="inner").sort_values("date")

    if merged.empty:
        return None

    cash, shares, position = SEED_MONEY, 0.0, "OUT"
    daily_values, trades = [], []

    for _, row in merged.iterrows():
        dec   = str(row["decision"]).upper()
        price = float(row["close"])

        if dec == "BUY" and position == "OUT":
            ep       = price * (1 + SLIPPAGE)
            comm     = cash * COMMISSION
            shares   = (cash - comm) / ep
            cash     = 0.0
            position = "IN"
            trades.append({"action": "BUY", "price": price})

        elif dec == "SELL" and position == "IN":
            ep       = price * (1 - SLIPPAGE)
            proceeds = shares * ep
            comm     = proceeds * COMMISSION
            cash     = proceeds - comm
            trades.append({"action": "SELL", "price": price, "entry_price": trades[-1]["price"] if trades else price})
            shares   = 0.0
            position = "OUT"

        daily_values.append(cash + shares * price)

    final_value = daily_values[-1] if daily_values else SEED_MONEY

    # CR%
    cr = (final_value - SEED_MONEY) / SEED_MONEY * 100

    # ARR% (annualised)
    n_days = len(merged)
    arr = ((1 + cr / 100) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0.0

    # MDD%
    vals  = pd.Series(daily_values)
    peaks = vals.cummax()
    mdd   = ((vals - peaks) / peaks * 100).min() if len(vals) > 1 else 0.0

    # Sharpe Ratio (daily returns)
    if len(daily_values) > 1:
        rets   = pd.Series(daily_values).pct_change().dropna()
        excess = rets - RISK_FREE
        sr     = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
    else:
        sr = 0.0

    # Win rate (profitable round-trips)
    buy_prices  = [t["price"] for t in trades if t["action"] == "BUY"]
    sell_prices = [t["price"] for t in trades if t["action"] == "SELL"]
    round_trips = list(zip(buy_prices, sell_prices))
    wins        = sum(1 for b, s in round_trips if s > b)
    total_rt    = len(round_trips)
    win_rate    = round(wins / total_rt * 100, 1) if total_rt > 0 else 0.0
    total_trades = len(trades)

    return {
        "CR%":          round(cr,  2),
        "ARR%":         round(arr, 2),
        "SR":           round(sr,  3),
        "MDD%":         round(mdd, 2),
        "total_trades": total_trades,
        "win_rate":     win_rate,
    }


def main():
    print(f"Reading decisions from: {RESULTS_CSV}")
    if not Path(RESULTS_CSV).exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run main.py first.")
        return

    all_decisions = pd.read_csv(RESULTS_CSV)
    tickers = sorted(all_decisions["ticker"].unique())
    print(f"Tickers found: {tickers}\n")

    rows = []
    for ticker in tickers:
        print(f"  [{ticker}] ", end="", flush=True)
        ticker_dec = all_decisions[all_decisions["ticker"] == ticker]

        try:
            prices  = load_prices(ticker)
            metrics = simulate(ticker_dec, prices)
            if metrics is None:
                print("SKIP (no overlapping dates)")
                continue

            row = {
                "CR%":          metrics["CR%"],
                "ARR%":         metrics["ARR%"],
                "SR":           metrics["SR"],
                "MDD%":         metrics["MDD%"],
                "total_trades": metrics["total_trades"],
                "win_rate":     metrics["win_rate"],
                "strategy":     STRATEGY,
                "ticker":       ticker,
            }
            rows.append(row)
            print(f"CR={metrics['CR%']:+.2f}%  SR={metrics['SR']:.3f}  MDD={metrics['MDD%']:.2f}%  trades={metrics['total_trades']}  wr={metrics['win_rate']}%")

        except Exception as e:
            print(f"ERROR: {e}")

    if not rows:
        print("\nNo results generated.")
        return

    out_df = pd.DataFrame(rows, columns=["CR%","ARR%","SR","MDD%","total_trades","win_rate","strategy","ticker"])
    Path("baseline_reports").mkdir(exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved → {OUTPUT_CSV}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
