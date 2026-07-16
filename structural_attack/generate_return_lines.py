"""
Generate Cumulative Return & Rolling Variance line charts per ticker.
Layout: 1 figure per ticker, 2 rows (CR / variance) × 5 cols (structures).
x-axis: date, lines: scenarios (color-coded, Clean always on top).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

BASE        = Path(__file__).parent
RESULTS_DIR = BASE / "trading_results" / "2026-01-01_2026-03-31"
PRICE_DIR   = BASE / "tradingagents" / "dataflows" / "local_data" / "market_data" / "price_data"
OUT_DIR     = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)

TICKERS    = ["AAPL", "BTC-USD", "MSFT", "NVDA", "TSLA"]
STRUCTURES = ["linear", "centralized", "decentralized", "moa", "hybrid"]
STRUCT_LABELS = {
    "linear": "Linear", "centralized": "Centralized",
    "decentralized": "Decentralized", "moa": "MoA", "hybrid": "Hybrid",
}

SCENARIOS = [
    ("Clean",   "gpt-4.1_{struct}"),
    ("D1-BUY",  "adv_gpt_D1_buy_{struct}"),
    ("D1-SELL", "adv_gpt_D1_sell_{struct}"),
    ("D2-BUY",  "adv_gpt_D2_buy_{struct}"),
    ("D2-SELL", "adv_gpt_D2_sell_{struct}"),
    ("A1-BUY",  "adv_gpt_A1_buy_{struct}"),
    ("A1-SELL", "adv_gpt_A1_sell_{struct}"),
    ("A2-BUY",  "adv_gpt_A2_buy_{struct}"),
    ("A2-SELL", "adv_gpt_A2_sell_{struct}"),
    ("A3-BUY",  "adv_gpt_A3_buy_{struct}"),
    ("A3-SELL", "adv_gpt_A3_sell_{struct}"),
]

# (color, linestyle, linewidth, zorder)
COLORS = {
    "Clean":   ("#000000", "-",  3.0, 10),
    "D1-BUY":  ("#1565C0", "-",  1.3,  3),
    "D1-SELL": ("#42A5F5", "--", 1.3,  3),
    "D2-BUY":  ("#0D47A1", "-",  1.3,  3),
    "D2-SELL": ("#90CAF9", "--", 1.3,  3),
    "A1-BUY":  ("#E65100", "-",  1.3,  3),
    "A1-SELL": ("#FFCC80", "--", 1.3,  3),
    "A2-BUY":  ("#BF360C", "-",  1.3,  3),
    "A2-SELL": ("#FF8A65", "--", 1.3,  3),
    "A3-BUY":  ("#B71C1C", "-",  1.8,  4),
    "A3-SELL": ("#EF9A9A", "--", 1.8,  4),
}

ROLL_WIN = 10  # rolling variance window (trading days)


def load_price(ticker: str) -> pd.DataFrame | None:
    path = PRICE_DIR / f"{ticker}-YFin-data-2025-01-01-2026-03-31.csv"
    if not path.exists():
        candidates = sorted(PRICE_DIR.glob(f"{ticker}-YFin-data-*.csv"))
        if not candidates:
            return None
        path = candidates[-1]
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["Date"] >= "2026-01-01"].sort_values("Date").reset_index(drop=True)
    return df[["Date", "Close"]]


def compute_series(decisions_df: pd.DataFrame,
                   prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame[cum_ret_pct, roll_var] indexed by Date.
    Strategy: long-only, 1-day signal lag.
    """
    merged = (
        decisions_df[["date", "decision"]]
        .rename(columns={"date": "Date"})
        .merge(prices_df, on="Date", how="inner")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if merged.empty:
        return pd.DataFrame(columns=["cum_ret_pct", "roll_var"])

    position, positions = 0, []
    for dec in merged["decision"].str.upper():
        if dec == "BUY" and position == 0:
            position = 1
        elif dec == "SELL" and position == 1:
            position = 0
        positions.append(position)

    merged["pos"]         = positions
    merged["price_ret"]   = merged["Close"].pct_change().fillna(0)
    merged["strat_ret"]   = merged["pos"].shift(1).fillna(0) * merged["price_ret"]
    merged["cum_ret_pct"] = ((1 + merged["strat_ret"]).cumprod() - 1) * 100
    merged["roll_var"]    = (
        merged["strat_ret"]
        .rolling(window=ROLL_WIN, min_periods=2)
        .var() * 10_000
    )
    return merged.set_index("Date")[["cum_ret_pct", "roll_var"]]


def plot_ticker(ticker: str, prices: pd.DataFrame):
    # Draw attack lines first, Clean last so it's always on top
    ordered = [s for s in SCENARIOS if s[0] != "Clean"] + \
              [s for s in SCENARIOS if s[0] == "Clean"]

    # Pre-compute all series
    cache: dict[str, dict[str, pd.DataFrame]] = {s: {} for s in STRUCTURES}
    for struct in STRUCTURES:
        for label, dir_tpl in ordered:
            csv_path = RESULTS_DIR / dir_tpl.format(struct=struct) / f"{ticker}_results.csv"
            if not csv_path.exists():
                continue
            dec_df = pd.read_csv(csv_path, parse_dates=["date"])
            cache[struct][label] = compute_series(dec_df, prices)

    fig, axes = plt.subplots(2, 5, figsize=(30, 9), sharey="row",
                             gridspec_kw={"hspace": 0.45})
    fig.suptitle(
        f"Cumulative Return & Rolling Variance ({ROLL_WIN}-day) — {ticker}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    row_cfg = [
        (0, "cum_ret_pct", "Cumulative Return (%)"),
        (1, "roll_var",    f"Rolling Variance (×10⁻⁴, {ROLL_WIN}-day)"),
    ]

    for row, col_key, ylabel in row_cfg:
        for col, struct in enumerate(STRUCTURES):
            ax = axes[row][col]
            if row == 0:
                ax.set_title(STRUCT_LABELS[struct], fontsize=12, fontweight="bold")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

            for label, _ in ordered:
                df = cache[struct].get(label)
                if df is None or df.empty:
                    continue
                ser = df[col_key].dropna()
                if ser.empty:
                    continue

                color, ls, lw, zo = COLORS[label]
                ax.plot(ser.index, ser.values,
                        label=label, color=color, linestyle=ls,
                        linewidth=lw, alpha=(1.0 if label == "Clean" else 0.75),
                        zorder=zo)

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.xaxis.set_major_locator(
                mdates.WeekdayLocator(byweekday=mdates.MO, interval=3))
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
            ax.set_xlabel("Date", fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(axis="both", alpha=0.25)

    # Shared legend (collect once, dedup)
    handles, lbls, seen = [], [], set()
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h); lbls.append(l); seen.add(l)

    fig.legend(handles, lbls, loc="lower center", ncol=11,
               fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    out = OUT_DIR / f"fig_cr_var_line_{ticker.replace('-', '')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


def main():
    for ticker in TICKERS:
        print(f"Processing {ticker} ...")
        prices = load_price(ticker)
        if prices is None or prices.empty:
            print(f"  [SKIP] no price data")
            continue
        plot_ticker(ticker, prices)
    print("Done.")


if __name__ == "__main__":
    main()
