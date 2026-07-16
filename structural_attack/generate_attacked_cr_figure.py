"""
figI4_cr_attacked.png 생성
Clean CR (실선) vs. 공격받은 CR (점선, D1+D2+A1+A2 평균) 비교
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE        = Path(__file__).parent
RESULTS_DIR = BASE / "trading_results" / "2026-01-01_2026-03-31"
PRICE_DIR   = BASE / "tradingagents" / "dataflows" / "local_data" / "market_data" / "price_data"
FIG_DIR     = BASE / "figures" / "07_insights"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURES = ["decentralized", "hybrid", "centralized", "linear"]
STRUCT_LABELS = {
    "decentralized": "Decentralized",
    "hybrid":        "Hybrid",
    "centralized":   "Centralized",
    "linear":        "Linear",
}
PALETTE = {
    "decentralized": "#4C72B0",
    "hybrid":        "#7B2D8B",
    "centralized":   "#DD8452",
    "linear":        "#C44E52",
}
ATTACKS_MAIN = ["D1", "D2", "A1", "A2"]
TICKERS = ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]

# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def extract_signal(d):
    if not isinstance(d, str): return "UNKNOWN"
    if d.startswith("ERROR:"): return "ERROR"
    m = re.search(r"FINAL TRANSACTION PROPOSAL[:\s*]+\*{0,2}(BUY|SELL|HOLD)\*{0,2}", d, re.IGNORECASE)
    if m: return m.group(1).upper()
    t = re.findall(r"\b(BUY|SELL|HOLD)\b", d, re.IGNORECASE)
    return t[-1].upper() if t else "UNKNOWN"

def load_clean(struct):
    path = RESULTS_DIR / f"gpt-4.1_{struct}" / "all_tickers_results.csv"
    df = pd.read_csv(path)
    df["signal"] = df["decision"].apply(extract_signal)
    return df[["ticker", "date", "signal"]]

def load_attacked(struct, attack, direction="BUY"):
    path = RESULTS_DIR / f"adv_gpt_{attack}_{direction.lower()}_{struct}" / "all_tickers_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["signal"] = df["decision"].apply(extract_signal)
    return df[["ticker", "date", "signal"]]

def load_prices():
    prices = {}
    for ticker in TICKERS:
        candidates = sorted(PRICE_DIR.glob(f"{ticker}-YFin-data-2025*.csv"))
        if not candidates:
            continue
        df = pd.read_csv(candidates[-1], parse_dates=["Date"])
        df = df[df["Date"] >= "2026-01-01"].copy()
        df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
        prices[ticker] = df.set_index("date")["Close"].to_dict()
    return prices

def compute_cr_series(decisions_df, prices_dict):
    """Long-only equal-weight portfolio, 1-day lag. Returns CR series."""
    all_d = sorted({d for p in prices_dict.values() for d in p})
    d_idx = {d: i for i, d in enumerate(all_d)}

    ticker_daily = {}
    for ticker in decisions_df["ticker"].unique():
        t_dec = (decisions_df[decisions_df["ticker"] == ticker]
                 [["date", "signal"]].sort_values("date").reset_index(drop=True))
        position = 0
        daily_ret = {}
        for _, row in t_dec.iterrows():
            sig, d_str = row["signal"], row["date"]
            if sig == "BUY"  and position == 0: position = 1
            elif sig == "SELL" and position == 1: position = 0
            idx = d_idx.get(d_str)
            if idx is None or idx + 1 >= len(all_d): continue
            p0 = prices_dict.get(ticker, {}).get(d_str)
            p1 = prices_dict.get(ticker, {}).get(all_d[idx + 1])
            if p0 and p1 and p0 != 0:
                daily_ret[d_str] = position * (p1 - p0) / p0
        ticker_daily[ticker] = daily_ret

    if not ticker_daily:
        return pd.Series(dtype=float)

    portfolio_daily = {}
    for d_str in all_d:
        rets = [v.get(d_str) for v in ticker_daily.values() if v.get(d_str) is not None]
        portfolio_daily[d_str] = np.mean(rets) if rets else 0.0

    dates_sorted = sorted(portfolio_daily.keys())
    vals = [portfolio_daily[d] for d in dates_sorted]
    cum  = (np.cumprod([1 + r for r in vals]) - 1) * 100
    return pd.Series(cum, index=dates_sorted)

def compute_bh_series(prices_dict):
    """Buy & Hold: equally weighted, hold all period."""
    all_d = sorted({d for p in prices_dict.values() for d in p})
    d_idx = {d: i for i, d in enumerate(all_d)}
    ticker_daily = {}
    for ticker, pmap in prices_dict.items():
        daily_ret = {}
        for d_str in all_d:
            idx = d_idx.get(d_str)
            if idx is None or idx + 1 >= len(all_d): continue
            p0 = pmap.get(d_str)
            p1 = pmap.get(all_d[idx + 1])
            if p0 and p1 and p0 != 0:
                daily_ret[d_str] = (p1 - p0) / p0
        ticker_daily[ticker] = daily_ret
    portfolio_daily = {}
    for d_str in all_d:
        rets = [v.get(d_str) for v in ticker_daily.values() if v.get(d_str) is not None]
        portfolio_daily[d_str] = np.mean(rets) if rets else 0.0
    dates_sorted = sorted(portfolio_daily.keys())
    vals = [portfolio_daily[d] for d in dates_sorted]
    cum  = (np.cumprod([1 + r for r in vals]) - 1) * 100
    return pd.Series(cum, index=dates_sorted)

# ── 데이터 계산 ───────────────────────────────────────────────────────────────
print("Loading prices...")
prices = load_prices()

print("Computing clean CR series...")
clean_series = {}
clean_cr_final = {}
for struct in STRUCTURES:
    df = load_clean(struct)
    ser = compute_cr_series(df, prices)
    clean_series[struct] = ser
    clean_cr_final[struct] = ser.iloc[-1] if len(ser) else 0.0

print("Computing attacked CR series (avg D1+D2+A1+A2)...")
attacked_series = {}
attacked_cr_final = {}
for struct in STRUCTURES:
    all_attack_series = []
    for atk in ATTACKS_MAIN:
        atk_df = load_attacked(struct, atk, "BUY")
        if atk_df is None:
            continue
        ser = compute_cr_series(atk_df, prices)
        if len(ser):
            all_attack_series.append(ser)

    if all_attack_series:
        # 날짜 기준으로 평균
        combined = pd.concat(all_attack_series, axis=1)
        avg_ser = combined.mean(axis=1)
        attacked_series[struct] = avg_ser
        attacked_cr_final[struct] = avg_ser.iloc[-1]
    else:
        attacked_series[struct] = pd.Series(dtype=float)
        attacked_cr_final[struct] = 0.0

print("Computing Buy & Hold...")
bh_series = compute_bh_series(prices)

# ── ASR 데이터 계산 ───────────────────────────────────────────────────────────
print("Computing ASR data...")

asr_data = {}
for struct in STRUCTURES:
    raw_clean = pd.read_csv(RESULTS_DIR / f"gpt-4.1_{struct}" / "all_tickers_results.csv")
    raw_clean["clean_signal"] = raw_clean["decision"].apply(extract_signal)
    clean_base = raw_clean[["ticker", "date", "clean_signal"]]

    asr_data[struct] = {}
    for atk in ATTACKS_MAIN:
        atk_raw = load_attacked(struct, atk, "BUY")
        if atk_raw is None:
            continue
        merged = clean_base.merge(atk_raw, on=["ticker", "date"], how="inner")
        valid  = merged[merged["signal"].isin(["BUY", "SELL", "HOLD"])]
        if len(valid) == 0:
            continue
        asr_data[struct][atk] = round((valid["signal"] == "BUY").sum() / len(valid) * 100, 1)

# ── 그림 생성 (2×2 서브플롯, y축=ASR) ────────────────────────────────────────
print("Generating figure...")

ATK_COLORS = {"D1": "#1565C0", "D2": "#0288D1", "A1": "#E65100", "A2": "#BF360C"}
ATK_LABELS = {"D1": "D1\n(Data Poison)", "D2": "D2\n(Indirect Inj.)",
              "A1": "A1\n(1-Analyst)",   "A2": "A2\n(2-Analyst)"}
acs_map    = {"decentralized": 0.33, "hybrid": 0.60, "centralized": 1.00, "linear": 1.00}

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
axes_flat = axes.flatten()

for ax, struct in zip(axes_flat, STRUCTURES):
    color    = PALETTE[struct]
    label    = STRUCT_LABELS[struct]
    asr_vals = asr_data.get(struct, {})

    x     = np.arange(len(ATTACKS_MAIN))
    vals  = [asr_vals.get(atk, 0) for atk in ATTACKS_MAIN]
    bars  = ax.bar(x, vals, color=[ATK_COLORS[a] for a in ATTACKS_MAIN],
                   edgecolor="white", linewidth=0.8, alpha=0.85, zorder=3)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    avg_asr = np.mean(vals)
    ax.axhline(avg_asr, color=color, linewidth=1.8, linestyle="--",
               alpha=0.8, label=f"Avg  {avg_asr:.1f}%", zorder=4)

    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels([ATK_LABELS[a] for a in ATTACKS_MAIN], fontsize=10)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=10)
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.set_title(f"{label}  (ACS = {acs_map[struct]:.2f})",
                 fontsize=11, fontweight="bold", color=color)

plt.suptitle("Attack Success Rate (ASR) by Architecture and Attack Type\n"
             "(BUY-targeted; dashed line = per-architecture average)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out = FIG_DIR / "figI4_cr_attacked.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── CR 단일 패널 그림 (서브플롯 없음) ─────────────────────────────────────────
print("Generating single-panel CR drop figure...")

acs_map = {"decentralized": 0.33, "hybrid": 0.60, "centralized": 1.00, "linear": 1.00}
bh_cr  = bh_series.iloc[-1] if len(bh_series) else 0.0

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
axes_flat = axes.flatten()

for ax, struct in zip(axes_flat, STRUCTURES):
    color = PALETTE[struct]
    label = STRUCT_LABELS[struct]
    ser_c = clean_series.get(struct, pd.Series())
    ser_a = attacked_series.get(struct, pd.Series())
    cr_c  = clean_cr_final[struct]
    cr_a  = attacked_cr_final[struct]
    delta = cr_a - cr_c

    # Buy & Hold
    ax.plot(range(len(bh_series)), bh_series.values,
            color="black", linewidth=1.4, linestyle="--",
            label=f"Buy & Hold ({bh_cr:.1f}%)", zorder=3, alpha=0.6)

    # Clean
    if len(ser_c):
        ax.plot(range(len(ser_c)), ser_c.values,
                color=color, linewidth=2.2, linestyle="-",
                label=f"Clean ({cr_c:.1f}%)", zorder=5)

    # Attacked + 음영
    if len(ser_a):
        ax.plot(range(len(ser_a)), ser_a.values,
                color=color, linewidth=1.6, linestyle=":",
                alpha=0.85, label=f"Attacked ({cr_a:.1f}%, Δ{delta:+.1f}%)", zorder=4)

        common_len = min(len(ser_c), len(ser_a))
        y_c = ser_c.values[:common_len]
        y_a = ser_a.values[:common_len]
        ax.fill_between(range(common_len), y_c, y_a,
                        where=(y_a < y_c), interpolate=True,
                        color=color, alpha=0.18, label=f"ΔCR = {delta:+.1f}%")

    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.set_xlabel("Trading Day", fontsize=10)
    ax.set_ylabel("Cumulative Return (%)", fontsize=10)
    ax.set_title(f"{label}  (ACS = {acs_map[struct]:.2f})",
                 fontsize=11, fontweight="bold", color=color)
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)

plt.suptitle("Clean vs. Attacked Cumulative Return by Architecture\n"
             "(Solid = Clean; Dotted = Attacked avg. D1+D2+A1+A2 BUY; shaded = ΔCR drop)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out2 = FIG_DIR / "figI4_cr_drop.png"
plt.savefig(out2, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")
