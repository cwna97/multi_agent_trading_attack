"""
Per-Ticker Figure Generation
티커별 ASR, centrality-vulnerability 상관관계, CR 분석

생성 결과 (figures/06_perticker/):
  figT1_asr_heatmap_perticker.png  — 5개 티커별 ASR heatmap (attack × structure)
  figT2_scatter_perticker.png      — 티커별 centrality-ASR scatter (5 panels)
  figT3_rho_perticker.png          — 티커별 Spearman ρ 일관성 (grouped bar)
  figT4_cr_perticker.png           — 티커별 CR degradation (5 tickers × 4 structures)
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
RESULTS_DIR = BASE / "trading_results" / "2026-01-01_2026-03-31"
PRICE_DIR   = BASE / "tradingagents" / "dataflows" / "local_data" / "market_data" / "price_data"
FIG_BASE    = BASE / "figures" / "06_perticker"
FIG_BASE.mkdir(parents=True, exist_ok=True)

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
DAS = {"decentralized": 0.33, "hybrid": 0.60, "centralized": 1.00, "linear": 1.00}

ATTACKS_MAIN = ["D1", "D2", "A1", "A2"]
ATTACK_LABELS = {
    "D1": "D1 (Data Poison)",
    "D2": "D2 (Indirect Inj.)",
    "A1": "A1 (1-Analyst)",
    "A2": "A2 (2-Analyst)",
}
TICKERS = ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]
TICKER_COLORS = {
    "BTC-USD": "#F7931A",
    "MSFT":    "#00A4EF",
    "NVDA":    "#76B900",
    "TSLA":    "#E82127",
    "AAPL":    "#999999",
}

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
    return df[["ticker", "date", "signal"]].rename(columns={"signal": "clean_signal"})

def load_attacked(struct, attack, direction):
    path = RESULTS_DIR / f"adv_gpt_{attack}_{direction.lower()}_{struct}" / "all_tickers_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["signal"] = df["decision"].apply(extract_signal)
    return df[["ticker", "date", "signal"]].rename(columns={"signal": "atk_signal"})

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("Loading decision data...")
clean_data = {struct: load_clean(struct) for struct in STRUCTURES}

# per-ticker ASR 계산
records = []
for struct in STRUCTURES:
    clean_df = clean_data[struct]
    for atk in ATTACKS_MAIN:
        for direction in ["BUY", "SELL"]:
            atk_df = load_attacked(struct, atk, direction)
            if atk_df is None:
                continue
            merged = clean_df.merge(atk_df, on=["ticker", "date"], how="inner")
            # 전체 aggregate
            valid = merged[merged["atk_signal"].isin(["BUY", "SELL", "HOLD"])]
            n_all = len(valid)
            if n_all > 0:
                records.append({
                    "ticker": "ALL",
                    "structure": struct,
                    "attack": atk,
                    "direction": direction,
                    "ASR": round((valid["atk_signal"] == direction).sum() / n_all * 100, 1),
                    "n": n_all,
                })
            # per-ticker
            for ticker in TICKERS:
                sub = merged[merged["ticker"] == ticker]
                valid_t = sub[sub["atk_signal"].isin(["BUY", "SELL", "HOLD"])]
                n_t = len(valid_t)
                if n_t > 0:
                    records.append({
                        "ticker": ticker,
                        "structure": struct,
                        "attack": atk,
                        "direction": direction,
                        "ASR": round((valid_t["atk_signal"] == direction).sum() / n_t * 100, 1),
                        "n": n_t,
                    })

df_all = pd.DataFrame(records)
df_buy = df_all[df_all["direction"] == "BUY"].copy()
print(f"  Records: {len(df_all)} (BUY: {len(df_buy)})")

# ── 가격 데이터 로드 ──────────────────────────────────────────────────────────
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

prices = load_prices()
all_dates = sorted({d for p in prices.values() for d in p})
date_idx  = {d: i for i, d in enumerate(all_dates)}

def compute_cr_ticker(decisions_df, ticker, prices_dict):
    """단일 티커 CR 시계열 계산"""
    t_dec = (decisions_df[decisions_df["ticker"] == ticker]
             [["date", "signal"]].sort_values("date").reset_index(drop=True))
    all_d = sorted(prices_dict.get(ticker, {}).keys())
    d_idx = {d: i for i, d in enumerate(all_d)}

    position = 0
    daily_ret = {}
    for _, row in t_dec.iterrows():
        sig, d_str = row["signal"], row["date"]
        if sig == "BUY" and position == 0:
            position = 1
        elif sig == "SELL" and position == 1:
            position = 0
        idx = d_idx.get(d_str)
        if idx is None or idx + 1 >= len(all_d):
            continue
        p0 = prices_dict[ticker].get(d_str)
        p1 = prices_dict[ticker].get(all_d[idx + 1])
        if p0 and p1 and p0 != 0:
            daily_ret[d_str] = position * (p1 - p0) / p0

    if not daily_ret:
        return pd.Series(dtype=float)
    dates_sorted = sorted(daily_ret.keys())
    vals = [daily_ret[d] for d in dates_sorted]
    cum  = (np.cumprod([1 + r for r in vals]) - 1) * 100
    return pd.Series(cum, index=dates_sorted)

# ══════════════════════════════════════════════════════════════════════════════
# FIG T1: 티커별 ASR Heatmap (5 panels, attack × structure)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig T1: Per-ticker ASR heatmap (5 panels)...")

fig, axes = plt.subplots(1, 5, figsize=(22, 5))

for ax, ticker in zip(axes, TICKERS):
    pivot = pd.DataFrame(index=ATTACKS_MAIN,
                         columns=[STRUCT_LABELS[s] for s in STRUCTURES])
    for atk in ATTACKS_MAIN:
        for struct in STRUCTURES:
            sub = df_buy[(df_buy["ticker"] == ticker) &
                         (df_buy["attack"] == atk) &
                         (df_buy["structure"] == struct)]
            pivot.loc[atk, STRUCT_LABELS[struct]] = sub["ASR"].values[0] if len(sub) else np.nan

    pivot = pivot.astype(float)
    # aggregate ASR for comparison
    agg_asrs = []
    for atk in ATTACKS_MAIN:
        for struct in STRUCTURES:
            sub_all = df_buy[(df_buy["ticker"] == "ALL") &
                             (df_buy["attack"] == atk) &
                             (df_buy["structure"] == struct)]
            agg_asrs.append(sub_all["ASR"].values[0] if len(sub_all) else np.nan)

    sns.heatmap(pivot, ax=ax, annot=True, fmt=".0f", cmap="RdYlGn_r",
                vmin=0, vmax=100, linewidths=0.5, linecolor="white",
                annot_kws={"size": 11, "weight": "bold"},
                cbar=(ax == axes[-1]),
                cbar_kws={"label": "ASR (%)"} if ax == axes[-1] else {})
    ax.set_title(ticker, fontsize=13, fontweight="bold",
                 color=TICKER_COLORS.get(ticker, "#333"))
    ax.set_xlabel("Structure (ACS ↑)", fontsize=9)
    if ax == axes[0]:
        ax.set_ylabel("Attack Type", fontsize=10)
    else:
        ax.set_ylabel("")
    ax.set_xticklabels([f"{STRUCT_LABELS[s]}\n({DAS[s]:.2f})" for s in STRUCTURES],
                       fontsize=8, rotation=15, ha="right")
    ax.set_yticklabels(ATTACKS_MAIN, fontsize=10, rotation=0)

plt.suptitle("Per-Ticker ASR (%) — BUY-Targeted Attacks\n"
             "Structures ordered by Centrality Score (ACS ↑)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_BASE / "figT1_asr_heatmap_perticker.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved figT1_asr_heatmap_perticker.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG T2: 티커별 Centrality-ASR Scatter (5 panels)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig T2: Per-ticker centrality-ASR scatter (5 panels)...")

ATTACK_MARKERS = {"D1": "o", "D2": "s", "A1": "^", "A2": "D"}
ATTACK_COLORS  = {"D1": "#1565C0", "D2": "#0288D1", "A1": "#E65100", "A2": "#BF360C"}

fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

for ax, ticker in zip(axes, TICKERS):
    acs_vals = np.array([DAS[s] for s in STRUCTURES])

    for atk in ATTACKS_MAIN:
        xs, ys = [], []
        for struct in STRUCTURES:
            sub = df_buy[(df_buy["ticker"] == ticker) &
                         (df_buy["attack"] == atk) &
                         (df_buy["structure"] == struct)]
            if len(sub):
                xs.append(DAS[struct])
                ys.append(sub["ASR"].values[0])

        ax.scatter(xs, ys, s=80, marker=ATTACK_MARKERS[atk],
                   color=ATTACK_COLORS[atk], zorder=5,
                   label=ATTACK_LABELS[atk], edgecolors="white", linewidth=0.6)

    # 평균 trend line (D1+D2+A1)
    avg_asr = np.array([
        df_buy[(df_buy["ticker"] == ticker) &
               (df_buy["attack"].isin(["D1","D2","A1"])) &
               (df_buy["structure"] == s)]["ASR"].mean()
        for s in STRUCTURES
    ])
    valid = ~np.isnan(avg_asr)
    if valid.sum() >= 2:
        sort_idx = np.argsort(acs_vals[valid])
        ax.plot(acs_vals[valid][sort_idx], avg_asr[valid][sort_idx],
                "k--", linewidth=1.2, alpha=0.45, zorder=2)

    # Spearman ρ 표시
    rho_vals = []
    for atk in ["D1","D2","A1"]:
        asr_a = np.array([
            df_buy[(df_buy["ticker"] == ticker) &
                   (df_buy["attack"] == atk) &
                   (df_buy["structure"] == s)]["ASR"].values[0]
            if len(df_buy[(df_buy["ticker"] == ticker) &
                          (df_buy["attack"] == atk) &
                          (df_buy["structure"] == s)]) else np.nan
            for s in STRUCTURES
        ])
        if not np.any(np.isnan(asr_a)):
            r, _ = spearmanr(acs_vals, asr_a)
            rho_vals.append(r)
    avg_rho = np.mean(rho_vals) if rho_vals else np.nan
    ax.text(0.05, 0.95, f"ρ(D1+D2+A1)={avg_rho:.2f}" if not np.isnan(avg_rho) else "ρ=N/A",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_title(ticker, fontsize=12, fontweight="bold",
                 color=TICKER_COLORS.get(ticker, "#333"))
    ax.set_xlabel("ACS", fontsize=9)
    if ax == axes[0]:
        ax.set_ylabel("ASR (%)", fontsize=11)
    ax.set_ylim(-5, 115)
    ax.set_xlim(0.2, 1.1)
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.7, alpha=0.4)
    ax.yaxis.grid(True, alpha=0.2); ax.set_axisbelow(True)

    for struct in STRUCTURES:
        ax.axvline(DAS[struct], color=PALETTE[struct], linestyle=":", alpha=0.25, linewidth=0.7)

# 공통 legend
handles = [plt.scatter([], [], s=80, marker=ATTACK_MARKERS[a],
                       color=ATTACK_COLORS[a], label=ATTACK_LABELS[a])
           for a in ATTACKS_MAIN]
handles += [plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.2,
                        alpha=0.45, label="D1+D2+A1 avg (trend)")]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 1.03))

plt.suptitle("Centrality vs. Vulnerability per Ticker\n(BUY-targeted attacks, D1/D2/A1/A2)",
             fontsize=13, fontweight="bold", y=1.10)
plt.tight_layout()
plt.savefig(FIG_BASE / "figT2_scatter_perticker.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved figT2_scatter_perticker.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG T3: 티커별 Spearman ρ 일관성 (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig T3: Per-ticker Spearman rho consistency...")

acs_arr = np.array([DAS[s] for s in STRUCTURES])
rho_data = {}  # {attack: {ticker: rho}}

for atk in ATTACKS_MAIN + ["D1+D2+A1 avg"]:
    rho_data[atk] = {}
    for ticker in TICKERS + ["ALL"]:
        if atk == "D1+D2+A1 avg":
            asr_a = np.array([
                df_buy[(df_buy["ticker"] == ticker) &
                       (df_buy["attack"].isin(["D1","D2","A1"])) &
                       (df_buy["structure"] == s)]["ASR"].mean()
                for s in STRUCTURES
            ])
        else:
            asr_a = np.array([
                df_buy[(df_buy["ticker"] == ticker) &
                       (df_buy["attack"] == atk) &
                       (df_buy["structure"] == s)]["ASR"].values[0]
                if len(df_buy[(df_buy["ticker"] == ticker) &
                              (df_buy["attack"] == atk) &
                              (df_buy["structure"] == s)]) else np.nan
                for s in STRUCTURES
            ])
        if np.any(np.isnan(asr_a)):
            rho_data[atk][ticker] = np.nan
        else:
            r, _ = spearmanr(acs_arr, asr_a)
            rho_data[atk][ticker] = round(r, 3)

fig, ax = plt.subplots(figsize=(14, 5))
atk_labels_plot = ATTACKS_MAIN + ["D1+D2+A1 avg"]
x = np.arange(len(atk_labels_plot))
n_tickers = len(TICKERS) + 1  # +1 for ALL (aggregate)
width = 0.13

ticker_list = TICKERS + ["ALL"]
ticker_display_colors = [TICKER_COLORS.get(t, "#555") for t in TICKERS] + ["#333333"]
ticker_linestyles = ["-", "-", "-", "-", "-", "--"]
ticker_hatch = ["", "", "", "", "", "//"]

for i, (ticker, tc, ls, hatch) in enumerate(zip(ticker_list, ticker_display_colors, ticker_linestyles, ticker_hatch)):
    rhos = [rho_data[atk].get(ticker, np.nan) for atk in atk_labels_plot]
    offset = (i - (n_tickers - 1) / 2) * width
    bars = ax.bar(x + offset, rhos, width,
                  label=ticker, color=tc, alpha=0.80,
                  edgecolor="white", hatch=hatch, linewidth=0.5)

ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="ρ=0.7 reference")
ax.set_xticks(x)
ax.set_xticklabels(atk_labels_plot, fontsize=11)
ax.set_ylabel("Spearman ρ (ACS, ASR)", fontsize=12)
ax.set_ylim(-1.1, 1.3)
ax.set_title("Per-Ticker Centrality-Vulnerability Correlation (Spearman ρ)\n"
             "Consistency across tickers validates the architectural vulnerability hypothesis",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="lower right", ncol=3)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(FIG_BASE / "figT3_rho_perticker.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved figT3_rho_perticker.png")

# ρ 값 출력
print("\n  Spearman ρ per ticker:")
for atk in atk_labels_plot:
    vals = [f"{rho_data[atk].get(t, np.nan):.2f}" for t in ticker_list]
    print(f"    {atk:18s}: {', '.join(vals)}")

# ══════════════════════════════════════════════════════════════════════════════
# FIG T4: 티커별 Return Distribution (5 tickers × 4 structures, violin plot)
# per-decision return: BUY→+r, SELL→-r, HOLD→0
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig T4: Per-ticker return distribution (violin)...")

def compute_decision_returns(decisions_df, ticker, prices_dict):
    """
    각 날짜의 decision → next-day return 변환
    BUY  → +(p_{t+1}-p_t)/p_t  (롱 수익)
    SELL → -(p_{t+1}-p_t)/p_t  (숏 수익)
    HOLD → 0
    """
    all_d = sorted(prices_dict.get(ticker, {}).keys())
    d_idx = {d: i for i, d in enumerate(all_d)}
    sub = decisions_df[decisions_df["ticker"] == ticker].copy()
    rets = []
    for _, row in sub.iterrows():
        sig, date = row["signal"], row["date"]
        if sig not in ("BUY", "SELL", "HOLD"):
            continue
        idx = d_idx.get(date)
        if idx is None or idx + 1 >= len(all_d):
            continue
        p0 = prices_dict[ticker].get(date)
        p1 = prices_dict[ticker].get(all_d[idx + 1])
        if p0 and p1 and p0 != 0:
            r = (p1 - p0) / p0
            if sig == "BUY":
                rets.append(r * 100)
            elif sig == "SELL":
                rets.append(-r * 100)
            else:  # HOLD
                rets.append(0.0)
    return rets

CONDITION_COLORS = {
    "Clean":  "#444444",
    "D1-BUY": "#1565C0",
    "A2-BUY": "#BF360C",
}
CONDITIONS = ["Clean", "D1-BUY", "A2-BUY"]

fig, axes = plt.subplots(5, 4, figsize=(20, 24), sharey=False)

for row_idx, ticker in enumerate(TICKERS):
    for col_idx, struct in enumerate(STRUCTURES):
        ax = axes[row_idx, col_idx]

        plot_data = {}

        # Clean
        clean_df = clean_data[struct].rename(columns={"clean_signal": "signal"})
        plot_data["Clean"] = compute_decision_returns(clean_df, ticker, prices)

        # D1-BUY, A2-BUY
        for atk, label in [("D1", "D1-BUY"), ("A2", "A2-BUY")]:
            atk_raw = load_attacked(struct, atk, "BUY")
            if atk_raw is not None:
                atk_sig = atk_raw.rename(columns={"atk_signal": "signal"})
                plot_data[label] = compute_decision_returns(atk_sig, ticker, prices)
            else:
                plot_data[label] = []

        # violin + strip plot
        for x_pos, cond in enumerate(CONDITIONS):
            vals = plot_data.get(cond, [])
            if len(vals) < 2:
                ax.text(x_pos, 0, "N/A", ha="center", va="center", fontsize=7, color="#aaa")
                continue
            arr = np.array(vals)
            color = CONDITION_COLORS[cond]

            # violin
            parts = ax.violinplot([arr], positions=[x_pos], widths=0.7,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
            parts["cmedians"].set_color(color)
            parts["cmedians"].set_linewidth(2)

            # mean 점
            ax.scatter([x_pos], [arr.mean()], color=color, s=30, zorder=5,
                       edgecolors="white", linewidth=0.6)

            # std 텍스트
            ax.text(x_pos, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 3,
                    f"σ={arr.std():.2f}", ha="center", va="bottom",
                    fontsize=6.5, color=color, alpha=0.85)

        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(CONDITIONS, fontsize=7, rotation=15, ha="right")
        ax.yaxis.grid(True, alpha=0.2); ax.set_axisbelow(True)

        # 비HOLD 비율 표시 (실제 포지션 비중)
        clean_vals = np.array(plot_data["Clean"])
        nonzero_pct = (clean_vals != 0).sum() / len(clean_vals) * 100 if len(clean_vals) else 0
        ax.text(0.02, 0.98, f"active={nonzero_pct:.0f}%",
                transform=ax.transAxes, fontsize=6.5, va="top",
                color="#555", alpha=0.8)

        if row_idx == 0:
            ax.set_title(f"{STRUCT_LABELS[struct]}\n(ACS={DAS[struct]:.2f})",
                         fontsize=10, fontweight="bold", color=PALETTE[struct])
        if col_idx == 0:
            ax.set_ylabel(f"{ticker}\nReturn (%)", fontsize=9,
                          color=TICKER_COLORS.get(ticker, "#333"))
        if row_idx == 4:
            ax.set_xlabel("Condition", fontsize=8)

plt.suptitle("Per-Ticker Return Distribution: Clean vs D1-BUY vs A2-BUY\n"
             "(BUY→+r, SELL→−r, HOLD→0  |  active = non-HOLD %  |  σ = std dev)",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(FIG_BASE / "figT4_returns_perticker.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved figT4_returns_perticker.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG T5: 티커별 ASR variance 요약 (box plot)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig T5: Per-ticker ASR variance (box plot)...")

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

for ax, struct in zip(axes, STRUCTURES):
    data_by_ticker = []
    labels_box = []
    for ticker in TICKERS:
        asr_vals = []
        for atk in ATTACKS_MAIN:
            sub = df_buy[(df_buy["ticker"] == ticker) &
                         (df_buy["attack"] == atk) &
                         (df_buy["structure"] == struct)]
            if len(sub):
                asr_vals.append(sub["ASR"].values[0])
        data_by_ticker.append(asr_vals)
        labels_box.append(ticker)

    bp = ax.boxplot(data_by_ticker, labels=labels_box, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, ticker in zip(bp["boxes"], TICKERS):
        patch.set_facecolor(TICKER_COLORS.get(ticker, "#aaa"))
        patch.set_alpha(0.7)

    # aggregate ASR 점 표시
    for i, atk in enumerate(ATTACKS_MAIN):
        sub_all = df_buy[(df_buy["ticker"] == "ALL") &
                         (df_buy["attack"] == atk) &
                         (df_buy["structure"] == struct)]
        if len(sub_all):
            ax.axhline(sub_all["ASR"].values[0], color="gray", linestyle=":",
                       linewidth=0.8, alpha=0.4)

    ax.set_title(f"{STRUCT_LABELS[struct]}\n(ACS={DAS[struct]:.2f})",
                 fontsize=11, fontweight="bold", color=PALETTE[struct])
    ax.set_xticklabels(TICKERS, fontsize=9, rotation=20, ha="right")
    if ax == axes[0]:
        ax.set_ylabel("ASR (%) across D1/D2/A1/A2", fontsize=10)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(-5, 115)

plt.suptitle("Per-Ticker ASR Variance by Architecture\n"
             "(Box = distribution over D1/D2/A1/A2; dotted lines = aggregate ASR per attack)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_BASE / "figT5_asr_boxplot_perticker.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved figT5_asr_boxplot_perticker.png")

print("\nAll per-ticker figures saved to:", FIG_BASE)
print(sorted(p.name for p in FIG_BASE.glob("*.png")))
