"""
Return Variance Figures
Decision(BUY/SELL/HOLD) → next-day return 변환 후 분포/분산 분석

생성 결과:
  figures/ret1_return_dist.png       — 구조별 return 분포 (baseline vs 공격)
  figures/ret2_variance_heatmap.png  — 공격별×구조별 return variance 히트맵
  figures/ret3_cumulative_pnl.png    — 누적 PnL (baseline vs 공격, 구조별)
  figures/ret4_sharpe_ratio.png      — Sharpe ratio 비교
  figures/ret5_mean_std_shift.png    — 공격으로 인한 mean·std 이동 scatter
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import glob
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "trading_results/2026-01-01_2026-03-31"
PRICE_DIR   = SCRIPT_DIR / "tradingagents/dataflows/local_data/market_data/price_data"
FIG_DIR     = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

STRUCTURES = ["linear", "centralized", "decentralized", "moa", "hybrid"]
ATTACKS    = ["D1", "D2", "A1", "A2", "A3"]
TICKERS    = ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]

STRUCT_LABELS = {"linear": "Linear", "centralized": "Centralized",
                 "decentralized": "Decentralized", "moa": "MoA", "hybrid": "Hybrid"}
PALETTE = {"linear": "#4C72B0", "centralized": "#DD8452",
           "decentralized": "#55A868", "moa": "#C44E52", "hybrid": "#7B2D8B"}

# ── 가격 데이터 로드 ────────────────────────────────────────────────────────

def load_prices() -> dict:
    """ticker → {date_str: close_price}"""
    prices = {}
    for ticker in TICKERS:
        files = sorted(glob.glob(str(PRICE_DIR / f"{ticker}-YFin-data-2025*.csv")))
        if not files:
            continue
        df = pd.read_csv(files[-1], parse_dates=["Date"])
        df = df[df["Date"] >= "2026-01-01"].copy()
        df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
        prices[ticker] = df.set_index("date")["Close"].to_dict()
    return prices

# ── signal 추출 ─────────────────────────────────────────────────────────────

def extract_signal(d: str) -> str:
    if not isinstance(d, str): return "UNKNOWN"
    if d.startswith("ERROR:"): return "ERROR"
    m = re.search(r"FINAL TRANSACTION PROPOSAL[:\s*]+\*{0,2}(BUY|SELL|HOLD)\*{0,2}",
                  d, re.IGNORECASE)
    if m: return m.group(1).upper()
    t = re.findall(r"\b(BUY|SELL|HOLD)\b", d, re.IGNORECASE)
    return t[-1].upper() if t else "UNKNOWN"

# ── next-day return 계산 ────────────────────────────────────────────────────

def compute_returns(df: pd.DataFrame, prices: dict) -> pd.Series:
    """
    BUY  → +(p_{t+1}-p_t)/p_t
    SELL → -(p_{t+1}-p_t)/p_t
    HOLD → 0
    """
    all_dates = sorted({d for p in prices.values() for d in p})
    date_idx  = {d: i for i, d in enumerate(all_dates)}
    returns = []
    for _, row in df.iterrows():
        ticker, date, signal = row["ticker"], row["date"], row["signal"]
        if signal not in ("BUY", "SELL", "HOLD") or ticker not in prices:
            continue
        if signal == "HOLD":
            returns.append(0.0)
            continue
        idx = date_idx.get(date)
        if idx is None or idx + 1 >= len(all_dates):
            continue
        p0 = prices[ticker].get(date)
        p1 = prices[ticker].get(all_dates[idx + 1])
        if p0 and p1 and p0 != 0:
            r = (p1 - p0) / p0
            returns.append(r if signal == "BUY" else -r)
    return pd.Series(returns)

def load_experiment(folder: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / folder / "all_tickers_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["signal"] = df["decision"].apply(extract_signal)
    return df

# ── 데이터 수집 ─────────────────────────────────────────────────────────────
print("Loading price data...")
prices = load_prices()

print("Loading experiment results...")
all_returns = {}  # (label, structure) → pd.Series

for s in STRUCTURES:
    df = load_experiment(f"gpt-4.1_{s}")
    if df is not None:
        all_returns[("Baseline", s)] = compute_returns(df, prices)
    for atk in ATTACKS:
        for sig in ["BUY", "SELL"]:
            df = load_experiment(f"adv_gpt_{atk}_{sig.lower()}_{s}")
            if df is not None:
                all_returns[(f"{atk}→{sig}", s)] = compute_returns(df, prices)

print(f"  Loaded {len(all_returns)} experiment return series")

# ══════════════════════════════════════════════════════════════════════════════
# FIG ret1: Return distribution violin (baseline vs 대표 공격 3개)
# ══════════════════════════════════════════════════════════════════════════════
print("ret1: return distribution violin...")
SHOW_ATTACKS  = ["Baseline", "D1→SELL", "A1→SELL", "A3→SELL"]
ATTACK_COLORS = {"Baseline": "#5B9BD5", "D1→SELL": "#E06C75",
                 "A1→SELL": "#E5C07B", "A3→SELL": "#C44E52"}

fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
for ax, struct in zip(axes, STRUCTURES):
    data, cols = [], []
    for label in SHOW_ATTACKS:
        r = all_returns.get((label, struct), pd.Series()).dropna()
        if len(r):
            data.append(r.clip(-0.15, 0.15).values)
            cols.append(ATTACK_COLORS[label])
    if not data:
        continue
    vp = ax.violinplot(data, positions=range(len(data)),
                       showmedians=True, showextrema=False, widths=0.7)
    for body, c in zip(vp["bodies"], cols):
        body.set_facecolor(c); body.set_alpha(0.75)
    vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(STRUCT_LABELS[struct], fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(SHOW_ATTACKS)))
    ax.set_xticklabels(SHOW_ATTACKS, rotation=30, ha="right", fontsize=8)
    if ax == axes[0]:
        ax.set_ylabel("Next-day Return", fontsize=11)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

handles = [mpatches.Patch(color=ATTACK_COLORS[l], label=l) for l in SHOW_ATTACKS]
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.02))
plt.suptitle("Return Distribution: Baseline vs. Selected Attacks (clipped ±15%)",
             fontsize=13, fontweight="bold", y=1.07)
plt.tight_layout()
plt.savefig(FIG_DIR / "ret1_return_dist.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved ret1")

# ══════════════════════════════════════════════════════════════════════════════
# FIG ret2: Return Variance & Mean Heatmap (attack × structure)
# ══════════════════════════════════════════════════════════════════════════════
print("ret2: variance heatmap...")
var_mat  = np.full((len(ATTACKS), len(STRUCTURES)), np.nan)
mean_mat = np.full((len(ATTACKS), len(STRUCTURES)), np.nan)

for j, struct in enumerate(STRUCTURES):
    for i, atk in enumerate(ATTACKS):
        vals_v, vals_m = [], []
        for sig in ["BUY", "SELL"]:
            r = all_returns.get((f"{atk}→{sig}", struct), pd.Series()).dropna()
            if len(r) > 1:
                vals_v.append(r.var())
                vals_m.append(r.mean())
        if vals_v:
            var_mat[i, j]  = np.mean(vals_v)
            mean_mat[i, j] = np.mean(vals_m)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
struct_xlabels = [STRUCT_LABELS[s] for s in STRUCTURES]

# --- variance panel
im1 = axes[0].imshow(var_mat * 1e4, cmap="YlOrRd", aspect="auto")
axes[0].set_xticks(range(len(STRUCTURES))); axes[0].set_xticklabels(struct_xlabels, fontsize=10)
axes[0].set_yticks(range(len(ATTACKS)));   axes[0].set_yticklabels(ATTACKS, fontsize=10)
axes[0].set_title("Return Variance (×10⁻⁴)", fontsize=12, fontweight="bold")
for i in range(len(ATTACKS)):
    for j in range(len(STRUCTURES)):
        if not np.isnan(var_mat[i, j]):
            txt = f"{var_mat[i,j]*1e4:.2f}"
            col = "white" if var_mat[i,j]*1e4 > np.nanmax(var_mat*1e4)*0.5 else "black"
            axes[0].text(j, i, txt, ha="center", va="center",
                         fontsize=10, fontweight="bold", color=col)
# baseline variance 행 추가 (x축 아래 주석)
base_vars = [all_returns.get(("Baseline", s), pd.Series()).dropna().var() for s in STRUCTURES]
for j, bv in enumerate(base_vars):
    axes[0].text(j, len(ATTACKS)-0.45, f"Base\n{bv*1e4:.2f}",
                 ha="center", va="top", fontsize=7, color="gray")
plt.colorbar(im1, ax=axes[0], shrink=0.85)

# --- mean return panel
vabs = max(abs(np.nanmax(mean_mat)), abs(np.nanmin(mean_mat)), 1e-5)
im2 = axes[1].imshow(mean_mat * 100, cmap="RdYlGn", aspect="auto",
                     vmin=-vabs*100, vmax=vabs*100)
axes[1].set_xticks(range(len(STRUCTURES))); axes[1].set_xticklabels(struct_xlabels, fontsize=10)
axes[1].set_yticks(range(len(ATTACKS)));   axes[1].set_yticklabels(ATTACKS, fontsize=10)
axes[1].set_title("Mean Next-day Return (%)", fontsize=12, fontweight="bold")
for i in range(len(ATTACKS)):
    for j in range(len(STRUCTURES)):
        if not np.isnan(mean_mat[i, j]):
            txt = f"{mean_mat[i,j]*100:.3f}"
            col = "white" if abs(mean_mat[i,j]*100) > vabs*50 else "black"
            axes[1].text(j, i, txt, ha="center", va="center",
                         fontsize=10, fontweight="bold", color=col)
plt.colorbar(im2, ax=axes[1], shrink=0.85)

plt.suptitle("Return Statistics by Attack and Graph Structure",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "ret2_variance_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved ret2")

# ══════════════════════════════════════════════════════════════════════════════
# FIG ret3: Cumulative PnL (baseline vs D1→SELL vs A3→SELL)
# ══════════════════════════════════════════════════════════════════════════════
print("ret3: cumulative PnL...")

def cumulative_pnl(df: pd.DataFrame, prices: dict) -> pd.Series:
    all_dates = sorted({d for p in prices.values() for d in p})
    date_idx  = {d: i for i, d in enumerate(all_dates)}
    daily = []
    for date in all_dates:
        day_df = df[df["date"] == date]
        rets = []
        for _, row in day_df.iterrows():
            ticker, signal = row["ticker"], row["signal"]
            if signal not in ("BUY", "SELL") or ticker not in prices:
                continue
            idx = date_idx.get(date)
            if idx is None or idx + 1 >= len(all_dates):
                continue
            p0 = prices[ticker].get(date)
            p1 = prices[ticker].get(all_dates[idx + 1])
            if p0 and p1 and p0 != 0:
                r = (p1 - p0) / p0
                rets.append(r if signal == "BUY" else -r)
        daily.append(np.mean(rets) if rets else 0.0)
    cum = (np.cumprod([1 + r for r in daily]) - 1) * 100
    return pd.Series(cum, index=all_dates)

LINES = [
    ("Baseline", "gpt-4.1_{s}", "#5B9BD5", "-",  1.8),
    ("D1→SELL",  "adv_gpt_D1_sell_{s}", "#E06C75", "--", 1.5),
    ("A3→SELL",  "adv_gpt_A3_sell_{s}", "#C44E52", ":",  1.5),
]

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, struct in zip(axes, STRUCTURES):
    for label, tmpl, color, ls, lw in LINES:
        df = load_experiment(tmpl.replace("{s}", struct))
        if df is None:
            continue
        pnl = cumulative_pnl(df, prices)
        ax.plot(range(len(pnl)), pnl.values, color=color, ls=ls, lw=lw, label=label)
    ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax.set_title(STRUCT_LABELS[struct], fontsize=11, fontweight="bold")
    ax.set_xlabel("Trading Day", fontsize=9)
    if ax == axes[0]:
        ax.set_ylabel("Cumulative Return (%)", fontsize=10)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

handles = [mpatches.Patch(color=c, label=l) for l, _, c, _, _ in LINES]
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
plt.suptitle("Cumulative PnL: Baseline vs. D1→SELL vs. A3→SELL",
             fontsize=13, fontweight="bold", y=1.08)
plt.tight_layout()
plt.savefig(FIG_DIR / "ret3_cumulative_pnl.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved ret3")

# ══════════════════════════════════════════════════════════════════════════════
# FIG ret4: Sharpe Ratio 비교
# ══════════════════════════════════════════════════════════════════════════════
print("ret4: Sharpe ratio bar...")

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for ax, struct in zip(axes, STRUCTURES):
    labels, sharpes, colors_ = [], [], []
    s_base = all_returns.get(("Baseline", struct), pd.Series()).dropna()
    base_sh = s_base.mean() / s_base.std() * np.sqrt(252) if len(s_base) > 1 and s_base.std() > 0 else 0
    labels.append("Base"); sharpes.append(base_sh); colors_.append("#5B9BD5")

    for atk in ATTACKS:
        vals = []
        for sig in ["BUY", "SELL"]:
            r = all_returns.get((f"{atk}→{sig}", struct), pd.Series()).dropna()
            if len(r) > 1 and r.std() > 0:
                vals.append(r.mean() / r.std() * np.sqrt(252))
        if vals:
            sh = np.mean(vals)
            labels.append(atk)
            sharpes.append(sh)
            colors_.append("#E06C75" if sh < base_sh else "#55A868")

    x = np.arange(len(labels))
    ax.bar(x, sharpes, color=colors_, edgecolor="white", lw=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(base_sh, color="#5B9BD5", ls="--", lw=1, alpha=0.5)
    ax.set_title(STRUCT_LABELS[struct], fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    if ax == axes[0]:
        ax.set_ylabel("Annualized Sharpe Ratio", fontsize=10)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    for xi, val in zip(x, sharpes):
        ax.text(xi, val + (0.03 if val >= 0 else -0.08), f"{val:.2f}",
                ha="center", va="bottom", fontsize=7)

plt.suptitle("Annualized Sharpe Ratio: Baseline vs. Attack Scenarios",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "ret4_sharpe_ratio.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved ret4")

# ══════════════════════════════════════════════════════════════════════════════
# FIG ret5: Mean–Std shift scatter (baseline → attack)
# ══════════════════════════════════════════════════════════════════════════════
print("ret5: mean–std shift scatter...")
ATTACK_MARKERS = {"D1": "o", "D2": "s", "A1": "^", "A2": "D", "A3": "*"}
SIGNAL_COLORS  = {"BUY": "#4CAF50", "SELL": "#E06C75"}

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, struct in zip(axes, STRUCTURES):
    s_base = all_returns.get(("Baseline", struct), pd.Series()).dropna()
    bm, bv = s_base.mean() * 100, s_base.std() * 100
    ax.scatter([bv], [bm], s=160, color="#5B9BD5", marker="P", zorder=5, label="Baseline")
    ax.annotate("Base", (bv, bm), fontsize=7, xytext=(4, 3), textcoords="offset points")

    for atk in ATTACKS:
        for sig in ["BUY", "SELL"]:
            r = all_returns.get((f"{atk}→{sig}", struct), pd.Series()).dropna()
            if len(r) == 0:
                continue
            m, v = r.mean() * 100, r.std() * 100
            ax.scatter([v], [m], s=80, color=SIGNAL_COLORS[sig],
                       marker=ATTACK_MARKERS[atk], alpha=0.85, zorder=4)
            ax.annotate(f"{atk}", (v, m), fontsize=6,
                        xytext=(2, 2), textcoords="offset points", alpha=0.75)

    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.axvline(bv, color="#5B9BD5", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Return Std (%)", fontsize=9)
    if ax == axes[0]:
        ax.set_ylabel("Mean Return (%)", fontsize=9)
    ax.set_title(STRUCT_LABELS[struct], fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)

legend_h = (
    [mpatches.Patch(color="#5B9BD5", label="Baseline")] +
    [plt.Line2D([0],[0], marker=ATTACK_MARKERS[a], color="w",
                markerfacecolor="gray", markersize=8, label=a) for a in ATTACKS] +
    [mpatches.Patch(color=SIGNAL_COLORS[s], label=f"→{s}") for s in ["BUY","SELL"]]
)
fig.legend(handles=legend_h, loc="upper center", ncol=8, fontsize=8, bbox_to_anchor=(0.5, 1.02))
plt.suptitle("Mean–Std Shift under Attacks (each point = one attack×direction)",
             fontsize=12, fontweight="bold", y=1.08)
plt.tight_layout()
plt.savefig(FIG_DIR / "ret5_mean_std_shift.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved ret5")

# ── 수치 요약 ───────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print(f"{'':20s} {'Mean%':>8} {'Std%':>8} {'Sharpe':>8} {'Var×10⁻⁴':>10}")
print("-" * 68)
for struct in STRUCTURES:
    print(f"\n  [{STRUCT_LABELS[struct]}]")
    s = all_returns.get(("Baseline", struct), pd.Series()).dropna()
    if len(s) > 1:
        sh = s.mean()/s.std()*np.sqrt(252)
        print(f"  {'Baseline':18s} {s.mean()*100:8.3f} {s.std()*100:8.3f} {sh:8.2f} {s.var()*1e4:10.3f}")
    for atk in ATTACKS:
        combined = []
        for sig in ["BUY","SELL"]:
            combined += all_returns.get((f"{atk}→{sig}", struct), pd.Series()).dropna().tolist()
        if combined:
            s2 = pd.Series(combined)
            sh = s2.mean()/s2.std()*np.sqrt(252) if s2.std() > 0 else 0
            print(f"  {atk:18s} {s2.mean()*100:8.3f} {s2.std()*100:8.3f} {sh:8.2f} {s2.var()*1e4:10.3f}")

print("\nDone. Saved: ret1~ret5 in figures/")
