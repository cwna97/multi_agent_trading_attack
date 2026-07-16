"""
Combined Finance Figure: Q1 2026 vs Q3 2025
Left : APS vs FWVS scatter (both periods, with trend lines)
Right: ΔCR grouped bar (both periods, annotated)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path

OUT = Path(__file__).parent / "figures" / "fig_finance_combined.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
STRUCTS  = ["Decentralized", "Hybrid", "Centralized", "Linear"]
APS      = [0.167, 0.222, 0.333, 0.600]
COLORS   = ["#4C72B0", "#7B2D8B", "#DD8452", "#C44E52"]

Q1 = dict(
    FWVS = [1.5078, 1.8187, 2.0754, 1.8181],
    DCR  = [-13.66, -2.78,  -2.88,  -5.98],
    label= "Q1 2026 (Bearish · BUY attacks)",
)
Q3 = dict(
    FWVS = [1.1652, 1.5167, 1.6516, 1.8439],
    DCR  = [+1.18,  -7.44,  -3.28,  -5.86],
    label= "Q3 2025 (Bullish · SELL attacks)",
)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor("white")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL: APS vs FWVS scatter
# ══════════════════════════════════════════════════════════════════════════════
aps = np.array(APS)

for period, marker, ls, zorder_base in [(Q1, "s", "--", 3), (Q3, "o", "-", 4)]:
    fwvs = np.array(period["FWVS"])

    # Trend line
    slope, intercept, r, p, _ = stats.linregress(aps, fwvs)
    x_line = np.linspace(0.13, 0.65, 100)
    ax1.plot(x_line, slope * x_line + intercept,
             color="gray" if period is Q1 else "#2a2a2a",
             lw=1.8, ls=ls, alpha=0.6, zorder=2)

    # Points
    for i, (s, c, a, f) in enumerate(zip(STRUCTS, COLORS, aps, fwvs)):
        ax1.scatter(a, f, color=c, marker=marker, s=160,
                    edgecolors="white", linewidths=1.2, zorder=zorder_base + i * 0.1,
                    alpha=0.95)

    # Period label at end of trend line
    x_end = 0.64
    y_end = slope * x_end + intercept
    rho_s, _ = stats.spearmanr(aps, fwvs)
    ax1.annotate(f"{period['label']}\n$\\rho_s={rho_s:.3f}$",
                 xy=(x_end, y_end),
                 xytext=(x_end - 0.04, y_end + (0.06 if period is Q3 else -0.12)),
                 fontsize=8.5, color="gray" if period is Q1 else "#2a2a2a",
                 ha="right",
                 arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

# Architecture labels (Q3 points only, to avoid clutter)
offsets = {"Decentralized": (-0.035, +0.04),
           "Hybrid":        (+0.010, +0.04),
           "Centralized":   (+0.010, +0.04),
           "Linear":        (+0.010, -0.07)}
for s, c, a, f in zip(STRUCTS, COLORS, APS, Q3["FWVS"]):
    dx, dy = offsets[s]
    ax1.text(a + dx, f + dy, s, color=c, fontsize=8.5, fontweight="bold", ha="center")

ax1.set_xlabel("Adversarial Propagation Score (APS)", fontsize=11)
ax1.set_ylabel("Financially Weighted Vulnerability Score\n(FWVS ×10⁻²)", fontsize=11)
ax1.set_title("(a) APS vs. FWVS", fontsize=13, fontweight="bold", pad=10)
ax1.set_xlim(0.10, 0.70)
ax1.set_ylim(0.90, 2.35)
ax1.yaxis.grid(True, alpha=0.25); ax1.set_axisbelow(True)

# Legend: shape = period
leg_handles = [
    Line2D([0], [0], marker="o", color="gray", lw=1.5, ls="-",
           markerfacecolor="gray", markersize=7, label="Q3 2025 (Bullish)"),
    Line2D([0], [0], marker="s", color="gray", lw=1.5, ls="--",
           markerfacecolor="gray", markersize=7, label="Q1 2026 (Bearish)"),
]
ax1.legend(handles=leg_handles, fontsize=8.5, loc="upper left",
           framealpha=0.85, edgecolor="#cccccc")

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL: ΔCR grouped bar
# ══════════════════════════════════════════════════════════════════════════════
x      = np.arange(len(STRUCTS))
width  = 0.35

bars_q1 = ax2.bar(x - width/2, Q1["DCR"], width,
                  color=COLORS, alpha=0.55, edgecolor="white",
                  hatch="///", label="Q1 2026")
bars_q3 = ax2.bar(x + width/2, Q3["DCR"], width,
                  color=COLORS, alpha=0.92, edgecolor="white",
                  label="Q3 2025")

ax2.axhline(0, color="black", lw=0.8)

# Value labels
for bar in list(bars_q1) + list(bars_q3):
    v = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2,
             v + (0.3 if v >= 0 else -0.8),
             f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top",
             fontsize=7.5, color="#333333")

# Annotations
# CR paradox arrow: Decentralized Q1
ax2.annotate("CR Paradox\n(low ASR, high loss)",
             xy=(x[0] - width/2, Q1["DCR"][0]),
             xytext=(x[0] - width/2 - 0.1, Q1["DCR"][0] - 3.5),
             fontsize=7.8, color="#4C72B0", ha="center",
             arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.0))

# Peer-review anomaly: Hybrid Q3
ax2.annotate("Peer-Review\nAmplification",
             xy=(x[1] + width/2, Q3["DCR"][1]),
             xytext=(x[1] + width/2 + 0.55, Q3["DCR"][1] - 2.0),
             fontsize=7.8, color="#7B2D8B", ha="center",
             arrowprops=dict(arrowstyle="->", color="#7B2D8B", lw=1.0))

ax2.set_xticks(x)
ax2.set_xticklabels(STRUCTS, fontsize=10)
ax2.set_ylabel("ΔCumulative Return (%)", fontsize=11)
ax2.set_title("(b) Cumulative Return Degradation", fontsize=13, fontweight="bold", pad=10)
ax2.yaxis.grid(True, alpha=0.25); ax2.set_axisbelow(True)
ax2.set_ylim(-20, 6)

# Legend: hatch = Q1, solid = Q3
leg2 = [
    mpatches.Patch(facecolor="gray", alpha=0.55, hatch="///",
                   edgecolor="gray", label="Q1 2026 (Bearish · BUY)"),
    mpatches.Patch(facecolor="gray", alpha=0.92,
                   edgecolor="gray", label="Q3 2025 (Bullish · SELL)"),
]
ax2.legend(handles=leg2, fontsize=8.5, loc="lower right",
           framealpha=0.85, edgecolor="#cccccc")

# APS annotation on x-axis
for i, (s, a) in enumerate(zip(STRUCTS, APS)):
    ax2.text(i, -19.5, f"APS={a}", ha="center", fontsize=7.5,
             color="#888888", style="italic")

plt.suptitle(
    "Financial Impact of Adversarial Attacks Across Market Regimes",
    fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.close()
print(f"Saved → {OUT}")
