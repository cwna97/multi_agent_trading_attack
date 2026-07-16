"""
Agent structure diagram – v2.
2×2 grid.  Each panel has a completely different topology
to make the structural difference visually obvious.

  Linear        : sequential chain  →  Synthesizer reads ①②③④ in order
  Centralized   : all reports fan into one Central Coordinator (star)
  Decentralized : each analyst casts a VOTE  →  majority tally
  MoA           : reports merge in two phases (diversity → consensus)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "figures" / "fig_agent_structures_v2.png"
OUT.parent.mkdir(exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
P = dict(
    analyst   = "#1565C0",
    tool      = "#2E7D32",
    linear    = "#4A148C",
    central   = "#BF360C",
    decent    = "#006064",
    moa       = "#880E4F",
    se        = "#37474F",
    vote_buy  = "#1B5E20",
    vote_sell = "#B71C1C",
    vote_hold = "#E65100",
    bg        = "#F8FAFC",
    grid_line = "#CFD8DC",
    white     = "#FFFFFF",
    dark      = "#1A1A2E",
)

ANALYSTS = ["Market\nAnalyst", "Social\nAnalyst",
            "News\nAnalyst",   "Fund.\nAnalyst"]

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.patch.set_facecolor(P["bg"])
fig.suptitle("Multi-Agent Graph Structures for Financial Decision-Making",
             fontsize=16, fontweight="bold", color=P["dark"], y=0.98)

for ax in axes.flat:
    ax.set_facecolor(P["bg"])
    ax.axis("off")

# ── shared drawing helpers ────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, fc, ec="#FFFFFF", lw=1.8, alpha=1.0, radius=0.05):
    patch = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=3
    )
    ax.add_patch(patch)

def txt(ax, cx, cy, s, size=9, color="white", bold=False, style="normal"):
    ax.text(cx, cy, s, ha="center", va="center",
            fontsize=size, fontweight="bold" if bold else "normal",
            fontstyle=style,
            color=color, zorder=5)

def circle(ax, cx, cy, r, fc, ec="white", lw=1.5):
    c = plt.Circle((cx, cy), r, facecolor=fc, edgecolor=ec,
                   linewidth=lw, zorder=3)
    ax.add_patch(c)

def arr(ax, x1, y1, x2, y2, color=P["se"], lw=1.5, headw=0.018,
        style="->", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=style,
                    color=color, lw=lw,
                    connectionstyle=f"arc3,rad={rad}",
                    shrinkA=5, shrinkB=5,
                ))

def section_title(ax, title, sub, color, xlim=(0, 1), y=0.97):
    ax.text(np.mean(xlim), y, title,
            ha="center", va="top",
            transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=color)
    ax.text(np.mean(xlim), y - 0.055, sub,
            ha="center", va="top",
            transform=ax.transAxes,
            fontsize=9, color="#546E7A", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# (a) LINEAR  –  strict sequential pipeline, synthesizer sees ①②③④ in order
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0][0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
section_title(ax, "(a) Linear", "Sequential accumulation: reports processed one-by-one",
              P["linear"])

# START
circle(ax, 5, 9.2, 0.28, P["se"])
txt(ax, 5, 9.2, "START", 7.5, bold=True)

# Analyst chain
ay = [8.2, 7.1, 6.0, 4.9]
nums = ["①", "②", "③", "④"]
arr(ax, 5, 8.92, 5, 8.52)
for i, (y, label, num) in enumerate(zip(ay, ANALYSTS, nums)):
    rbox(ax, 5, y, 2.8, 0.62, P["analyst"])
    txt(ax, 5, y, label.replace("\n", "  "), 9, bold=True)
    # numbered badge
    circle(ax, 3.4, y + 0.18, 0.18, P["linear"], ec="white", lw=1.5)
    txt(ax, 3.4, y + 0.18, num, 8, bold=True)
    # tool mini-box
    rbox(ax, 7.1, y, 0.8, 0.34, P["tool"], radius=0.03)
    txt(ax, 7.1, y, "Tools", 7)
    ax.annotate("", xy=(6.8, y + 0.07), xytext=(6.4, y + 0.07),
                arrowprops=dict(arrowstyle="->", color=P["tool"], lw=1.0,
                                shrinkA=3, shrinkB=3))
    ax.annotate("", xy=(6.4, y - 0.07), xytext=(6.8, y - 0.07),
                arrowprops=dict(arrowstyle="->", color=P["tool"], lw=1.0,
                                connectionstyle="arc3,rad=0.3",
                                shrinkA=3, shrinkB=3))
    if i < len(ay) - 1:
        arr(ax, 5, y - 0.31, 5, ay[i+1] + 0.31)

# Sequential-accumulation arrow with label
ax.annotate("", xy=(5, 3.72), xytext=(5, ay[-1] - 0.31),
            arrowprops=dict(arrowstyle="->, head_width=0.3, head_length=0.15",
                            color=P["linear"], lw=2.0, shrinkA=5, shrinkB=5))
ax.text(5.22, (ay[-1] - 0.31 + 3.72) / 2, "Reports ①②③④\nfed in order",
        ha="left", va="center", fontsize=7.5, color=P["linear"], style="italic")

# Linear Synthesizer
rbox(ax, 5, 3.2, 3.2, 0.80, P["linear"], lw=2.2)
txt(ax, 5, 3.38, "Linear Synthesizer", 10, bold=True)
txt(ax, 5, 3.05, "accumulates context sequentially", 7.5, style="italic")

# END
arr(ax, 5, 2.8, 5, 2.12, color=P["linear"], lw=2.0)
circle(ax, 5, 1.85, 0.28, P["se"])
txt(ax, 5, 1.85, "END", 7.5, bold=True)
ax.text(5, 1.42, "BUY / SELL / HOLD", ha="center", va="center",
        fontsize=8, color="#546E7A", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# (b) CENTRALIZED  –  star topology: all reports → one coordinator
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0][1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
section_title(ax, "(b) Centralized",
              "All reports fed to a single authority simultaneously", P["central"])

# START
circle(ax, 2.5, 9.2, 0.28, P["se"])
txt(ax, 2.5, 9.2, "START", 7.5, bold=True)
arr(ax, 2.5, 8.92, 2.5, 8.52)

# Analysts on LEFT column
ay = [8.2, 7.1, 6.0, 4.9]
ax.annotate("", xy=(2.5, ay[0] + 0.31), xytext=(2.5, 8.92),
            arrowprops=dict(arrowstyle="->", color=P["se"], lw=1.5, shrinkA=5, shrinkB=5))
for i, (y, label) in enumerate(zip(ay, ANALYSTS)):
    rbox(ax, 2.5, y, 2.6, 0.62, P["analyst"])
    txt(ax, 2.5, y, label.replace("\n", "  "), 9, bold=True)
    # tool mini
    rbox(ax, 0.75, y, 0.7, 0.32, P["tool"], radius=0.03)
    txt(ax, 0.75, y, "Tools", 7)
    ax.annotate("", xy=(1.1, y + 0.07), xytext=(1.2, y + 0.07),
                arrowprops=dict(arrowstyle="->", color=P["tool"], lw=1.0, shrinkA=3, shrinkB=3))
    ax.annotate("", xy=(1.2, y - 0.07), xytext=(1.1, y - 0.07),
                arrowprops=dict(arrowstyle="->", color=P["tool"], lw=1.0,
                                connectionstyle="arc3,rad=-0.3", shrinkA=3, shrinkB=3))
    if i < len(ay) - 1:
        arr(ax, 2.5, y - 0.31, 2.5, ay[i+1] + 0.31)

# Central Coordinator  – RIGHT, larger
cc_x, cc_y = 7.3, 6.55
rbox(ax, cc_x, cc_y, 3.2, 1.8, P["central"], lw=2.5)
txt(ax, cc_x, cc_y + 0.4,  "Central",      11, bold=True)
txt(ax, cc_x, cc_y,        "Coordinator",  11, bold=True)
txt(ax, cc_x, cc_y - 0.42, "holistic synthesis\nof all reports", 7.5, style="italic")

# Fan arrows: each analyst → CC (all 4 arrows)
report_labels = ["Market\nReport", "Social\nReport", "News\nReport", "Fund.\nReport"]
colors_fan = ["#42A5F5", "#66BB6A", "#FFA726", "#EC407A"]
for y, rl, fc in zip(ay, report_labels, colors_fan):
    ax.annotate("", xy=(cc_x - 1.6, cc_y + 0.55 - ay.index(y) * 0.35),
                xytext=(2.5 + 1.3, y),
                arrowprops=dict(arrowstyle="->, head_width=0.25, head_length=0.15",
                                color=fc, lw=1.6,
                                connectionstyle="arc3,rad=0.1",
                                shrinkA=5, shrinkB=5))

ax.text(5.05, 6.55, "All 4\nreports", ha="center", va="center",
        fontsize=7.5, color=P["central"], style="italic",
        bbox=dict(fc="#FBE9E7", ec=P["central"], lw=0.8, boxstyle="round,pad=0.3"))

# END
circle(ax, cc_x, 1.85, 0.28, P["se"])
arr(ax, cc_x, cc_y - 0.9, cc_x, 2.13, color=P["central"], lw=2.0)
txt(ax, cc_x, 1.85, "END", 7.5, bold=True)
ax.text(cc_x, 1.42, "BUY / SELL / HOLD", ha="center", va="center",
        fontsize=8, color="#546E7A", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# (c) DECENTRALIZED  –  each analyst → independent vote → tally
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1][0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
section_title(ax, "(c) Decentralized",
              "Each analyst casts an independent vote → majority wins", P["decent"])

circle(ax, 2.5, 9.2, 0.28, P["se"])
txt(ax, 2.5, 9.2, "START", 7.5, bold=True)
arr(ax, 2.5, 8.92, 2.5, 8.52)

# Example votes (simulated)
votes      = ["BUY",  "BUY",  "HOLD", "BUY"]
vote_colors= [P["vote_buy"], P["vote_buy"], P["vote_hold"], P["vote_buy"]]

for i, (y, label, vote, vc) in enumerate(zip(ay, ANALYSTS, votes, vote_colors)):
    rbox(ax, 2.5, y, 2.6, 0.62, P["analyst"])
    txt(ax, 2.5, y, label.replace("\n", "  "), 9, bold=True)
    # tool mini
    rbox(ax, 0.75, y, 0.7, 0.32, P["tool"], radius=0.03)
    txt(ax, 0.75, y, "Tools", 7)
    ax.annotate("", xy=(1.1, y + 0.07), xytext=(1.2, y + 0.07),
                arrowprops=dict(arrowstyle="->", color=P["tool"], lw=1.0, shrinkA=3, shrinkB=3))
    ax.annotate("", xy=(1.2, y - 0.07), xytext=(1.1, y - 0.07),
                arrowprops=dict(arrowstyle="->", color=P["tool"], lw=1.0,
                                connectionstyle="arc3,rad=-0.3", shrinkA=3, shrinkB=3))
    if i < len(ay) - 1:
        arr(ax, 2.5, y - 0.31, 2.5, ay[i+1] + 0.31)

    # Vote badge
    rbox(ax, 4.3, y, 0.72, 0.34, vc, ec="white", lw=1.5, radius=0.08)
    txt(ax, 4.3, y, vote, 8.5, bold=True)

    # Arrow analyst → vote badge
    arr(ax, 3.8, y, 3.94, y, color=vc, lw=1.3)

# Vote Aggregator
va_x, va_y = 7.5, 6.55
rbox(ax, va_x, va_y, 3.2, 1.9, P["decent"], lw=2.5)
txt(ax, va_x, va_y + 0.52, "Vote Aggregator", 10.5, bold=True)
# Tally
tally_y = va_y + 0.08
for label, count, color, dots in [
    ("BUY",  "3", P["vote_buy"],  "● ● ●"),
    ("HOLD", "1", P["vote_hold"], "●    "),
    ("SELL", "0", P["vote_sell"], "      "),
]:
    ax.text(va_x - 1.0, tally_y, f"{label}:", ha="left", va="center",
            fontsize=8.5, color="white", fontweight="bold")
    ax.text(va_x + 0.1, tally_y, dots, ha="left", va="center",
            fontsize=9, color=color)
    tally_y -= 0.38
ax.text(va_x, va_y - 0.72,
        "majority → BUY ✓", ha="center", va="center",
        fontsize=8.5, color="#A5D6A7", fontweight="bold", style="italic")

# Arrows: vote badge → aggregator
for y in ay:
    ax.annotate("", xy=(va_x - 1.6, va_y + 0.6 - ay.index(y) * 0.38),
                xytext=(4.65, y),
                arrowprops=dict(arrowstyle="->, head_width=0.2, head_length=0.12",
                                color=P["decent"], lw=1.4,
                                connectionstyle="arc3,rad=0.08",
                                shrinkA=4, shrinkB=4))

# END
circle(ax, va_x, 1.85, 0.28, P["se"])
arr(ax, va_x, va_y - 0.95, va_x, 2.13, color=P["decent"], lw=2.0)
txt(ax, va_x, 1.85, "END", 7.5, bold=True)
ax.text(va_x, 1.42, "BUY / SELL / HOLD", ha="center", va="center",
        fontsize=8, color="#546E7A", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# (d) MoA  –  two-phase: diversity detection → consensus synthesis
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1][1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
section_title(ax, "(d) MoA (Mixture of Agents)",
              "Diversity-aware aggregation: convergence + divergence reasoning", P["moa"])

circle(ax, 2.5, 9.2, 0.28, P["se"])
txt(ax, 2.5, 9.2, "START", 7.5, bold=True)
arr(ax, 2.5, 8.92, 2.5, 8.52)

for i, (y, label) in enumerate(zip(ay, ANALYSTS)):
    rbox(ax, 2.5, y, 2.6, 0.62, P["analyst"])
    txt(ax, 2.5, y, label.replace("\n", "  "), 9, bold=True)
    rbox(ax, 0.75, y, 0.7, 0.32, P["tool"], radius=0.03)
    txt(ax, 0.75, y, "Tools", 7)
    ax.annotate("", xy=(1.1, y + 0.07), xytext=(1.2, y + 0.07),
                arrowprops=dict(arrowstyle="->", color=P["tool"], lw=1.0, shrinkA=3, shrinkB=3))
    ax.annotate("", xy=(1.2, y - 0.07), xytext=(1.1, y - 0.07),
                arrowprops=dict(arrowstyle="->", color=P["tool"], lw=1.0,
                                connectionstyle="arc3,rad=-0.3", shrinkA=3, shrinkB=3))
    if i < len(ay) - 1:
        arr(ax, 2.5, y - 0.31, 2.5, ay[i+1] + 0.31)

# MoA Aggregator – two-phase box
ma_x, ma_y = 7.3, 6.55
rbox(ax, ma_x, ma_y, 3.4, 2.2, P["moa"], lw=2.5)
txt(ax, ma_x, ma_y + 0.75, "MoA Aggregator", 10.5, bold=True)

# Phase 1: Divergence
rbox(ax, ma_x, ma_y + 0.22, 2.8, 0.50, "#6A0080", ec="white", lw=1.2, radius=0.04)
txt(ax, ma_x, ma_y + 0.22, "Phase 1 · Divergence Detection", 7.5, bold=True)
# Phase 2: Convergence
rbox(ax, ma_x, ma_y - 0.42, 2.8, 0.50, "#4A0060", ec="white", lw=1.2, radius=0.04)
txt(ax, ma_x, ma_y - 0.42, "Phase 2 · Consensus Synthesis", 7.5, bold=True)
arr(ax, ma_x, ma_y - 0.03, ma_x, ma_y - 0.17, color="#CE93D8", lw=1.3)
txt(ax, ma_x, ma_y - 0.95, "diversity-aware final decision", 7.5, style="italic")

# Fan arrows: analyst → MoA (with diversity colors)
fan_colors = ["#CE93D8", "#F48FB1", "#80CBC4", "#FFE082"]
for y, fc in zip(ay, fan_colors):
    ax.annotate("", xy=(ma_x - 1.7, ma_y + 0.55 - ay.index(y) * 0.35),
                xytext=(2.5 + 1.3, y),
                arrowprops=dict(arrowstyle="->, head_width=0.2, head_length=0.12",
                                color=fc, lw=1.5,
                                connectionstyle="arc3,rad=0.1",
                                shrinkA=5, shrinkB=5))

# END
circle(ax, ma_x, 1.85, 0.28, P["se"])
arr(ax, ma_x, ma_y - 1.1, ma_x, 2.13, color=P["moa"], lw=2.0)
txt(ax, ma_x, 1.85, "END", 7.5, bold=True)
ax.text(ma_x, 1.42, "BUY / SELL / HOLD", ha="center", va="center",
        fontsize=8, color="#546E7A", style="italic")

# ── Legend (global, below figure) ────────────────────────────────────────────
legend_items = [
    mpatches.Patch(fc=P["analyst"],  label="Analyst Agent  (shared, all structures)"),
    mpatches.Patch(fc=P["tool"],     label="Tool Node"),
    mpatches.Patch(fc=P["linear"],   label="Linear Synthesizer"),
    mpatches.Patch(fc=P["central"],  label="Central Coordinator"),
    mpatches.Patch(fc=P["decent"],   label="Vote Aggregator"),
    mpatches.Patch(fc=P["moa"],      label="MoA Aggregator"),
    mpatches.Patch(fc=P["se"],       label="START / END"),
]
fig.legend(handles=legend_items, loc="lower center", ncol=7,
           fontsize=9, frameon=True, fancybox=True,
           framealpha=0.95, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}")
