"""
Visualize the 4 multi-agent graph structures used in the experiment.
- Linear / Centralized / Decentralized / MoA
Each structure shares the same 4-analyst first layer;
the architectural difference is only in the final decision node.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────────────
C = {
    "bg":         "#F7F9FC",
    "start_end":  "#455A64",   # dark blue-gray
    "analyst":    "#1565C0",   # deep blue
    "tool":       "#43A047",   # green
    "linear":     "#6A1B9A",   # purple  → Linear Synthesizer
    "centralized":"#E65100",   # deep orange → Central Coordinator
    "decent":     "#00695C",   # teal → Vote Aggregator
    "moa":        "#AD1457",   # pink → MoA Aggregator
    "edge":       "#37474F",
    "text_light": "#FFFFFF",
    "text_dark":  "#212121",
    "tool_edge":  "#2E7D32",
}

ANALYSTS = ["Market\nAnalyst", "Social\nAnalyst", "News\nAnalyst", "Fund.\nAnalyst"]

STRUCTURES = [
    {
        "title":       "(a) Linear",
        "dec_label":   "Linear\nSynthesizer",
        "dec_sub":     "Sequential\naccumulation",
        "dec_color":   C["linear"],
        "x_center":    2.0,
    },
    {
        "title":       "(b) Centralized",
        "dec_label":   "Central\nCoordinator",
        "dec_sub":     "Holistic\nsynthesis",
        "dec_color":   C["centralized"],
        "x_center":    6.5,
    },
    {
        "title":       "(c) Decentralized",
        "dec_label":   "Vote\nAggregator",
        "dec_sub":     "Majority\nvoting",
        "dec_color":   C["decent"],
        "x_center":    11.0,
    },
    {
        "title":       "(d) MoA",
        "dec_label":   "MoA\nAggregator",
        "dec_sub":     "Diversity-aware\naggregation",
        "dec_color":   C["moa"],
        "x_center":    15.5,
    },
]

BOX_W   = 2.8    # analyst / decision box width
BOX_H   = 0.72   # analyst box height
DEC_H   = 0.88   # decision node height
SE_R    = 0.30   # START/END circle radius
GAP_Y   = 1.15   # vertical gap between analyst rows
TOOL_W  = 0.9
TOOL_H  = 0.42

# vertical anchor points (bottom of each element)
Y_START = 12.4
Y_A     = [10.8, 9.4, 8.0, 6.6]   # Market, Social, News, Fundamentals
Y_DEC   = 5.0
Y_END   = 3.7


def draw_rounded_box(ax, cx, cy, w, h, fc, ec, label, sub=None,
                     fontsize=8.5, lw=1.5, alpha=1.0):
    x = cx - w / 2
    y = cy - h / 2
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.08",
                         facecolor=fc, edgecolor=ec,
                         linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)
    if sub:
        ax.text(cx, cy + 0.10, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color=C["text_light"], zorder=4)
        ax.text(cx, cy - 0.22, sub, ha="center", va="center",
                fontsize=6.5, color=C["text_light"], alpha=0.88, zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color=C["text_light"], zorder=4)


def draw_circle(ax, cx, cy, r, fc, ec, label, fontsize=8.0):
    circ = plt.Circle((cx, cy), r, facecolor=fc, edgecolor=ec,
                       linewidth=1.8, zorder=3)
    ax.add_patch(circ)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=C["text_light"], zorder=4)


def arrow(ax, x1, y1, x2, y2, color=C["edge"], lw=1.4,
          style="->", shrink=4):
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style,
                                color=color,
                                lw=lw,
                                connectionstyle="arc3,rad=0.0",
                                shrinkA=shrink, shrinkB=shrink),
                zorder=2)


def draw_tool_loop(ax, cx, cy_analyst, analyst_h):
    """Small 'tool call' loop on the right side of an analyst box."""
    rx  = cx + BOX_W / 2 + 0.05   # right edge of analyst
    ty  = cy_analyst                 # tool box center y
    tw, th = TOOL_W, TOOL_H

    # tool box
    box = FancyBboxPatch((rx, ty - th / 2), tw, th,
                         boxstyle="round,pad=0.06",
                         facecolor=C["tool"], edgecolor=C["tool_edge"],
                         linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(rx + tw / 2, ty, "Tools", ha="center", va="center",
            fontsize=6.5, fontweight="bold", color="white", zorder=4)

    # analyst → tool (down-right)
    ax.annotate("",
                xy=(rx, ty),
                xytext=(cx + BOX_W / 2, cy_analyst + 0.10),
                arrowprops=dict(arrowstyle="->", color=C["tool_edge"],
                                lw=1.0, shrinkA=3, shrinkB=3),
                zorder=2)
    # tool → analyst (up-left)
    ax.annotate("",
                xy=(cx + BOX_W / 2, cy_analyst - 0.10),
                xytext=(rx, ty),
                arrowprops=dict(arrowstyle="->", color=C["tool_edge"],
                                lw=1.0, connectionstyle="arc3,rad=0.3",
                                shrinkA=3, shrinkB=3),
                zorder=2)


def draw_structure(ax, s):
    cx = s["x_center"]

    # Title
    ax.text(cx, 13.6, s["title"], ha="center", va="center",
            fontsize=11, fontweight="bold", color=C["text_dark"])

    # START
    draw_circle(ax, cx, Y_START, SE_R, C["start_end"], C["start_end"],
                "START", fontsize=7.5)

    # Analysts
    a_colors = [C["analyst"]] * 4
    for i, (label, yc) in enumerate(zip(ANALYSTS, Y_A)):
        draw_rounded_box(ax, cx, yc, BOX_W, BOX_H,
                         a_colors[i], "#0D47A1", label, fontsize=8.0)
        draw_tool_loop(ax, cx, yc, BOX_H)

    # Edges: START → Market, analysts chain
    arrow(ax, cx, Y_START - SE_R, cx, Y_A[0] + BOX_H / 2)
    for i in range(len(Y_A) - 1):
        arrow(ax, cx, Y_A[i] - BOX_H / 2, cx, Y_A[i + 1] + BOX_H / 2)

    # Decision node
    draw_rounded_box(ax, cx, Y_DEC, BOX_W, DEC_H,
                     s["dec_color"], s["dec_color"],
                     s["dec_label"], sub=s["dec_sub"],
                     fontsize=9.0, lw=2.0)

    # Fundamentals → Decision
    arrow(ax, cx, Y_A[-1] - BOX_H / 2, cx, Y_DEC + DEC_H / 2, lw=1.6)

    # END
    draw_circle(ax, cx, Y_END, SE_R, C["start_end"], C["start_end"],
                "END", fontsize=7.5)

    # Decision → END
    arrow(ax, cx, Y_DEC - DEC_H / 2, cx, Y_END + SE_R, lw=1.6)

    # BUY/SELL/HOLD label
    ax.text(cx + 0.5, Y_END - 0.6, "BUY / SELL / HOLD",
            ha="center", va="center", fontsize=7.5,
            color=C["text_dark"], style="italic")


def draw_shared_badge(ax):
    """Annotation indicating analysts are identical across all structures."""
    ax.text(8.75, 9.85, "← Identical across all structures →",
            ha="center", va="center", fontsize=8.0,
            color="#546E7A", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD",
                      ec="#90CAF9", lw=1.0, alpha=0.85))

    brace_y = [Y_A[0] + BOX_H / 2 + 0.1, Y_A[-1] - BOX_H / 2 - 0.1]
    for x in [0.2, 17.3]:
        ax.annotate("", xy=(8.75, brace_y[0]),
                    xytext=(x, brace_y[0]),
                    arrowprops=dict(arrowstyle="-", color="#90CAF9",
                                   lw=1.0, linestyle="dashed"))
        ax.annotate("", xy=(8.75, brace_y[1]),
                    xytext=(x, brace_y[1]),
                    arrowprops=dict(arrowstyle="-", color="#90CAF9",
                                   lw=1.0, linestyle="dashed"))


def make_legend(ax):
    handles = [
        mpatches.Patch(fc=C["analyst"],    ec="#0D47A1", label="Analyst Agent  (shared LLM)"),
        mpatches.Patch(fc=C["tool"],       ec=C["tool_edge"], label="Tool Node  (data retrieval)"),
        mpatches.Patch(fc=C["linear"],     label="Linear Synthesizer  (sequential accumulation)"),
        mpatches.Patch(fc=C["centralized"],label="Central Coordinator  (holistic synthesis)"),
        mpatches.Patch(fc=C["decent"],     label="Vote Aggregator  (majority voting)"),
        mpatches.Patch(fc=C["moa"],        label="MoA Aggregator  (diversity-aware)"),
        mpatches.Patch(fc=C["start_end"],  label="START / END"),
    ]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.04),
              ncol=4, fontsize=8.5, frameon=True,
              fancybox=True, framealpha=0.95)


# ── Main ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 11))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(-0.4, 18.0)
ax.set_ylim(2.5, 14.4)
ax.axis("off")

fig.suptitle("Multi-Agent Graph Structures for Financial Decision-Making",
             fontsize=14, fontweight="bold", y=0.98, color=C["text_dark"])

# Draw separator lines between structures
for x in [4.25, 8.75, 13.25]:
    ax.axvline(x, color="#CFD8DC", lw=1.0, linestyle="--", zorder=0)

for s in STRUCTURES:
    draw_structure(ax, s)

draw_shared_badge(ax)
make_legend(ax)

plt.tight_layout()
out = OUT_DIR / "fig_agent_structures.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
