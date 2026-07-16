import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ANALYST_C  = '#4C72B0'
VOTE_C     = '#2CA02C'
REVIEW_C   = '#7B2D8B'
DECISION_C = '#C44E52'
EDGE_C     = '#333333'
FONT       = 'DejaVu Sans'

W, H = 3.2, 1.8   # uniform canvas — all 4 PDFs will be exactly this size

def new_fig():
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax

def node(ax, x, y, label, color, r=0.10, fs=8):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor='white',
                        linewidth=1.4, zorder=4))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs,
            color='white', fontweight='bold', fontfamily=FONT, zorder=5)

def arrow(ax, x0, y0, x1, y1, r=0.10, lw=1.2, color=EDGE_C):
    dx, dy = x1 - x0, y1 - y0
    dist = np.hypot(dx, dy)
    if dist == 0:
        return
    ux, uy = dx / dist, dy / dist
    ax.annotate('',
        xy=(x1 - ux*(r+0.015), y1 - uy*(r+0.015)),
        xytext=(x0 + ux*(r+0.01), y0 + uy*(r+0.01)),
        arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                        mutation_scale=10, connectionstyle='arc3,rad=0.0'),
        zorder=3)

def save(name):
    plt.savefig(f'figs/arch_{name}.pdf', transparent=True)
    plt.savefig(f'figs/arch_{name}.png', dpi=220, transparent=True)
    plt.close()
    print(f"Saved arch_{name}")


# ── LINEAR  M → S → N → Σ ────────────────────────────────────────────────────
fig, ax = new_fig()
pts = [(0.45, 0.90, 'M', ANALYST_C),
       (1.25, 0.90, 'S', ANALYST_C),
       (2.05, 0.90, 'N', ANALYST_C),
       (2.85, 0.90, 'Σ', DECISION_C)]
for x, y, l, c in pts:
    node(ax, x, y, l, c)
for i in range(len(pts) - 1):
    arrow(ax, pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
save('linear')


# ── CENTRALIZED  M,S,N → Σ ───────────────────────────────────────────────────
fig, ax = new_fig()
analysts = [(0.55, 1.50), (0.55, 0.90), (0.55, 0.30)]
coord    = (2.00, 0.90)
for (x, y), lbl in zip(analysts, ['M', 'S', 'N']):
    node(ax, x, y, lbl, ANALYST_C)
node(ax, *coord, 'Σ', DECISION_C)
for x, y in analysts:
    arrow(ax, x, y, coord[0], coord[1])
save('centralized')


# ── DECENTRALIZED  M,S,N → V ─────────────────────────────────────────────────
fig, ax = new_fig()
analysts = [(0.55, 1.50), (0.55, 0.90), (0.55, 0.30)]
vote     = (2.00, 0.90)
for (x, y), lbl in zip(analysts, ['M', 'S', 'N']):
    node(ax, x, y, lbl, ANALYST_C)
node(ax, *vote, 'V', VOTE_C)
for x, y in analysts:
    arrow(ax, x, y, vote[0], vote[1])
save('decentralized')


# ── HYBRID  M,S,N → M',S',N' → Σ ────────────────────────────────────────────
fig, ax = new_fig()
l1  = [(0.42, 1.50), (0.42, 0.90), (0.42, 0.30)]
l2  = [(1.68, 1.50), (1.68, 0.90), (1.68, 0.30)]
agg = (2.80, 0.90)
for (x, y), lbl in zip(l1, ['M', 'S', 'N']):
    node(ax, x, y, lbl, ANALYST_C)
for (x, y), lbl in zip(l2, ["M'", "S'", "N'"]):
    node(ax, x, y, lbl, REVIEW_C, fs=7)
node(ax, *agg, 'Σ', DECISION_C)
for x0, y0 in l1:
    for x1, y1 in l2:
        arrow(ax, x0, y0, x1, y1, lw=0.7)
for x, y in l2:
    arrow(ax, x, y, agg[0], agg[1], lw=1.2)
save('hybrid')
