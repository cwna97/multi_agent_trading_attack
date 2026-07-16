import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

x = np.arange(4)
arch_labels_wrap = ['Dec.\n(most robust)', 'Hyb.', 'Cent.', 'Lin.\n(most vulnerable)']

# ── Panel (a) data: conventional centrality ───────────────────────────────────
centrality_data = {
    'Betweenness': np.array([1.000, 0.133, 1.000, 0.667]),
    'PageRank':    np.array([0.542, 0.401, 0.542, 0.370]),
    'In-Degree':   np.array([1.000, 0.500, 1.000, 0.333]),
    'Closeness':   np.array([1.000, 0.750, 1.000, 0.750]),
}
colors_centrality = {
    'Betweenness': '#5B9BD5',
    'PageRank':    '#ED7D31',
    'In-Degree':   '#70AD47',
    'Closeness':   '#9B59B6',
}
markers_c = {'Betweenness': 'o', 'PageRank': 's', 'In-Degree': '^', 'Closeness': 'D'}

# ── Panel (b) data: structural baselines ─────────────────────────────────────
# Dec, Hyb, Cen, Lin  (APS ascending order)
# Edge count (normalized by max=12)
# Depth      (normalized by max=3)
# Max fan-in (normalized by max=3)
baseline_data = {
    'Edge Count':  np.array([3/12, 12/12, 3/12, 3/12]),    # [0.25, 1.00, 0.25, 0.25]
    'Depth':       np.array([1/3,  2/3,   1/3,  3/3]),     # [0.33, 0.67, 0.33, 1.00]
    'Max Fan-in':  np.array([3/3,  3/3,   3/3,  1/3]),     # [1.00, 1.00, 1.00, 0.33]
}
colors_baseline = {
    'Edge Count': '#17BECF',   # teal
    'Depth':      '#8C564B',   # brown
    'Max Fan-in': '#E377C2',   # pink
}
markers_b = {'Edge Count': 'P', 'Depth': 'X', 'Max Fan-in': 'v'}

# ── Panel (c) data: APS ───────────────────────────────────────────────────────
aps          = np.array([0.167, 0.222, 0.333, 0.600])
arch_colors  = ['#FFAAAA', '#F06060', '#C0202A', '#7B0000']

# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(26, 10),
                               gridspec_kw={'width_ratios': [1.2, 1.0]})

TICK_FS  = 28
TITLE_FS = 28
ANNOT_FS = 24
LABEL_FS = 20
LEGEND_FS = 18
VAL_FS   = 12

# ══ Panel (a): Conventional Graph Centrality Metrics ═════════════════════════
ideal = np.linspace(0, 1, 4)
ax1.plot(x, ideal, color='lightgray', linewidth=2.0, linestyle='--',
         label='Expected trend', zorder=1)

offsets_c = {
    'Betweenness': ( 0,    +0.06),
    'PageRank':    ( 0,    -0.09),
    'In-Degree':   (-0.15, +0.06),
    'Closeness':   (+0.15, -0.09),
}
for name, vals in centrality_data.items():
    ax1.plot(x, vals, marker=markers_c[name], markersize=10, linewidth=2.5,
             color=colors_centrality[name], label=name, zorder=4)

for name, vals in centrality_data.items():
    for i, v in enumerate(vals):
        ox, oy = offsets_c[name]
        ax1.text(i+ox, v+oy, f'{v:.2f}', ha='center', va='center',
                 fontsize=VAL_FS, color=colors_centrality[name])

# structural baselines — dashed lines
offsets_b = {
    'Edge Count':  (+0.15, -0.09),
    'Depth':       (-0.15, -0.09),
    'Max Fan-in':  ( 0,    +0.06),
}
for name, vals in baseline_data.items():
    ax1.plot(x, vals, marker=markers_b[name], markersize=9, linewidth=2.0,
             linestyle='--', color=colors_baseline[name], label=name, zorder=3)
for name, vals in baseline_data.items():
    for i, v in enumerate(vals):
        ox, oy = offsets_b[name]
        ax1.text(i+ox, v+oy, f'{v:.2f}', ha='center', va='center',
                 fontsize=VAL_FS, color=colors_baseline[name])

ax1.text(0.02, 0.97,
         '✗  Fail to capture\nvulnerability ordering',
         ha='left', va='top', fontsize=ANNOT_FS, color='crimson', fontweight='bold',
         transform=ax1.transAxes,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff0f0',
                   edgecolor='crimson', alpha=0.9))

ax1.set_xticks(x)
ax1.set_xticklabels(arch_labels_wrap, fontsize=TICK_FS)
ax1.set_ylabel('Score (normalized)', fontsize=LABEL_FS)
ax1.set_title('(a) Graph Centrality Metrics & Structural Baselines',
              fontsize=TITLE_FS, fontweight='bold', pad=16)
ax1.legend(fontsize=LEGEND_FS, loc='lower right', framealpha=0.92,
           ncol=2, columnspacing=0.8, handlelength=1.8)
ax1.set_ylim(-0.05, 1.45)
ax1.yaxis.grid(True, alpha=0.25); ax1.set_axisbelow(True)
ax1.tick_params(axis='y', labelsize=12)

# ══ Panel (c): Adversarial Signal Preservation Score (APS) ═══════════════════
bars = ax3.bar(x, aps, width=0.55, color=arch_colors, alpha=0.88,
               edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, aps):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.013,
             f'{val:.3f}', ha='center', va='bottom', fontsize=18,
             fontweight='bold')

for i in range(3):
    ax3.annotate('',
        xy=(x[i+1]-0.22, aps[i+1]+0.028),
        xytext=(x[i]+0.22, aps[i]+0.028),
        arrowprops=dict(arrowstyle='->', color='#8B0000', lw=2.2))

ax3.text(0.02, 0.97,
         '✓  Correctly captures\nvulnerability ordering',
         ha='left', va='top', fontsize=ANNOT_FS, color='#2e7d32', fontweight='bold',
         transform=ax3.transAxes,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0fff0',
                   edgecolor='#2e7d32', alpha=0.9))

ax3.set_xticks(x)
ax3.set_xticklabels(arch_labels_wrap, fontsize=TICK_FS)
ax3.set_ylabel('APS', fontsize=LABEL_FS)
ax3.set_title('(b) Adversarial Signal Preservation Score',
              fontsize=TITLE_FS, fontweight='bold', pad=16)
ax3.set_ylim(0, 1.0)
ax3.yaxis.grid(True, alpha=0.25); ax3.set_axisbelow(True)
ax3.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.savefig('figures/fig_centrality_vs_aps.png', dpi=180, bbox_inches='tight')
plt.savefig('figures/fig_centrality_vs_aps.pdf', bbox_inches='tight')
print("Saved figures/fig_centrality_vs_aps.png and .pdf")
