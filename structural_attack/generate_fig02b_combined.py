import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# APS values per architecture
aps = np.array([0.167, 0.222, 0.333, 0.600])
arch_labels = ['Dec.\n(0.167)', 'Hyb.\n(0.222)', 'Cen.\n(0.333)', 'Lin.\n(0.600)']

# Q1 2026 (BUY-targeted, bearish)
q1_data = {
    'D1':  np.array([36.7, 99.3, 98.7, 97.7]),
    'D2':  np.array([37.0, 99.0, 96.7, 98.0]),
    'A1':  np.array([37.4, 100.0, 100.0, 100.0]),
    'A1o': np.array([ 6.6,  99.0,  88.9,  77.0]),
    'A2':  np.array([99.7, 100.0, 100.0, 100.0]),
    'A2s': np.array([14.1,  48.9,  69.2,  63.6]),
}

# Q3 2025 (SELL-targeted, bullish)
q3_data = {
    'D1':  np.array([10.4, 78.1, 99.1, 99.1]),
    'D2':  np.array([ 1.9, 60.9, 98.8, 99.0]),
    'A1':  np.array([ 0.3,  4.1, 38.4, 12.9]),
    'A1o': np.array([ 0.0,  5.6, 56.2, 26.2]),
    'A2':  np.array([ 0.6,  5.9, 60.0, 27.1]),
    'A2s': np.array([ 1.2,  4.4, 38.4, 11.2]),
}

attack_labels = {
    'D1':  'D1 (News Poison)',
    'D2':  'D2 (Indirect Inj.)',
    'A1':  'A1 (1-Analyst, subtle)',
    'A1o': 'A1o (1-Analyst, overt)',
    'A2':  'A2 (2-Analyst, overt)',
    'A2s': 'A2s (2-Analyst, subtle)',
}

markers  = {'D1': 'o', 'D2': 's', 'A1': '^', 'A1o': 'v', 'A2': 'D', 'A2s': 'P'}
marker_size = 130

# Period colors
color_q1 = '#C0392B'
color_q3 = '#2980B9'

atk_order = ['D1', 'D2', 'A1', 'A1o', 'A2', 'A2s']

fig, ax = plt.subplots(figsize=(10, 8))

# ── scatter points ─────────────────────────────────────────────────────────────
for atk in atk_order:
    ax.scatter(aps, q1_data[atk], marker=markers[atk], s=marker_size,
               color=color_q1, edgecolors='white', linewidths=0.9, zorder=5)
    ax.scatter(aps, q3_data[atk], marker=markers[atk], s=marker_size,
               color=color_q3, edgecolors='white', linewidths=0.9, zorder=5)

# ── OLS trend lines per period ─────────────────────────────────────────────────
all_q1_x, all_q1_y, all_q3_x, all_q3_y = [], [], [], []
for atk in atk_order:
    all_q1_x.extend(aps.tolist()); all_q1_y.extend(q1_data[atk].tolist())
    all_q3_x.extend(aps.tolist()); all_q3_y.extend(q3_data[atk].tolist())

x_line = np.linspace(0.13, 0.65, 100)
for xs, ys, col in [(all_q1_x, all_q1_y, color_q1), (all_q3_x, all_q3_y, color_q3)]:
    slope, intercept, *_ = stats.linregress(xs, ys)
    ax.plot(x_line, slope * x_line + intercept, color=col,
            linewidth=2.0, linestyle='--', alpha=0.55, zorder=3)

# ── Legend: attack shapes ──────────────────────────────────────────────────────
shape_handles = [
    ax.scatter([], [], marker=markers[atk], s=marker_size,
               color='#555555', edgecolors='white', linewidths=0.9,
               label=attack_labels[atk])
    for atk in atk_order
]
period_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color_q1,
           markersize=12, label='Q1 2026 (BUY-targeted, bearish)', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color_q3,
           markersize=12, label='Q3 2025 (SELL-targeted, bullish)', linestyle='None'),
]

leg1 = ax.legend(handles=shape_handles, title='Attack type', fontsize=13,
                 title_fontsize=13, loc='upper left', framealpha=0.93)
ax.add_artist(leg1)
ax.legend(handles=period_handles, title='Period', fontsize=13,
          title_fontsize=13, loc='lower right', framealpha=0.93)

# ── Axes ───────────────────────────────────────────────────────────────────────
ax.set_xticks(aps)
ax.set_xticklabels(arch_labels, fontsize=16)
ax.set_xlabel('APS  (Adversarial Signal Preservation Score)', fontsize=16)
ax.set_ylabel('ASR (%)', fontsize=16)
ax.set_title('Centrality–Vulnerability Correlation\n(attack type = shape,  period = color)',
             fontsize=17, fontweight='bold', pad=14)
ax.set_xlim(0.10, 0.68)
ax.set_ylim(-8, 118)
ax.yaxis.grid(True, alpha=0.25)
ax.set_axisbelow(True)
ax.tick_params(axis='y', labelsize=14)

# APS direction arrow
ax.annotate('', xy=(0.62, 111), xytext=(0.12, 111),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2.0))
ax.text(0.37, 114.5, 'APS  →  more vulnerable', ha='center', va='center',
        fontsize=13, color='gray')

plt.tight_layout()
plt.savefig('figures/fig02b_centrality_asr_scatter_combined.png', dpi=180, bbox_inches='tight')
plt.savefig('figures/fig02b_centrality_asr_scatter_combined.pdf', bbox_inches='tight')
print("Saved figures/fig02b_centrality_asr_scatter_combined.png and .pdf")
