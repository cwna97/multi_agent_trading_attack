import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

aps = np.array([0.167, 0.222, 0.333, 0.600])
arch_labels = ['Dec.\n(0.167)', 'Hyb.\n(0.222)', 'Cen.\n(0.333)', 'Lin.\n(0.600)']

q1_data = {
    'D1':  np.array([36.7, 99.3, 98.7, 97.7]),
    'D2':  np.array([37.0, 99.0, 96.7, 98.0]),
    'A1':  np.array([37.4, 100.0, 100.0, 100.0]),
    'A1o': np.array([ 6.6,  99.0,  88.9,  77.0]),
    'A2':  np.array([99.7, 100.0, 100.0, 100.0]),
    'A2s': np.array([14.1,  48.9,  69.2,  63.6]),
}
q3_data = {
    'D1':  np.array([10.4, 78.1, 99.1, 99.1]),
    'D2':  np.array([ 1.9, 60.9, 98.8, 99.0]),
    'A1':  np.array([ 0.3,  4.1, 38.4, 12.9]),
    'A1o': np.array([ 0.0,  5.6, 56.2, 26.2]),
    'A2':  np.array([ 0.6,  5.9, 60.0, 27.1]),
    'A2s': np.array([ 1.2,  4.4, 38.4, 11.2]),
}

# average across the two periods
avg_data = {atk: (q1_data[atk] + q3_data[atk]) / 2 for atk in q1_data}

attack_labels = {
    'D1':  'D1 (News Poison)',
    'D2':  'D2 (Indirect Inj.)',
    'A1':  r'$A1_{subtle}$ (1-Analyst)',
    'A1o': r'$A1_{overt}$ (1-Analyst)',
    'A2':  r'$A2_{overt}$ (2-Analyst)',
    'A2s': r'$A2_{subtle}$ (2-Analyst)',
}
markers   = {'D1': 'o', 'D2': 's', 'A1': '^', 'A1o': 'v', 'A2': 'D', 'A2s': 'P'}
colors    = {
    'D1':  '#2980B9', 'D2':  '#1ABC9C',
    'A1':  '#E67E22', 'A1o': '#E74C3C',
    'A2':  '#9B59B6', 'A2s': '#F39C12',
}
marker_size = 130
atk_order = ['D1', 'D2', 'A1', 'A1o', 'A2s', 'A2']

fig, ax = plt.subplots(figsize=(12, 5))

# ── scatter points ─────────────────────────────────────────────────────────────
for atk in atk_order:
    ax.scatter(aps, avg_data[atk], marker=markers[atk], s=marker_size,
               color=colors[atk], edgecolors='white', linewidths=0.9,
               zorder=5, label=attack_labels[atk])

# ── APS trend line ─────────────────────────────────────────────────────────────
all_x = np.tile(aps, len(atk_order))
all_y = np.concatenate([avg_data[atk] for atk in atk_order])
slope, intercept, r, p, _ = stats.linregress(all_x, all_y)
x_line = np.linspace(0.13, 0.65, 100)
ax.plot(x_line, slope * x_line + intercept, color='gray',
        linewidth=2.0, linestyle='--', alpha=0.6, zorder=3,
        label=f'APS trend  ($r={r:.2f}$)')

# ── axes ───────────────────────────────────────────────────────────────────────
ax.set_xticks(aps)
ax.set_xticklabels(arch_labels, fontsize=14)
ax.set_xlabel('APS  (Adversarial Signal Preservation Score)', fontsize=14)
ax.set_ylabel('Mean ASR (%)  [avg. Q1 & Q3]', fontsize=14)
ax.set_xlim(0.10, 0.68)
ax.set_ylim(0, 100)
ax.yaxis.set_major_locator(plt.MultipleLocator(25))
ax.yaxis.grid(True, alpha=0.25)
ax.set_axisbelow(True)
ax.tick_params(axis='y', labelsize=12)

ax.legend(fontsize=11, loc='upper left', framealpha=0.93)


plt.tight_layout()
plt.savefig('figures/fig02b_centrality_asr_scatter_avg.png', dpi=180, bbox_inches='tight')
plt.savefig('figures/fig02b_centrality_asr_scatter_avg.pdf', bbox_inches='tight')
print("Saved figures/fig02b_centrality_asr_scatter_avg.png and .pdf")
