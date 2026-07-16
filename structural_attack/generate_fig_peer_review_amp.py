import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

aps  = np.array([0.167, 0.222, 0.333, 0.600])
x    = np.arange(4)
arch_labels = ['Dec.\n(0.167)', 'Hyb.\n(0.222)', 'Cen.\n(0.333)', 'Lin.\n(0.600)']

# Q1 2026 agent-level attacks
attacks = {
    'A1$_{subtle}$':  np.array([37.4, 100.0, 100.0, 100.0]),
    'A1$_{overt}$':   np.array([ 6.6,  99.0,  88.9,  77.0]),
    'A2$_{subtle}$':  np.array([14.1,  48.9,  69.2,  63.6]),
    'A2$_{overt}$':   np.array([99.7, 100.0, 100.0, 100.0]),
}
colors = {
    'A1$_{subtle}$': '#E67E22',
    'A1$_{overt}$':  '#C0392B',
    'A2$_{subtle}$': '#F39C12',
    'A2$_{overt}$':  '#9B59B6',
}
markers = {
    'A1$_{subtle}$': '^', 'A1$_{overt}$': 'v',
    'A2$_{subtle}$': 'P', 'A2$_{overt}$': 'D',
}

# APS-proportional expected value (linear interp Dec→Lin per attack)
def expected_hyb(vals):
    t = (aps[1] - aps[0]) / (aps[3] - aps[0])
    return vals[0] + t * (vals[3] - vals[0])

fig, ax2 = plt.subplots(figsize=(7, 5))

# ══ Amplification excess bars ═════════════════════════════════════════════════
atk_names  = list(attacks.keys())
excess     = np.array([attacks[n][1] - expected_hyb(attacks[n]) for n in atk_names])
bar_colors = [colors[n] for n in atk_names]

bars = ax2.barh(np.arange(len(atk_names)), excess, 0.5,
                color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, excess):
    ax2.text(val + 0.8, bar.get_y() + bar.get_height()/2,
             f'{val:+.0f}%', va='center', fontsize=13, fontweight='bold',
             color=bar.get_facecolor())

ax2.axvline(0, color='gray', linewidth=1.3, linestyle='-', alpha=0.6)
ax2.set_yticks(np.arange(len(atk_names)))
ax2.set_yticklabels(atk_names, fontsize=14)
ax2.set_xlabel('Amplification excess at Hybrid\n(Actual ASR − APS-expected ASR)', fontsize=13)
ax2.set_title('Hybrid Amplification Excess\n(Q1 2026, agent-level attacks)',
              fontsize=14, fontweight='bold', pad=12)
ax2.set_xlim(0, 105)
ax2.xaxis.grid(True, alpha=0.25); ax2.set_axisbelow(True)
ax2.tick_params(axis='x', labelsize=12)

plt.tight_layout()
plt.savefig('figures/fig_peer_review_amplification.png', dpi=180, bbox_inches='tight')
plt.savefig('figures/fig_peer_review_amplification.pdf', bbox_inches='tight')
print("Saved figures/fig_peer_review_amplification.png and .pdf")
