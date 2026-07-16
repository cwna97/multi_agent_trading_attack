import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# order: D1, D2, A1_subtle, A1_overt, A2_subtle, A2_overt
attacks   = ['D1', 'D2', 'A1', 'A1o', 'A2s', 'A2']
atk_labels = ['D1', 'D2',
               r'$A1_{subtle}$', r'$A1_{overt}$',
               r'$A2_{subtle}$', r'$A2_{overt}$']

q1_mean = np.array([71.5, 68.5, 49.0, 67.9, 48.9, 68.2])
q3_mean = np.array([71.7, 65.2, 13.9, 22.0, 13.8, 23.4])
ratio    = q1_mean / q3_mean

color_q1 = '#C0392B'
color_q3 = '#2980B9'

x = np.arange(len(attacks))
w = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

# ── grouped bars ──────────────────────────────────────────────────────────────
b1 = ax.bar(x - w/2, q1_mean, w, color=color_q1, alpha=0.88,
            edgecolor='white', linewidth=1.2, label='Q1 2026 — BUY-targeted')
b2 = ax.bar(x + w/2, q3_mean, w, color=color_q3, alpha=0.88,
            edgecolor='white', linewidth=1.2, label='Q3 2025 — SELL-targeted')

for bar, val in zip(b1, q1_mean):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.2, f'{val:.0f}',
            ha='center', va='bottom', fontsize=12, color=color_q1, fontweight='bold')
for bar, val in zip(b2, q3_mean):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.2, f'{val:.0f}',
            ha='center', va='bottom', fontsize=12, color=color_q3, fontweight='bold')

# ── difference arrows for agent-level only ────────────────────────────────────
for i in range(2, 6):
    y_top = q1_mean[i]
    y_bot = q3_mean[i]
    xc    = x[i]
    # bracket line
    ax.annotate('', xy=(xc + 0.03, y_bot + 1), xytext=(xc + 0.03, y_top - 1),
                arrowprops=dict(arrowstyle='<->', color='#555555',
                                lw=1.8, mutation_scale=12))
    ax.text(xc + 0.17, (y_top + y_bot) / 2,
            f'Δ{ratio[i]:.1f}×', ha='left', va='center',
            fontsize=11.5, fontweight='bold', color='#333333')

# ── shading + region labels ───────────────────────────────────────────────────
ax.axvspan(-0.5, 1.5, color='#e8f4fd', alpha=0.40, zorder=0)
ax.axvspan(1.5,  5.5, color='#fff3e0', alpha=0.40, zorder=0)
ax.axvline(1.5, color='gray', lw=1.3, linestyle='--', alpha=0.5)

# Data-level / Agent-level labels at top of shaded regions
ax.text(0.5,  104, 'Data-level',  ha='center', fontsize=13,
        color='#1a6090', fontweight='bold')
ax.text(3.5,  104, 'Agent-level', ha='center', fontsize=13,
        color='#a04000', fontweight='bold')

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(atk_labels, fontsize=14)
ax.set_ylabel('Mean ASR (%) across architectures', fontsize=13)
ax.set_ylim(0, 112)
ax.legend(fontsize=12, loc='upper right', framealpha=0.93,
          bbox_to_anchor=(1.0, 0.88))
ax.yaxis.grid(True, alpha=0.25)
ax.set_axisbelow(True)
ax.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.savefig('figures/fig_llm_bias_asymmetry.png', dpi=180, bbox_inches='tight')
plt.savefig('figures/fig_llm_bias_asymmetry.pdf', bbox_inches='tight')
print("Saved figures/fig_llm_bias_asymmetry.png and .pdf")
