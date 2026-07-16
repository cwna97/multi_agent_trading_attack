import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D
import pandas as pd

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

df = pd.read_csv("/tmp/fwvs_per_attack.csv")

aps  = np.array([0.167, 0.222, 0.333, 0.600])
arch_map = {"Decentralized": 0.167, "Hybrid": 0.222, "Centralized": 0.333, "Linear": 0.600}
arch_labels = ['Dec.\n(0.167)', 'Hyb.\n(0.222)', 'Cen.\n(0.333)', 'Lin.\n(0.600)']

atk_order = ['D1', 'D2', 'A1', 'A1o', 'A2s', 'A2']
attack_labels = {
    'D1':  'D1 (News Poison)',
    'D2':  'D2 (Indirect Inj.)',
    'A1':  r'$A1_{subtle}$ (1-Analyst)',
    'A1o': r'$A1_{overt}$ (1-Analyst)',
    'A2s': r'$A2_{subtle}$ (2-Analyst)',
    'A2':  r'$A2_{overt}$ (2-Analyst)',
}
markers   = {'D1':'o','D2':'s','A1':'^','A1o':'v','A2s':'P','A2':'D'}
color_q1  = '#C0392B'
color_q3  = '#2980B9'
marker_size = 120

fig, ax = plt.subplots(figsize=(10, 6))

# ── scatter ───────────────────────────────────────────────────────────────────
for atk in atk_order:
    for period, col in [("Q1 2026", color_q1), ("Q3 2025", color_q3)]:
        sub = df[(df["Attack"]==atk) & (df["Period"]==period)].copy()
        sub["x"] = sub["Architecture"].map(arch_map)
        ax.scatter(sub["x"], sub["FWVS"], marker=markers[atk],
                   s=marker_size, color=col, edgecolors='white',
                   linewidths=0.9, zorder=5)

# ── trend lines per period ────────────────────────────────────────────────────
for period, col in [("Q1 2026", color_q1), ("Q3 2025", color_q3)]:
    sub = df[df["Period"]==period].copy()
    sub["x"] = sub["Architecture"].map(arch_map)
    slopes, intercepts = [], []
    for atk in atk_order:
        s_atk = sub[sub["Attack"]==atk].sort_values("x")
        if len(s_atk) >= 2:
            s, i, *_ = stats.linregress(s_atk["x"], s_atk["FWVS"])
            slopes.append(s); intercepts.append(i)
    avg_s = np.mean(slopes); avg_i = np.mean(intercepts)
    x_line = np.linspace(0.13, 0.65, 100)
    ax.plot(x_line, avg_s * x_line + avg_i, color=col,
            linewidth=2.0, linestyle='--', alpha=0.55, zorder=3)

# ── legend: shapes ────────────────────────────────────────────────────────────
shape_handles = [
    ax.scatter([], [], marker=markers[atk], s=marker_size,
               color='#555555', edgecolors='white', linewidths=0.9,
               label=attack_labels[atk])
    for atk in atk_order
]
period_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=color_q1,
           markersize=11, label='Q1 2026 (BUY-targeted, bearish)', linestyle='None'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=color_q3,
           markersize=11, label='Q3 2025 (SELL-targeted, bullish)', linestyle='None'),
]
leg1 = ax.legend(handles=shape_handles, title='Attack type', fontsize=11,
                 title_fontsize=11, loc='upper left', framealpha=0.93)
ax.add_artist(leg1)
ax.legend(handles=period_handles, title='Period', fontsize=11,
          title_fontsize=11, loc='lower right', framealpha=0.93)

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(aps)
ax.set_xticklabels(arch_labels, fontsize=14)
ax.set_xlabel('APS  (Adversarial Signal Preservation Score)', fontsize=14)
ax.set_ylabel(r'FWVS  (×10²)', fontsize=14)
ax.set_xlim(0.10, 0.68)
ax.set_ylim(0.4, 2.9)
ax.yaxis.grid(True, alpha=0.25)
ax.set_axisbelow(True)
ax.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.savefig('figures/fig_fwvs_scatter.png', dpi=180, bbox_inches='tight')
plt.savefig('figures/fig_fwvs_scatter.pdf', bbox_inches='tight')
print("Saved figures/fig_fwvs_scatter.png and .pdf")
