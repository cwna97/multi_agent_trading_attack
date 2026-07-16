import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fwvs_df = pd.read_csv("/tmp/fwvs_per_attack.csv")
fin_df  = pd.read_csv("/tmp/finance_metrics.csv")

arch_map   = {"Decentralized": 0.167, "Hybrid": 0.222, "Centralized": 0.333, "Linear": 0.600}
arch_short = {"Decentralized": "Dec", "Hybrid": "Hyb", "Centralized": "Cen", "Linear": "Lin"}
aps_vals   = np.array([0.167, 0.222, 0.333, 0.600])
arch_labels = ['Dec.\n(0.167)', 'Hyb.\n(0.222)', 'Cen.\n(0.333)', 'Lin.\n(0.600)']

# Attack group colors
atk_colors = {
    'D1':  '#2166AC',  # dark blue
    'D2':  '#74ADD1',  # light blue
    'A1':  '#E87B2C',  # dark orange
    'A1o': '#F5A65B',  # light orange
    'A2s': '#C44E52',  # red-orange
    'A2':  '#D62728',  # dark red
}
atk_labels_map = {
    'D1':  'D1 (Data Poison)',
    'D2':  'D2 (Indirect Inj.)',
    'A1':  r'$A1_{subtle}$ (1-Analyst)',
    'A1o': r'$A1_{overt}$ (1-Analyst)',
    'A2s': r'$A2_{subtle}$ (2-Analyst)',
    'A2':  r'$A2_{overt}$ (2-Analyst)',
}
atk_order = ['D1', 'D2', 'A1', 'A1o', 'A2s', 'A2']

# Merge FWVS + ΔCR
merged = fwvs_df.merge(
    fin_df[["Period", "Attack", "Architecture", "ΔCR(%)"]],
    on=["Period", "Attack", "Architecture"], how="left"
)
merged["APS"]  = merged["Architecture"].map(arch_map)
merged["arch_short"] = merged["Architecture"].map(arch_short)
merged["abs_dcr"] = merged["ΔCR(%)"].abs()

BUBBLE_SCALE = 400  # scale factor for bubble size


def draw_panel(ax, period, period_title):
    sub = merged[merged["Period"] == period].copy()
    # vertical dashed lines at each APS
    for a in aps_vals:
        ax.axvline(a, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=1)

    for atk in atk_order:
        rows = sub[sub["Attack"] == atk].sort_values("APS")
        if rows.empty:
            continue
        sizes = rows["abs_dcr"] / rows["abs_dcr"].max() * BUBBLE_SCALE + 60
        sc = ax.scatter(rows["APS"], rows["FWVS"],
                        s=sizes, color=atk_colors[atk],
                        edgecolors='white', linewidths=1.0,
                        alpha=0.88, zorder=3,
                        label=atk_labels_map[atk])
        # architecture labels
        for _, row in rows.iterrows():
            ax.text(row["APS"] + 0.008, row["FWVS"] + 0.025,
                    row["arch_short"], fontsize=8.5, color=atk_colors[atk],
                    va='bottom', ha='left')

    ax.set_xticks(aps_vals)
    ax.set_xticklabels(arch_labels, fontsize=12)
    ax.set_xlabel('APS  (Adversarial Signal Preservation Score)', fontsize=12)
    ax.set_ylabel('FWVS  (×10²)', fontsize=12)
    ax.set_xlim(0.10, 0.68)
    ax.set_ylim(0.5, 2.9)
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_title(f'Financially Weighted Vulnerability vs. Centrality\n({period})\n'
                 r'Bubble size $\propto$ |ΔCR|',
                 fontsize=12, fontweight='bold', pad=10)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True; h2.append(h); l2.append(l)
    ax.legend(h2, l2, fontsize=9, loc='upper left', framealpha=0.92,
              title='Attack type', title_fontsize=9)


# ── v1: Q1 + Q3 side by side ──────────────────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
draw_panel(ax1, "Q1 2026", "Q1 2026")
draw_panel(ax2, "Q3 2025", "Q3 2025")
plt.tight_layout()
fig1.savefig('figures/fig_fwvs_bubble_v1.png', dpi=180, bbox_inches='tight')
fig1.savefig('figures/fig_fwvs_bubble_v1.pdf', bbox_inches='tight')
print("Saved v1")

# ── v2: Q3 2025 single panel ──────────────────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(8, 6))
draw_panel(ax3, "Q3 2025", "Q3 2025")
plt.tight_layout()
fig2.savefig('figures/fig_fwvs_bubble_v2.png', dpi=180, bbox_inches='tight')
fig2.savefig('figures/fig_fwvs_bubble_v2.pdf', bbox_inches='tight')
print("Saved v2")

# ── v3: both periods in one panel (border=period, fill=attack) ────────────────
fig3, ax4 = plt.subplots(figsize=(9, 6))
period_style = {"Q1 2026": ("D", 1.5), "Q3 2025": ("o", 1.5)}  # marker, lw
period_color_edge = {"Q1 2026": "#C0392B", "Q3 2025": "#2980B9"}

for a in aps_vals:
    ax4.axvline(a, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=1)

for period, (mrkr, lw) in period_style.items():
    sub = merged[merged["Period"] == period].copy()
    for atk in atk_order:
        rows = sub[sub["Attack"] == atk].sort_values("APS")
        if rows.empty:
            continue
        sizes = rows["abs_dcr"] / merged["abs_dcr"].max() * BUBBLE_SCALE + 60
        ax4.scatter(rows["APS"], rows["FWVS"],
                    s=sizes, color=atk_colors[atk],
                    edgecolors=period_color_edge[period],
                    linewidths=1.8, alpha=0.82, zorder=3,
                    marker=mrkr)

# legend: attack colors
atk_handles = [plt.scatter([], [], color=atk_colors[a], s=80, edgecolors='white',
                            linewidths=0.8, label=atk_labels_map[a])
               for a in atk_order]
from matplotlib.lines import Line2D
period_handles = [
    Line2D([0],[0], marker='D', color='w', markerfacecolor='gray',
           markeredgecolor='#C0392B', markeredgewidth=1.8, markersize=9,
           label='Q1 2026 (BUY, bearish)', linestyle='None'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
           markeredgecolor='#2980B9', markeredgewidth=1.8, markersize=9,
           label='Q3 2025 (SELL, bullish)', linestyle='None'),
]
leg1 = ax4.legend(handles=atk_handles, title='Attack type', fontsize=9,
                  title_fontsize=9, loc='upper left', framealpha=0.92)
ax4.add_artist(leg1)
ax4.legend(handles=period_handles, title='Period (edge color)', fontsize=9,
           title_fontsize=9, loc='lower right', framealpha=0.92)

ax4.set_xticks(aps_vals)
ax4.set_xticklabels(arch_labels, fontsize=12)
ax4.set_xlabel('APS  (Adversarial Signal Preservation Score)', fontsize=12)
ax4.set_ylabel('FWVS  (×10²)', fontsize=12)
ax4.set_xlim(0.10, 0.68)
ax4.set_ylim(0.5, 2.9)
ax4.yaxis.grid(True, alpha=0.25)
ax4.set_axisbelow(True)
ax4.tick_params(axis='y', labelsize=11)
ax4.set_title('Financially Weighted Vulnerability vs. Centrality\n'
              r'(Bubble size $\propto$ |ΔCR|, edge = period)',
              fontsize=12, fontweight='bold', pad=10)

plt.tight_layout()
fig3.savefig('figures/fig_fwvs_bubble_v3.png', dpi=180, bbox_inches='tight')
fig3.savefig('figures/fig_fwvs_bubble_v3.pdf', bbox_inches='tight')
print("Saved v3")
