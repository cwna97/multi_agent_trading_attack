import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

plt.rcParams.update({'font.family': 'DejaVu Sans'})

df = pd.read_csv("/tmp/finance_metrics.csv")

atk_order  = ["D1","D2","A1","A1o","A2s","A2"]
atk_labels = ["D1","D2",r"$A1_{subtle}$",r"$A1_{overt}$",r"$A2_{subtle}$",r"$A2_{overt}$"]
arch_order = ["Decentralized","Hybrid","Centralized","Linear"]
arch_short = ["Dec.","Hyb.","Cen.","Lin."]

def pivot(period, metric):
    mat = np.full((len(atk_order), len(arch_order)), np.nan)
    for i, atk in enumerate(atk_order):
        for j, arch in enumerate(arch_order):
            row = df[(df["Period"]==period)&(df["Attack"]==atk)&(df["Architecture"]==arch)]
            if len(row): mat[i,j] = row[metric].values[0]
    return mat

fig, axes = plt.subplots(3, 2, figsize=(13, 13))
fig.subplots_adjust(hspace=0.55, wspace=0.35)

metrics = [
    ("ΔCR(%)",   "ΔCR (%)",   "RdYlGn"),
    ("ΔSharpe",  "ΔSharpe",   "RdYlGn"),
    ("ΔMDD(%)",  "ΔMDD (%)",  "RdYlGn_r"),  # MDD: less negative = better → reversed
]

for row_idx, (metric, label, cmap_name) in enumerate(metrics):
    for col_idx, (period, period_label) in enumerate([
        ("Q1 2026","Q1 2026 (BUY-targeted, bearish)"),
        ("Q3 2025","Q3 2025 (SELL-targeted, bullish)")
    ]):
        ax = axes[row_idx][col_idx]
        mat = pivot(period, metric)

        # symmetric colormap around 0
        vmax = np.nanmax(np.abs(mat))
        vmin = -vmax

        im = ax.imshow(mat, cmap=cmap_name, vmin=vmin, vmax=vmax, aspect='auto')

        # cell annotations
        for i in range(len(atk_order)):
            for j in range(len(arch_order)):
                val = mat[i, j]
                if np.isnan(val): continue
                norm_val = (val - vmin) / (vmax - vmin + 1e-10)
                txt_color = 'white' if (norm_val < 0.25 or norm_val > 0.75) else 'black'
                prefix = "+" if val > 0 else ""
                ax.text(j, i, f"{prefix}{val:.1f}", ha='center', va='center',
                        fontsize=11.5, fontweight='bold', color=txt_color)

        ax.set_xticks(range(len(arch_order)))
        ax.set_xticklabels(arch_short, fontsize=12)
        ax.set_yticks(range(len(atk_order)))
        ax.set_yticklabels(atk_labels if col_idx == 0 else [], fontsize=12)
        ax.set_title(f"{label}\n{period_label}", fontsize=12, fontweight='bold', pad=8)

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=10)

# row labels on the left
for row_idx, (_, label, _) in enumerate(metrics):
    axes[row_idx][0].set_ylabel(label, fontsize=13, fontweight='bold', labelpad=8)

fig.suptitle("Financial Impact of Adversarial Attacks\n"
             "(ΔCR, ΔSharpe, ΔMDD relative to clean baseline)",
             fontsize=15, fontweight='bold', y=0.99)

plt.savefig('figures/fig_finance_impact_heatmap.png', dpi=160, bbox_inches='tight')
plt.savefig('figures/fig_finance_impact_heatmap.pdf', bbox_inches='tight')
print("Saved figures/fig_finance_impact_heatmap.png and .pdf")
