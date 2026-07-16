import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
})

arch_labels = ['Dec.\n(0.167)', 'Hyb.\n(0.222)', 'Cen.\n(0.333)', 'Lin.\n(0.600)']
x = np.arange(4)
w = 0.28

# FWVS (×10²)
fwvs_q1 = np.array([1.5078, 1.8187, 2.0754, 1.8181])
fwvs_q3 = np.array([1.1652, 1.5167, 1.6516, 1.8439])

# avg ΔCR across attacks
dcr_q1 = np.array([ 3.12, -2.73, -10.03,  1.40])
dcr_q3 = np.array([ 1.32, -9.35,  -4.23, -7.63])

# avg ΔSharpe
dsh_q1 = np.array([ 0.249,  0.891, -0.284,  0.373])
dsh_q3 = np.array([ 0.502, -2.653, -1.350, -1.993])

# avg ΔMDD
dmdd_q1 = np.array([ 3.10, -3.90, -11.75,  1.05])
dmdd_q3 = np.array([ 5.08,  1.61,   4.40,  2.80])

arch_colors = ['#4C72B0','#7B2D8B','#DD8452','#C44E52']

fig, axes = plt.subplots(2, 1, figsize=(11, 10))
fig.subplots_adjust(hspace=0.52)

for ax, (period_label, fwvs, dcr, dsh, dmdd) in zip(axes, [
    ("Q1 2026  —  BUY-targeted (bearish)", fwvs_q1, dcr_q1, dsh_q1, dmdd_q1),
    ("Q3 2025  —  SELL-targeted (bullish)", fwvs_q3, dcr_q3, dsh_q3, dmdd_q3),
]):
    ax2 = ax.twinx()  # secondary axis for FWVS

    # ── FWVS bars (background, secondary axis) ────────────────────────────────
    fwvs_bars = ax2.bar(x, fwvs, width=0.72, color=arch_colors, alpha=0.18,
                        edgecolor='none', zorder=1, label='FWVS (×10²)')
    ax2.set_ylabel('FWVS (×10²)', fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=11)
    ax2.set_ylim(0, max(fwvs) * 2.2)
    ax2.spines['top'].set_visible(False)
    # FWVS value labels
    for xi, v in zip(x, fwvs):
        ax2.text(xi, v + 0.04, f'{v:.2f}', ha='center', va='bottom',
                 fontsize=10, color='gray', fontstyle='italic')

    # ── ΔCR / ΔSharpe / ΔMDD bars (primary axis) ─────────────────────────────
    b1 = ax.bar(x - w, dcr,  w, color=arch_colors, alpha=0.88,
                edgecolor='white', linewidth=1.0, label='ΔCR (%)', zorder=3)
    b2 = ax.bar(x,     dsh,  w, color=arch_colors, alpha=0.55,
                edgecolor='white', linewidth=1.0, hatch='//', label='ΔSharpe', zorder=3)
    b3 = ax.bar(x + w, dmdd, w, color=arch_colors, alpha=0.35,
                edgecolor='white', linewidth=1.0, hatch='xx', label='ΔMDD (%)', zorder=3)

    # value labels on ΔCR only (most important)
    for bar, val in zip(b1, dcr):
        ypos = val + 0.3 if val >= 0 else val - 0.8
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:+.1f}', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=10, fontweight='bold', color=bar.get_facecolor())

    ax.axhline(0, color='black', linewidth=0.8, alpha=0.4, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, fontsize=13)
    ax.set_ylabel('Portfolio metric change', fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.yaxis.grid(True, alpha=0.2); ax.set_axisbelow(True)
    ax.set_title(period_label, fontsize=13, fontweight='bold', pad=10)

    # legend (primary axis only)
    handles1, labels1 = ax.get_legend_handles_labels()
    from matplotlib.patches import Patch
    fwvs_patch = Patch(facecolor='gray', alpha=0.25, label='FWVS (×10²)')
    ax.legend(handles=handles1 + [fwvs_patch],
              fontsize=11, loc='lower left', framealpha=0.92, ncol=2)

plt.suptitle('Financial Impact of Adversarial Attacks\nΔCR, ΔSharpe, ΔMDD (avg. across attacks)  +  FWVS per architecture',
             fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('figures/fig_finance_fwvs.png', dpi=160, bbox_inches='tight')
plt.savefig('figures/fig_finance_fwvs.pdf', bbox_inches='tight')
print("Saved figures/fig_finance_fwvs.png and .pdf")
