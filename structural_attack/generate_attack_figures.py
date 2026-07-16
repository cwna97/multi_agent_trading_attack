"""
Generate Attack Effectiveness Visualizations

1. Grouped Bar Chart: ASR vs DCR for each attack type
2. Clustered Bar by Ticker: Attack effectiveness per ticker
3. Heatmap: Attack Type vs Ticker matrix
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import seaborn as sns

# Configuration
BASE_DIR = Path(__file__).parent
ATTACK_RESULTS_DIR = BASE_DIR / "attack_results" / "2025-01-01_2025-03-31"
OUTPUT_DIR = BASE_DIR / "figures"

SINGLE_ATTACKS = [
    ("attack_01_data_poisoning", "Data\nPoisoning"),
    ("attack_02_indirect_injection", "Indirect\nInjection"),
    ("attack_03_persuasive_bullish", "Persuasive\nAdversary"),
    ("attack_04_objective_hijacking", "Objective\nHijacking"),
    ("attack_05_jailbreaking_authority_override", "Jailbreak\n(Auth)"),
    ("attack_05_jailbreaking_hypothetical", "Jailbreak\n(Hypo)"),
]

COMBINED_ATTACKS = [
    ("combined_c1_insider_collusion", "C1"),
    ("combined_c2_news_risk_bypass", "C2"),
    ("combined_c3_data_bias", "C3"),
    ("combined_c4_bias_risk_ignore", "C4"),
    ("combined_c5_full_chain", "C5"),
    ("combined_c6_analysis_trade", "C6"),
]

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]


def load_attack_data():
    """Load ASR and DCR data for all attacks"""
    data = {}

    all_attacks = SINGLE_ATTACKS + COMBINED_ATTACKS

    for attack_id, attack_name in all_attacks:
        json_path = ATTACK_RESULTS_DIR / attack_id / "attack_summary.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                summary = json.load(f)
            metrics = summary.get('metrics', {})
            data[attack_name] = {
                'ASR': metrics.get('attack_success_rate', 0),
                'DCR': metrics.get('decision_change_rate', 0),
            }

    return data


def load_per_ticker_data():
    """Load per-ticker attack data"""
    # This would require parsing individual results
    # For now, we'll use the CSV files
    import pandas as pd

    ticker_data = {}

    for attack_id, attack_name in SINGLE_ATTACKS:
        csv_path = ATTACK_RESULTS_DIR / attack_id / "all_attack_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            ticker_data[attack_name] = {}

            for ticker in TICKERS:
                ticker_df = df[df['ticker'] == ticker]
                total = len(ticker_df)
                if total > 0:
                    # Count decision changes
                    if 'decision_changed' in ticker_df.columns:
                        dcr = ticker_df['decision_changed'].sum() / total * 100
                    else:
                        dcr = 0

                    # Count attack successes
                    if 'attack_succeeded' in ticker_df.columns:
                        asr = ticker_df['attack_succeeded'].sum() / total * 100
                    else:
                        asr = 0

                    ticker_data[attack_name][ticker] = {'ASR': asr, 'DCR': dcr}

    return ticker_data


def figure1_grouped_bar(data, output_path):
    """
    Figure 1: Grouped Bar Chart - ASR vs DCR for Single Attacks
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get single attack data
    single_names = [name for _, name in SINGLE_ATTACKS]
    asr_values = [data.get(name, {}).get('ASR', 0) for name in single_names]
    dcr_values = [data.get(name, {}).get('DCR', 0) for name in single_names]

    x = np.arange(len(single_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, asr_values, width, label='ASR (%)', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, dcr_values, width, label='DCR (%)', color='#3498db', edgecolor='black')

    ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Attack Effectiveness: ASR vs DCR (Single Attacks)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(single_names, fontsize=10)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure2_combined_attacks_bar(data, output_path):
    """
    Figure 2: Grouped Bar Chart - ASR vs DCR for Combined Attacks
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get combined attack data
    combined_names = [name for _, name in COMBINED_ATTACKS]
    combined_full_names = [
        "C1: Insider\nCollusion",
        "C2: News Risk\nBypass",
        "C3: Data\nBias",
        "C4: Bias Risk\nIgnore",
        "C5: Full\nChain",
        "C6: Analysis\nTrade"
    ]
    asr_values = [data.get(name, {}).get('ASR', 0) for name in combined_names]
    dcr_values = [data.get(name, {}).get('DCR', 0) for name in combined_names]

    x = np.arange(len(combined_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, asr_values, width, label='ASR (%)', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, dcr_values, width, label='DCR (%)', color='#3498db', edgecolor='black')

    ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Attack Effectiveness: ASR vs DCR (Combined Attacks)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(combined_full_names, fontsize=9)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure3_heatmap_dcr(output_path):
    """
    Figure 3: Heatmap - DCR across Attack Types and Tickers
    """
    import pandas as pd

    # Build matrix
    attack_names = [name for _, name in SINGLE_ATTACKS]
    matrix = []

    for attack_id, attack_name in SINGLE_ATTACKS:
        csv_path = ATTACK_RESULTS_DIR / attack_id / "all_attack_results.csv"
        row = []
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for ticker in TICKERS:
                ticker_df = df[df['ticker'] == ticker]
                total = len(ticker_df)
                if total > 0 and 'decision_changed' in ticker_df.columns:
                    dcr = ticker_df['decision_changed'].sum() / total * 100
                else:
                    dcr = 0
                row.append(dcr)
        else:
            row = [0] * len(TICKERS)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    # Labels
    ax.set_xticks(np.arange(len(TICKERS)))
    ax.set_yticks(np.arange(len(attack_names)))
    ax.set_xticklabels(TICKERS, fontsize=11)
    ax.set_yticklabels([name.replace('\n', ' ') for name in attack_names], fontsize=10)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Decision Change Rate (%)', rotation=-90, va="bottom", fontsize=11)

    # Add text annotations
    for i in range(len(attack_names)):
        for j in range(len(TICKERS)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="black" if matrix[i, j] < 60 else "white",
                          fontsize=9)

    ax.set_title('Decision Change Rate (%) by Attack Type and Ticker', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ticker', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Type', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure7_asr_dcr_per_ticker(output_path):
    """
    Figure 7: ASR vs DCR overlay for each ticker (subplots)
    Excludes META and Jailbreaking (Authority)
    """
    import pandas as pd

    # Exclude META
    tickers_filtered = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "TSLA"]

    # Exclude Jailbreaking (Authority) - Use abbreviations
    attacks_filtered = [
        ("attack_01_data_poisoning", "D.P."),
        ("attack_02_indirect_injection", "I.I."),
        ("attack_03_persuasive_bullish", "P.A."),
        ("attack_04_objective_hijacking", "O.H."),
        ("attack_05_jailbreaking_hypothetical", "J."),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    attack_names = [name for _, name in attacks_filtered]
    attack_ids = [aid for aid, _ in attacks_filtered]

    for idx, ticker in enumerate(tickers_filtered):
        ax = axes[idx]

        asr_values = []
        dcr_values = []

        for attack_id in attack_ids:
            csv_path = ATTACK_RESULTS_DIR / attack_id / "all_attack_results.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                ticker_df = df[df['ticker'] == ticker]
                total = len(ticker_df)
                if total > 0:
                    asr = ticker_df['attack_succeeded'].sum() / total * 100 if 'attack_succeeded' in ticker_df.columns else 0
                    dcr = ticker_df['decision_changed'].sum() / total * 100 if 'decision_changed' in ticker_df.columns else 0
                else:
                    asr, dcr = 0, 0
            else:
                asr, dcr = 0, 0
            asr_values.append(asr)
            dcr_values.append(dcr)

        x = np.arange(len(attack_names))
        width = 0.6

        # Draw DCR (hatched)
        bars_dcr = ax.bar(x, dcr_values, width, label='DCR',
                          color='white', edgecolor='#e74c3c', linewidth=1.5,
                          hatch='///', alpha=0.9)

        # Draw ASR (solid)
        bars_asr = ax.bar(x, asr_values, width, label='ASR',
                          color='#3498db', edgecolor='#2980b9', linewidth=1,
                          alpha=0.85)

        ax.set_title(f'{ticker}', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(attack_names, fontsize=12, rotation=0, ha='center')
        ax.set_ylim(0, 115)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='y', labelsize=11)

        # Add value labels
        for bar_dcr, bar_asr in zip(bars_dcr, bars_asr):
            dcr_h = bar_dcr.get_height()
            asr_h = bar_asr.get_height()
            ax.annotate(f'{dcr_h:.0f}',
                        xy=(bar_dcr.get_x() + bar_dcr.get_width() / 2, dcr_h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color='#c0392b', fontweight='bold')
            if asr_h > 8:
                ax.annotate(f'{asr_h:.0f}',
                            xy=(bar_asr.get_x() + bar_asr.get_width() / 2, asr_h / 2),
                            ha='center', va='center', fontsize=10, color='white', fontweight='bold')

        if idx == 0:
            ax.legend(loc='upper right', fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel('Rate (%)', fontsize=12)

    # Title removed per user request
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure6_asr_dcr_overlay(data, output_path):
    """
    Figure 6: Overlaid Bar Chart - ASR (solid) inside DCR (hatched)
    Shows ASR as solid bars and DCR as hatched bars overlaid
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get single attack data
    single_names = [name for _, name in SINGLE_ATTACKS]
    asr_values = [data.get(name, {}).get('ASR', 0) for name in single_names]
    dcr_values = [data.get(name, {}).get('DCR', 0) for name in single_names]

    x = np.arange(len(single_names))
    width = 0.6

    # Draw DCR first (hatched, background)
    bars_dcr = ax.bar(x, dcr_values, width, label='DCR (%)',
                      color='white', edgecolor='#e74c3c', linewidth=2,
                      hatch='///', alpha=0.9)

    # Draw ASR on top (solid, foreground)
    bars_asr = ax.bar(x, asr_values, width, label='ASR (%)',
                      color='#3498db', edgecolor='#2980b9', linewidth=1.5,
                      alpha=0.85)

    ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Attack Effectiveness: ASR (solid) vs DCR (hatched)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(single_names, fontsize=10)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar_dcr, bar_asr) in enumerate(zip(bars_dcr, bars_asr)):
        dcr_h = bar_dcr.get_height()
        asr_h = bar_asr.get_height()
        # DCR label at top
        ax.annotate(f'{dcr_h:.1f}',
                    xy=(bar_dcr.get_x() + bar_dcr.get_width() / 2, dcr_h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#c0392b', fontweight='bold')
        # ASR label inside bar
        if asr_h > 10:
            ax.annotate(f'{asr_h:.1f}',
                        xy=(bar_asr.get_x() + bar_asr.get_width() / 2, asr_h / 2),
                        ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure5_heatmap_asr(output_path):
    """
    Figure 5: Heatmap - ASR across Attack Types and Tickers
    """
    import pandas as pd

    # Build matrix
    attack_names = [name for _, name in SINGLE_ATTACKS]
    matrix = []

    for attack_id, attack_name in SINGLE_ATTACKS:
        csv_path = ATTACK_RESULTS_DIR / attack_id / "all_attack_results.csv"
        row = []
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for ticker in TICKERS:
                ticker_df = df[df['ticker'] == ticker]
                total = len(ticker_df)
                if total > 0 and 'attack_succeeded' in ticker_df.columns:
                    asr = ticker_df['attack_succeeded'].sum() / total * 100
                else:
                    asr = 0
                row.append(asr)
        else:
            row = [0] * len(TICKERS)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    # Labels
    ax.set_xticks(np.arange(len(TICKERS)))
    ax.set_yticks(np.arange(len(attack_names)))
    ax.set_xticklabels(TICKERS, fontsize=11)
    ax.set_yticklabels([name.replace('\n', ' ') for name in attack_names], fontsize=10)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Attack Success Rate (%)', rotation=-90, va="bottom", fontsize=11)

    # Add text annotations
    for i in range(len(attack_names)):
        for j in range(len(TICKERS)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="black" if matrix[i, j] < 60 else "white",
                          fontsize=9)

    ax.set_title('Attack Success Rate (%) by Attack Type and Ticker', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ticker', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Type', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure4_all_attacks_comparison(data, output_path):
    """
    Figure 4: All attacks (Single + Combined) in one chart
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Combine all attacks
    all_names = [name for _, name in SINGLE_ATTACKS] + [name for _, name in COMBINED_ATTACKS]
    asr_values = [data.get(name, {}).get('ASR', 0) for name in all_names]
    dcr_values = [data.get(name, {}).get('DCR', 0) for name in all_names]

    x = np.arange(len(all_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, asr_values, width, label='ASR (%)', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, dcr_values, width, label='DCR (%)', color='#3498db', edgecolor='black')

    ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Attack Effectiveness: All Attacks Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, fontsize=9, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    # Add vertical line to separate single and combined attacks
    ax.axvline(x=5.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(2.5, 105, 'Single Attacks', ha='center', fontsize=11, style='italic')
    ax.text(8.5, 105, 'Combined Attacks', ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating Attack Effectiveness Figures")
    print("=" * 60)

    # Load data
    print("\nLoading attack data...")
    data = load_attack_data()

    print("\nData loaded:")
    for name, values in data.items():
        print(f"  {name}: ASR={values['ASR']:.1f}%, DCR={values['DCR']:.1f}%")

    # Generate figures
    print("\nGenerating figures...")

    # Figure 1: Single Attacks - Grouped Bar
    figure1_grouped_bar(data, OUTPUT_DIR / "fig1_single_attacks_bar.png")

    # Figure 2: Combined Attacks - Grouped Bar
    figure2_combined_attacks_bar(data, OUTPUT_DIR / "fig2_combined_attacks_bar.png")

    # Figure 3: Heatmap
    figure3_heatmap_dcr(OUTPUT_DIR / "fig3_dcr_heatmap.png")

    # Figure 4: All Attacks Combined
    figure4_all_attacks_comparison(data, OUTPUT_DIR / "fig4_all_attacks_comparison.png")

    # Figure 5: ASR Heatmap
    figure5_heatmap_asr(OUTPUT_DIR / "fig5_asr_heatmap.png")

    # Figure 6: ASR vs DCR Overlay
    figure6_asr_dcr_overlay(data, OUTPUT_DIR / "fig6_asr_dcr_overlay.png")

    # Figure 7: ASR vs DCR per Ticker
    figure7_asr_dcr_per_ticker(OUTPUT_DIR / "fig7_asr_dcr_per_ticker.png")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()