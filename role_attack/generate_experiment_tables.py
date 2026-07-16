"""
Generate LaTeX Experiment Tables for Attack Vulnerability Analysis

Table 1: Financial Performance Metrics (CR, AR, SR, MDD)
Table 2: Attack Success Metrics (ASR, DCR)
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from scipy import stats
from portfolio_simulator import PortfolioSimulator


# Configuration
BASE_DIR = Path(__file__).parent
ATTACK_RESULTS_DIR = BASE_DIR / "attack_results" / "2025-01-01_2025-03-31"
TRADING_RESULTS_DIR = BASE_DIR / "trading_results" / "2025-01-01_2025-03-31"
BASELINE_DIR = BASE_DIR / "baseline_reports"
PRICE_DATA_DIR = BASE_DIR / "tradingagents" / "dataflows" / "local_data" / "market_data" / "price_data"

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

SINGLE_ATTACKS = [
    "attack_01_data_poisoning",
    "attack_02_indirect_injection",
    "attack_03_persuasive_bullish",
    "attack_04_objective_hijacking",
    "attack_05_jailbreaking_authority_override",
    "attack_05_jailbreaking_hypothetical",
]

COMBINED_ATTACKS = [
    "combined_c1_insider_collusion",
    "combined_c2_news_risk_bypass",
    "combined_c3_data_bias",
    "combined_c4_bias_risk_ignore",
    "combined_c5_full_chain",
    "combined_c6_analysis_trade",
]

ATTACK_DISPLAY_NAMES = {
    "attack_01_data_poisoning": "Data Poisoning",
    "attack_02_indirect_injection": "Indirect Injection",
    "attack_03_persuasive_bullish": "Persuasive Adversary",
    "attack_04_objective_hijacking": "Objective Hijacking",
    "attack_05_jailbreaking_authority_override": "Jailbreaking (Authority)",
    "attack_05_jailbreaking_hypothetical": "Jailbreaking (Hypothetical)",
    "combined_c1_insider_collusion": "C1: Insider Collusion",
    "combined_c2_news_risk_bypass": "C2: News Risk Bypass",
    "combined_c3_data_bias": "C3: Data Bias",
    "combined_c4_bias_risk_ignore": "C4: Bias Risk Ignore",
    "combined_c5_full_chain": "C5: Full Chain",
    "combined_c6_analysis_trade": "C6: Analysis Trade",
}


def load_normal_performance():
    """Load normal trading performance (Ours)"""
    csv_path = TRADING_RESULTS_DIR / "performance_metrics.csv"
    df = pd.read_csv(csv_path)
    return df


def load_baseline_performance():
    """Load B&H and MACD baseline performance"""
    csv_path = BASELINE_DIR / "comparison_2025_Q1_final.csv"
    df = pd.read_csv(csv_path)
    return df


def load_attack_summary(attack_name):
    """Load attack summary (ASR, DCR)"""
    json_path = ATTACK_RESULTS_DIR / attack_name / "attack_summary.json"
    with open(json_path, 'r') as f:
        return json.load(f)


def calculate_metrics_from_history(history, seed_money, trading_days):
    """Calculate AR and SR from portfolio history"""
    if not history or len(history) < 2:
        return {'AR': 0, 'SR': 0}

    history_df = pd.DataFrame(history)

    # Calculate daily returns
    history_df['daily_return'] = history_df['portfolio_value'].pct_change()
    daily_returns = history_df['daily_return'].dropna()

    # Calculate CR
    final_value = history_df['portfolio_value'].iloc[-1]
    cr = (final_value - seed_money) / seed_money * 100

    # Calculate AR (Annualized Return)
    # AR = ((1 + CR/100)^(252/trading_days) - 1) * 100
    if trading_days > 0:
        ar = ((1 + cr/100) ** (252 / trading_days) - 1) * 100
    else:
        ar = 0

    # Calculate SR (Sharpe Ratio)
    # SR = mean(daily_returns) / std(daily_returns) * sqrt(252)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sr = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sr = 0

    return {'AR': ar, 'SR': sr}


def calculate_attack_performance(attack_name):
    """Calculate financial performance under attack scenario"""
    attack_csv = ATTACK_RESULTS_DIR / attack_name / "all_attack_results.csv"

    if not attack_csv.exists():
        print(f"Warning: {attack_csv} not found")
        return None

    # Load attack decisions
    attack_df = pd.read_csv(attack_csv)

    # Rename attack_decision to decision for portfolio simulator
    attack_df = attack_df.rename(columns={'attack_decision': 'decision'})
    attack_df['date'] = pd.to_datetime(attack_df['date'])

    # Initialize simulator
    simulator = PortfolioSimulator(seed_money=100000.0, commission_rate=0.001)

    results = {}
    for ticker in TICKERS:
        # Get price data
        price_file = PRICE_DATA_DIR / f"{ticker}-YFin-data-2023-01-01-2025-12-31.csv"
        if not price_file.exists():
            print(f"Warning: Price data not found for {ticker}")
            continue

        prices_df = pd.read_csv(price_file)
        prices_df['Date'] = pd.to_datetime(prices_df['Date']).dt.tz_localize(None)
        prices_df = prices_df.rename(columns={'Close': 'close', 'Date': 'date'})

        # Filter to date range
        prices_df = prices_df[(prices_df['date'] >= '2025-01-01') & (prices_df['date'] <= '2025-03-31')]

        # Run simulation
        sim_result = simulator.simulate(attack_df, prices_df[['date', 'close']], ticker)

        if 'error' not in sim_result:
            # Calculate AR and SR from history
            extra_metrics = calculate_metrics_from_history(
                sim_result.get('history', []),
                simulator.seed_money,
                sim_result.get('trading_days', 0)
            )

            results[ticker] = {
                'CR': sim_result['total_return_pct'],
                'AR': extra_metrics['AR'],
                'SR': extra_metrics['SR'],
                'MDD': sim_result['max_drawdown_pct'],
            }

    # Calculate averages
    if results:
        avg_cr = np.mean([r['CR'] for r in results.values()])
        avg_ar = np.mean([r['AR'] for r in results.values()])
        avg_sr = np.mean([r['SR'] for r in results.values()])
        avg_mdd = np.mean([r['MDD'] for r in results.values()])
        results['Average'] = {'CR': avg_cr, 'AR': avg_ar, 'SR': avg_sr, 'MDD': avg_mdd}

    return results


def generate_table1_latex():
    """Generate Table 1: Financial Performance Metrics"""

    # Load data
    normal_df = load_normal_performance()
    baseline_df = load_baseline_performance()

    # Extract baseline averages
    bh_data = baseline_df[baseline_df['strategy'] == 'B&H']
    macd_data = baseline_df[baseline_df['strategy'] == 'MACD']

    bh_avg = {
        'CR': bh_data['CR%'].mean(),
        'AR': bh_data['ARR%'].mean(),
        'SR': bh_data['SR'].mean(),
        'MDD': bh_data['MDD%'].mean(),
    }

    macd_avg = {
        'CR': macd_data['CR%'].mean(),
        'AR': macd_data['ARR%'].mean(),
        'SR': macd_data['SR'].mean(),
        'MDD': macd_data['MDD%'].mean(),
    }

    ours_avg = {
        'CR': normal_df['CR%'].mean(),
        'AR': normal_df['ARR%'].mean(),
        'SR': normal_df['SR'].mean(),
        'MDD': normal_df['MDD%'].mean(),
    }

    # Calculate attack performance
    attack_performance = {}
    all_attacks = SINGLE_ATTACKS + COMBINED_ATTACKS

    print("Calculating attack performance...")
    for attack in all_attacks:
        print(f"  Processing {attack}...")
        perf = calculate_attack_performance(attack)
        if perf and 'Average' in perf:
            attack_performance[attack] = perf['Average']

    # Generate LaTeX
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Financial Performance Comparison: Normal vs Under Attack}
\label{tab:financial_performance}
\begin{tabular}{l|rrrr}
\toprule
\textbf{Condition} & \textbf{CR (\%)} & \textbf{AR (\%)} & \textbf{SR} & \textbf{MDD (\%)} \\
\midrule
\multicolumn{5}{l}{\textit{Normal Environment}} \\
\midrule
B\&H & """ + f"{bh_avg['CR']:.2f}" + r""" & """ + f"{bh_avg['AR']:.2f}" + r""" & """ + f"{bh_avg['SR']:.3f}" + r""" & """ + f"{bh_avg['MDD']:.2f}" + r""" \\
MACD & """ + f"{macd_avg['CR']:.2f}" + r""" & """ + f"{macd_avg['AR']:.2f}" + r""" & """ + f"{macd_avg['SR']:.3f}" + r""" & """ + f"{macd_avg['MDD']:.2f}" + r""" \\
\textbf{Ours} & """ + f"\\textbf{{{ours_avg['CR']:.2f}}}" + r""" & """ + f"\\textbf{{{ours_avg['AR']:.2f}}}" + r""" & """ + f"{ours_avg['SR']:.3f}" + r""" & """ + f"\\textbf{{{ours_avg['MDD']:.2f}}}" + r""" \\
\midrule
\multicolumn{5}{l}{\textit{Under Single Attack}} \\
\midrule
"""

    # Add single attack rows
    for attack in SINGLE_ATTACKS:
        name = ATTACK_DISPLAY_NAMES[attack]
        if attack in attack_performance:
            perf = attack_performance[attack]
            latex += f"{name} & {perf['CR']:.2f} & {perf['AR']:.2f} & {perf['SR']:.3f} & {perf['MDD']:.2f} \\\\\n"
        else:
            latex += f"{name} & - & - & - & - \\\\\n"

    latex += r"""\midrule
\multicolumn{5}{l}{\textit{Under Combined Attack}} \\
\midrule
"""

    # Add combined attack rows
    for attack in COMBINED_ATTACKS:
        name = ATTACK_DISPLAY_NAMES[attack]
        if attack in attack_performance:
            perf = attack_performance[attack]
            latex += f"{name} & {perf['CR']:.2f} & {perf['AR']:.2f} & {perf['SR']:.3f} & {perf['MDD']:.2f} \\\\\n"
        else:
            latex += f"{name} & - & - & - & - \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex


def generate_per_ticker_tables_latex():
    """Generate per-ticker financial performance tables (7 tables)"""

    # Load data
    normal_df = load_normal_performance()
    baseline_df = load_baseline_performance()

    # Calculate attack performance for all attacks (with per-ticker data)
    all_attacks = SINGLE_ATTACKS + COMBINED_ATTACKS
    attack_performance_all = {}

    print("Calculating per-ticker attack performance...")
    for attack in all_attacks:
        print(f"  Processing {attack}...")
        perf = calculate_attack_performance(attack)
        if perf:
            attack_performance_all[attack] = perf

    tables = {}

    for ticker in TICKERS:
        # Get baseline data for this ticker
        bh_ticker = baseline_df[(baseline_df['strategy'] == 'B&H') & (baseline_df['ticker'] == ticker)]
        macd_ticker = baseline_df[(baseline_df['strategy'] == 'MACD') & (baseline_df['ticker'] == ticker)]
        ours_ticker = normal_df[normal_df['ticker'] == ticker]

        if len(bh_ticker) == 0 or len(ours_ticker) == 0:
            print(f"Warning: Missing data for {ticker}")
            continue

        bh_data = {
            'CR': bh_ticker['CR%'].values[0],
            'AR': bh_ticker['ARR%'].values[0],
            'SR': bh_ticker['SR'].values[0],
            'MDD': bh_ticker['MDD%'].values[0],
        }

        macd_data = {
            'CR': macd_ticker['CR%'].values[0] if len(macd_ticker) > 0 else 0,
            'AR': macd_ticker['ARR%'].values[0] if len(macd_ticker) > 0 else 0,
            'SR': macd_ticker['SR'].values[0] if len(macd_ticker) > 0 else 0,
            'MDD': macd_ticker['MDD%'].values[0] if len(macd_ticker) > 0 else 0,
        }

        ours_data = {
            'CR': ours_ticker['CR%'].values[0],
            'AR': ours_ticker['ARR%'].values[0],
            'SR': ours_ticker['SR'].values[0],
            'MDD': ours_ticker['MDD%'].values[0],
        }

        # Generate LaTeX for this ticker
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Financial Performance for """ + ticker + r""": Normal vs Under Attack}
\label{tab:financial_performance_""" + ticker.lower() + r"""}
\begin{tabular}{l|rrrr}
\toprule
\textbf{Condition} & \textbf{CR (\%)} & \textbf{AR (\%)} & \textbf{SR} & \textbf{MDD (\%)} \\
\midrule
\multicolumn{5}{l}{\textit{Normal Environment}} \\
\midrule
"""
        latex += f"B\\&H & {bh_data['CR']:.2f} & {bh_data['AR']:.2f} & {bh_data['SR']:.3f} & {bh_data['MDD']:.2f} \\\\\n"
        latex += f"MACD & {macd_data['CR']:.2f} & {macd_data['AR']:.2f} & {macd_data['SR']:.3f} & {macd_data['MDD']:.2f} \\\\\n"
        latex += f"\\textbf{{Ours}} & \\textbf{{{ours_data['CR']:.2f}}} & \\textbf{{{ours_data['AR']:.2f}}} & {ours_data['SR']:.3f} & \\textbf{{{ours_data['MDD']:.2f}}} \\\\\n"

        latex += r"""\midrule
\multicolumn{5}{l}{\textit{Under Single Attack}} \\
\midrule
"""

        # Add single attack rows for this ticker
        for attack in SINGLE_ATTACKS:
            name = ATTACK_DISPLAY_NAMES[attack]
            if attack in attack_performance_all and ticker in attack_performance_all[attack]:
                perf = attack_performance_all[attack][ticker]
                latex += f"{name} & {perf['CR']:.2f} & {perf['AR']:.2f} & {perf['SR']:.3f} & {perf['MDD']:.2f} \\\\\n"
            else:
                latex += f"{name} & - & - & - & - \\\\\n"

        latex += r"""\midrule
\multicolumn{5}{l}{\textit{Under Combined Attack}} \\
\midrule
"""

        # Add combined attack rows for this ticker
        for attack in COMBINED_ATTACKS:
            name = ATTACK_DISPLAY_NAMES[attack]
            if attack in attack_performance_all and ticker in attack_performance_all[attack]:
                perf = attack_performance_all[attack][ticker]
                latex += f"{name} & {perf['CR']:.2f} & {perf['AR']:.2f} & {perf['SR']:.3f} & {perf['MDD']:.2f} \\\\\n"
            else:
                latex += f"{name} & - & - & - & - \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

        tables[ticker] = latex

    return tables


def calculate_statistical_tests(normal_per_ticker, attack_performance_all):
    """
    Calculate statistical significance tests (p-values)

    Uses paired t-test and Wilcoxon signed-rank test to compare
    Normal (Ours) vs each Attack scenario across 7 tickers.
    """
    results = {}

    # Extract normal values per ticker
    normal_cr = [normal_per_ticker[t]['CR'] for t in TICKERS if t in normal_per_ticker]
    normal_mdd = [normal_per_ticker[t]['MDD'] for t in TICKERS if t in normal_per_ticker]

    all_attacks = SINGLE_ATTACKS + COMBINED_ATTACKS

    for attack in all_attacks:
        if attack not in attack_performance_all:
            continue

        attack_perf = attack_performance_all[attack]

        # Extract attack values per ticker (same order as normal)
        attack_cr = []
        attack_mdd = []
        valid_tickers = []

        for t in TICKERS:
            if t in normal_per_ticker and t in attack_perf:
                valid_tickers.append(t)
                attack_cr.append(attack_perf[t]['CR'])
                attack_mdd.append(attack_perf[t]['MDD'])

        if len(valid_tickers) < 3:  # Need at least 3 samples for meaningful test
            continue

        # Recalculate normal values for valid tickers only
        normal_cr_valid = [normal_per_ticker[t]['CR'] for t in valid_tickers]
        normal_mdd_valid = [normal_per_ticker[t]['MDD'] for t in valid_tickers]

        # Paired t-test for CR (two-tailed)
        try:
            t_stat_cr, p_value_cr_ttest = stats.ttest_rel(normal_cr_valid, attack_cr)
        except:
            p_value_cr_ttest = 1.0

        # Wilcoxon signed-rank test for CR (more robust for small samples)
        try:
            stat_cr, p_value_cr_wilcox = stats.wilcoxon(normal_cr_valid, attack_cr)
        except:
            p_value_cr_wilcox = 1.0

        # Paired t-test for MDD
        try:
            t_stat_mdd, p_value_mdd_ttest = stats.ttest_rel(normal_mdd_valid, attack_mdd)
        except:
            p_value_mdd_ttest = 1.0

        # Wilcoxon signed-rank test for MDD
        try:
            stat_mdd, p_value_mdd_wilcox = stats.wilcoxon(normal_mdd_valid, attack_mdd)
        except:
            p_value_mdd_wilcox = 1.0

        # Calculate effect size (Cohen's d) for CR
        diff_cr = np.array(normal_cr_valid) - np.array(attack_cr)
        if diff_cr.std() > 0:
            cohens_d_cr = diff_cr.mean() / diff_cr.std()
        else:
            cohens_d_cr = 0

        # Calculate mean difference
        mean_diff_cr = np.mean(normal_cr_valid) - np.mean(attack_cr)
        mean_diff_mdd = np.mean(normal_mdd_valid) - np.mean(attack_mdd)

        results[attack] = {
            'n_samples': len(valid_tickers),
            'CR': {
                'mean_diff': mean_diff_cr,
                'p_ttest': p_value_cr_ttest,
                'p_wilcoxon': p_value_cr_wilcox,
                'cohens_d': cohens_d_cr,
            },
            'MDD': {
                'mean_diff': mean_diff_mdd,
                'p_ttest': p_value_mdd_ttest,
                'p_wilcoxon': p_value_mdd_wilcox,
            }
        }

    return results


def calculate_asr_dcr_per_ticker(attack, tickers_to_use=None):
    """Calculate ASR and DCR per ticker for a given attack"""
    import pandas as pd

    if tickers_to_use is None:
        tickers_to_use = TICKERS

    csv_path = ATTACK_RESULTS_DIR / attack / "all_attack_results.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    results = {}

    for ticker in tickers_to_use:
        ticker_df = df[df['ticker'] == ticker]
        total = len(ticker_df)
        if total > 0:
            asr = ticker_df['attack_succeeded'].sum() / total * 100 if 'attack_succeeded' in ticker_df.columns else 0
            dcr = ticker_df['decision_changed'].sum() / total * 100 if 'decision_changed' in ticker_df.columns else 0
            results[ticker] = {'ASR': asr, 'DCR': dcr}

    return results


def generate_table3_statistical_latex_v2(normal_per_ticker, attack_performance_all):
    """
    Generate Table 3: Statistical Significance Test Results

    - Excludes META ticker (n=6)
    - Excludes Jailbreaking (Authority) attack
    - Separate statistics for CR, MDD, ASR, DCR (each with Mean±Std and p-value)
    """

    # Tickers to use (exclude META)
    TICKERS_V2 = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "TSLA"]

    # Attacks to use (exclude Jailbreaking Authority)
    ATTACKS_V2 = [
        "attack_01_data_poisoning",
        "attack_02_indirect_injection",
        "attack_03_persuasive_bullish",
        "attack_04_objective_hijacking",
        "attack_05_jailbreaking_hypothetical",
    ]

    def format_pvalue(p):
        """Format p-value with significance stars"""
        if p < 0.001:
            return f"{p:.3f}***"
        elif p < 0.01:
            return f"{p:.3f}**"
        elif p < 0.05:
            return f"{p:.3f}*"
        else:
            return f"{p:.3f}"

    # Calculate statistics for each attack
    attack_stats = {}

    for attack in ATTACKS_V2:
        if attack not in attack_performance_all:
            continue

        attack_perf = attack_performance_all[attack]

        # CR and MDD: Normal vs Attack (paired comparison)
        normal_cr = []
        attack_cr = []
        normal_mdd = []
        attack_mdd = []

        for t in TICKERS_V2:
            if t in normal_per_ticker and t in attack_perf:
                normal_cr.append(normal_per_ticker[t]['CR'])
                attack_cr.append(attack_perf[t]['CR'])
                normal_mdd.append(normal_per_ticker[t]['MDD'])
                attack_mdd.append(attack_perf[t]['MDD'])

        # CR statistics (Attack - Normal: negative means attack hurt performance)
        cr_diffs = [a - n for a, n in zip(attack_cr, normal_cr)]
        cr_diff_mean = np.mean(cr_diffs)
        cr_diff_std = np.std(cr_diffs, ddof=1)  # sample std
        try:
            _, cr_p = stats.wilcoxon(normal_cr, attack_cr)
        except:
            cr_p = 1.0

        # MDD statistics (Attack - Normal: negative means attack caused worse drawdown)
        mdd_diffs = [a - n for a, n in zip(attack_mdd, normal_mdd)]
        mdd_diff_mean = np.mean(mdd_diffs)
        mdd_diff_std = np.std(mdd_diffs, ddof=1)  # sample std
        try:
            _, mdd_p = stats.wilcoxon(normal_mdd, attack_mdd)
        except:
            mdd_p = 1.0

        # ASR and DCR: One-sample test against 0 (baseline = no attack)
        per_ticker_asr_dcr = calculate_asr_dcr_per_ticker(attack, TICKERS_V2)

        if per_ticker_asr_dcr:
            asr_values = [per_ticker_asr_dcr[t]['ASR'] for t in TICKERS_V2 if t in per_ticker_asr_dcr]
            dcr_values = [per_ticker_asr_dcr[t]['DCR'] for t in TICKERS_V2 if t in per_ticker_asr_dcr]

            # ASR: Mean, Std and one-sample test vs 0
            asr_mean = np.mean(asr_values)
            asr_std = np.std(asr_values, ddof=1)  # sample std
            try:
                # Wilcoxon test for one sample: compare to 0
                # We use wilcoxon with the values themselves (compared to zeros)
                _, asr_p = stats.wilcoxon(asr_values, alternative='greater')
            except:
                # If all values are the same or other issues, use t-test
                try:
                    _, asr_p = stats.ttest_1samp(asr_values, 0, alternative='greater')
                except:
                    asr_p = 1.0

            # DCR: Mean, Std and one-sample test vs 0
            dcr_mean = np.mean(dcr_values)
            dcr_std = np.std(dcr_values, ddof=1)  # sample std
            try:
                _, dcr_p = stats.wilcoxon(dcr_values, alternative='greater')
            except:
                try:
                    _, dcr_p = stats.ttest_1samp(dcr_values, 0, alternative='greater')
                except:
                    dcr_p = 1.0
        else:
            asr_mean, asr_std, asr_p = 0, 0, 1.0
            dcr_mean, dcr_std, dcr_p = 0, 0, 1.0

        attack_stats[attack] = {
            'cr_diff_mean': cr_diff_mean,
            'cr_diff_std': cr_diff_std,
            'cr_p': cr_p,
            'mdd_diff_mean': mdd_diff_mean,
            'mdd_diff_std': mdd_diff_std,
            'mdd_p': mdd_p,
            'asr_mean': asr_mean,
            'asr_std': asr_std,
            'asr_p': asr_p,
            'dcr_mean': dcr_mean,
            'dcr_std': dcr_std,
            'dcr_p': dcr_p,
        }

    # Generate LaTeX (v4: All metrics with Mean±Std and p-value)
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Attack Effectiveness Analysis (n=6 tickers, excluding META)}
\label{tab:attack_effectiveness}
\begin{tabular}{l|cc|cc|cc|cc}
\toprule
\multirow{2}{*}{\textbf{Attack Type}} & \multicolumn{2}{c|}{\textbf{$\Delta$CR (\%)}} & \multicolumn{2}{c|}{\textbf{$\Delta$MDD (\%)}} & \multicolumn{2}{c|}{\textbf{ASR (\%)}} & \multicolumn{2}{c}{\textbf{DCR (\%)}} \\
 & Mean$\pm$Std & p & Mean$\pm$Std & p & Mean$\pm$Std & p & Mean$\pm$Std & p \\
\midrule
"""

    for attack in ATTACKS_V2:
        if attack not in attack_stats:
            continue

        name = ATTACK_DISPLAY_NAMES[attack]
        s = attack_stats[attack]

        cr_str = f"{s['cr_diff_mean']:.1f}$\\pm${s['cr_diff_std']:.1f}"
        mdd_str = f"{s['mdd_diff_mean']:.1f}$\\pm${s['mdd_diff_std']:.1f}"
        asr_str = f"{s['asr_mean']:.1f}$\\pm${s['asr_std']:.1f}"
        dcr_str = f"{s['dcr_mean']:.1f}$\\pm${s['dcr_std']:.1f}"

        latex += f"{name} & {cr_str} & {format_pvalue(s['cr_p'])} & {mdd_str} & {format_pvalue(s['mdd_p'])} & {asr_str} & {format_pvalue(s['asr_p'])} & {dcr_str} & {format_pvalue(s['dcr_p'])} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex


def generate_combined_attacks_table():
    """
    Generate Table for All Attacks (Single + Combined)
    Shows ASR and DCR for each attack, excluding META
    No p-values, just mean values
    """

    # Tickers to use (exclude META)
    TICKERS_V2 = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "TSLA"]

    # Single attacks (exclude Jailbreaking Authority)
    SINGLE_ATTACKS_V2 = [
        "attack_01_data_poisoning",
        "attack_02_indirect_injection",
        "attack_03_persuasive_bullish",
        "attack_04_objective_hijacking",
        "attack_05_jailbreaking_hypothetical",
    ]

    COMBINED_ATTACKS_V2 = [
        "combined_c1_insider_collusion",
        "combined_c2_news_risk_bypass",
        "combined_c3_data_bias",
        "combined_c4_bias_risk_ignore",
        "combined_c5_full_chain",
        "combined_c6_analysis_trade",
    ]

    DISPLAY_NAMES = {
        "attack_01_data_poisoning": "Data Poisoning",
        "attack_02_indirect_injection": "Indirect Injection",
        "attack_03_persuasive_bullish": "Persuasive Adversary",
        "attack_04_objective_hijacking": "Objective Hijacking",
        "attack_05_jailbreaking_hypothetical": "Jailbreaking",
        "combined_c1_insider_collusion": "C1: Insider Collusion",
        "combined_c2_news_risk_bypass": "C2: News Risk Bypass",
        "combined_c3_data_bias": "C3: Data Bias",
        "combined_c4_bias_risk_ignore": "C4: Bias Risk Ignore",
        "combined_c5_full_chain": "C5: Full Chain",
        "combined_c6_analysis_trade": "C6: Analysis Trade",
    }

    # Calculate statistics for all attacks
    attack_stats = {}

    for attack in SINGLE_ATTACKS_V2 + COMBINED_ATTACKS_V2:
        per_ticker_asr_dcr = calculate_asr_dcr_per_ticker(attack, TICKERS_V2)

        if per_ticker_asr_dcr:
            asr_values = [per_ticker_asr_dcr[t]['ASR'] for t in TICKERS_V2 if t in per_ticker_asr_dcr]
            dcr_values = [per_ticker_asr_dcr[t]['DCR'] for t in TICKERS_V2 if t in per_ticker_asr_dcr]

            asr_mean = np.mean(asr_values)
            dcr_mean = np.mean(dcr_values)

            attack_stats[attack] = {
                'asr_mean': asr_mean,
                'dcr_mean': dcr_mean,
            }

    # Generate LaTeX
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Attack Effectiveness: ASR and DCR (n=6 tickers, excluding META)}
\label{tab:all_attacks}
\begin{tabular}{l|cc}
\toprule
\textbf{Attack Type} & \textbf{ASR (\%)} & \textbf{DCR (\%)} \\
\midrule
\multicolumn{3}{l}{\textit{Single Attacks}} \\
\midrule
"""

    # Single attacks
    for attack in SINGLE_ATTACKS_V2:
        if attack not in attack_stats:
            continue
        name = DISPLAY_NAMES[attack]
        s = attack_stats[attack]
        latex += f"{name} & {s['asr_mean']:.1f} & {s['dcr_mean']:.1f} \\\\\n"

    latex += r"""\midrule
\multicolumn{3}{l}{\textit{Combined Attacks}} \\
\midrule
"""

    # Combined attacks
    for attack in COMBINED_ATTACKS_V2:
        if attack not in attack_stats:
            continue
        name = DISPLAY_NAMES[attack]
        s = attack_stats[attack]
        latex += f"{name} & {s['asr_mean']:.1f} & {s['dcr_mean']:.1f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex


def generate_table3_statistical_latex(stat_results):
    """Generate Table 3: Statistical Significance Test Results with DCR and ASR statistics (Legacy)"""

    # Calculate per-ticker ASR/DCR and compute statistics
    attack_asr_dcr_stats = {}
    for attack in SINGLE_ATTACKS:
        per_ticker = calculate_asr_dcr_per_ticker(attack)
        if per_ticker and len(per_ticker) >= 3:
            asr_values = [per_ticker[t]['ASR'] for t in TICKERS if t in per_ticker]
            dcr_values = [per_ticker[t]['DCR'] for t in TICKERS if t in per_ticker]

            mean_asr = np.mean(asr_values)
            mean_dcr = np.mean(dcr_values)

            # Wilcoxon test: is DCR significantly different from ASR?
            try:
                stat, p_asr_dcr = stats.wilcoxon(asr_values, dcr_values)
            except:
                p_asr_dcr = 1.0

            attack_asr_dcr_stats[attack] = {
                'mean_asr': mean_asr,
                'mean_dcr': mean_dcr,
                'p_asr_vs_dcr': p_asr_dcr,
            }

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Statistical Significance: Normal vs Attack Performance (n=7 tickers)}
\label{tab:statistical_tests}
\begin{tabular}{l|cc|cc|ccc}
\toprule
\multirow{2}{*}{\textbf{Attack Type}} & \multicolumn{2}{c|}{\textbf{CR (\%)}} & \multicolumn{2}{c|}{\textbf{MDD (\%)}} & \multicolumn{3}{c}{\textbf{Attack Metrics (\%)}} \\
 & \textbf{$\Delta$Mean} & \textbf{p-value} & \textbf{$\Delta$Mean} & \textbf{p-value} & \textbf{ASR} & \textbf{DCR} & \textbf{p-value} \\
\midrule
"""

    def format_pvalue(p):
        """Format p-value with significance stars"""
        if p < 0.001:
            return f"{p:.3f}***"
        elif p < 0.01:
            return f"{p:.3f}**"
        elif p < 0.05:
            return f"{p:.3f}*"
        else:
            return f"{p:.3f}"

    # Single attacks
    for attack in SINGLE_ATTACKS:
        name = ATTACK_DISPLAY_NAMES[attack]
        asr_dcr = attack_asr_dcr_stats.get(attack, {'mean_asr': 0, 'mean_dcr': 0, 'p_asr_vs_dcr': 1.0})
        if attack in stat_results:
            res = stat_results[attack]
            cr_diff = res['CR']['mean_diff']
            cr_p = res['CR']['p_wilcoxon']
            mdd_diff = res['MDD']['mean_diff']
            mdd_p = res['MDD']['p_wilcoxon']
            latex += f"{name} & {cr_diff:+.2f} & {format_pvalue(cr_p)} & {mdd_diff:+.2f} & {format_pvalue(mdd_p)} & {asr_dcr['mean_asr']:.1f} & {asr_dcr['mean_dcr']:.1f} & {format_pvalue(asr_dcr['p_asr_vs_dcr'])} \\\\\n"
        else:
            latex += f"{name} & - & - & - & - & {asr_dcr['mean_asr']:.1f} & {asr_dcr['mean_dcr']:.1f} & {format_pvalue(asr_dcr['p_asr_vs_dcr'])} \\\\\n"

    latex += r"""\bottomrule
\multicolumn{8}{l}{\footnotesize Note: $\Delta$Mean = Normal - Attack. CR/MDD p-value: Wilcoxon test (Normal vs Attack).} \\
\multicolumn{8}{l}{\footnotesize ASR/DCR p-value: Wilcoxon test (ASR vs DCR). Significance: * p<0.05, ** p<0.01, *** p<0.001} \\
\end{tabular}
\end{table}
"""

    return latex


def generate_table2_latex():
    """Generate Table 2: Attack Success Metrics"""

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Attack Vulnerability Analysis: Decision Change Rate and Attack Success Rate}
\label{tab:attack_metrics}
\begin{tabular}{l|rr}
\toprule
\textbf{Attack Type} & \textbf{DCR (\%)} & \textbf{ASR (\%)} \\
\midrule
\multicolumn{3}{l}{\textit{Single Attacks}} \\
\midrule
"""

    # Single attacks
    for attack in SINGLE_ATTACKS:
        summary = load_attack_summary(attack)
        metrics = summary.get('metrics', {})
        dcr = metrics.get('decision_change_rate', 0)
        asr = metrics.get('attack_success_rate', 0)
        name = ATTACK_DISPLAY_NAMES[attack]
        latex += f"{name} & {dcr:.1f} & {asr:.1f} \\\\\n"

    latex += r"""\midrule
\multicolumn{3}{l}{\textit{Combined Attacks}} \\
\midrule
"""

    # Combined attacks
    for attack in COMBINED_ATTACKS:
        summary = load_attack_summary(attack)
        metrics = summary.get('metrics', {})
        dcr = metrics.get('decision_change_rate', 0)
        asr = metrics.get('attack_success_rate', 0)
        name = ATTACK_DISPLAY_NAMES[attack]
        latex += f"{name} & {dcr:.1f} & {asr:.1f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex


def main():
    print("=" * 60)
    print("Generating Experiment Tables")
    print("=" * 60)

    # Load normal performance per ticker for statistical tests
    normal_df = load_normal_performance()
    normal_per_ticker = {}
    for _, row in normal_df.iterrows():
        ticker = row['ticker']
        normal_per_ticker[ticker] = {
            'CR': row['CR%'],
            'AR': row['ARR%'],
            'SR': row['SR'],
            'MDD': row['MDD%'],
        }

    # Calculate attack performance for all attacks (with per-ticker data)
    all_attacks = SINGLE_ATTACKS + COMBINED_ATTACKS
    attack_performance_all = {}

    print("\nCalculating attack performance for statistical tests...")
    for attack in all_attacks:
        print(f"  Processing {attack}...")
        perf = calculate_attack_performance(attack)
        if perf:
            attack_performance_all[attack] = perf

    # Generate Table 1 (Average)
    print("\n[Table 1: Financial Performance Metrics (Average)]")
    table1 = generate_table1_latex()
    print(table1)

    # Generate Table 2
    print("\n[Table 2: Attack Success Metrics]")
    table2 = generate_table2_latex()
    print(table2)

    # Generate Table 3 (Statistical Significance) - Legacy version
    print("\n[Table 3: Statistical Significance Tests (Legacy, n=7)]")
    stat_results = calculate_statistical_tests(normal_per_ticker, attack_performance_all)
    table3 = generate_table3_statistical_latex(stat_results)
    print(table3)

    # Generate Table 3 v2 (Excluding META, Excluding Jailbreaking Authority)
    print("\n[Table 3 v2: Statistical Significance Tests (n=6, excl. META, excl. Jailbreaking Auth)]")
    table3_v2 = generate_table3_statistical_latex_v2(normal_per_ticker, attack_performance_all)
    print(table3_v2)

    # Generate Combined Attacks Table
    print("\n[Table: Combined Attacks (C1~C6, n=6, excl. META)]")
    table_combined = generate_combined_attacks_table()
    print(table_combined)

    # Print statistical results summary
    print("\n" + "=" * 60)
    print("Statistical Test Results Summary")
    print("=" * 60)
    print(f"{'Attack':<35} {'CR p-value':<12} {'MDD p-value':<12} {'Significant?'}")
    print("-" * 75)
    for attack in all_attacks:
        if attack in stat_results:
            res = stat_results[attack]
            cr_p = res['CR']['p_wilcoxon']
            mdd_p = res['MDD']['p_wilcoxon']
            sig = "Yes" if cr_p < 0.05 or mdd_p < 0.05 else "No"
            name = ATTACK_DISPLAY_NAMES[attack]
            print(f"{name:<35} {cr_p:<12.4f} {mdd_p:<12.4f} {sig}")

    # Generate Per-Ticker Tables
    print("\n[Per-Ticker Tables]")
    per_ticker_tables = generate_per_ticker_tables_latex()

    # Save to files
    output_dir = BASE_DIR / "latex_tables"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "table1_financial_performance.tex", 'w') as f:
        f.write(table1)

    with open(output_dir / "table2_attack_metrics.tex", 'w') as f:
        f.write(table2)

    with open(output_dir / "table3_statistical_tests.tex", 'w') as f:
        f.write(table3)

    with open(output_dir / "table3_statistical_tests_v2.tex", 'w') as f:
        f.write(table3_v2)

    with open(output_dir / "table_combined_attacks.tex", 'w') as f:
        f.write(table_combined)

    # Save per-ticker tables
    for ticker, latex in per_ticker_tables.items():
        filename = f"table_financial_performance_{ticker}.tex"
        with open(output_dir / filename, 'w') as f:
            f.write(latex)
        print(f"  Saved: {filename}")

    print(f"\nAll tables saved to: {output_dir}")


if __name__ == '__main__':
    main()
