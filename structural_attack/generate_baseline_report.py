"""
Generate Monthly Baseline Performance Report for 2025

Calculates CR%, ARR%, SR, MDD% for all baseline strategies
across all tickers, monthly for 2025.
Exports results to Excel.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import warnings
from baseline_strategies import BaselineStrategies, calculate_metrics

warnings.filterwarnings('ignore')


# Configuration
TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
STRATEGIES = ["B&H", "MACD", "KDJ&RSI", "ZMR", "SMA"]
YEAR = 2025

# Monthly periods for 2025
MONTHS = [
    ("2025-01", "2025-01-01", "2025-01-31"),
    ("2025-02", "2025-02-01", "2025-02-28"),
    ("2025-03", "2025-03-01", "2025-03-31"),
    ("2025-04", "2025-04-01", "2025-04-30"),
    ("2025-05", "2025-05-01", "2025-05-31"),
    ("2025-06", "2025-06-01", "2025-06-30"),
    ("2025-07", "2025-07-01", "2025-07-31"),
    ("2025-08", "2025-08-01", "2025-08-31"),
    ("2025-09", "2025-09-01", "2025-09-30"),
    ("2025-10", "2025-10-01", "2025-10-31"),
    ("2025-11", "2025-11-01", "2025-11-30"),
    ("2025-12", "2025-12-01", "2025-12-31"),
]


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLCV price data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return None

        prices_df = pd.DataFrame({
            'date': hist.index.strftime('%Y-%m-%d'),
            'open': hist['Open'].values,
            'high': hist['High'].values,
            'low': hist['Low'].values,
            'close': hist['Close'].values,
            'volume': hist['Volume'].values
        })

        return prices_df
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


def run_strategies(prices_df: pd.DataFrame) -> dict:
    """Run all baseline strategies and return decisions."""
    strategies = BaselineStrategies(prices_df)

    return {
        'B&H': strategies.buy_and_hold(),
        'MACD': strategies.macd_strategy(),
        'KDJ&RSI': strategies.kdj_rsi_strategy(),
        'ZMR': strategies.zmr_strategy(),
        'SMA': strategies.sma_crossover_strategy(),
    }


def calculate_all_metrics(ticker: str, month_name: str, start_date: str, end_date: str) -> list:
    """Calculate metrics for all strategies for a given ticker and period."""
    results = []

    # Fetch price data
    prices_df = get_price_data(ticker, start_date, end_date)

    if prices_df is None or len(prices_df) < 5:
        print(f"  Skipping {ticker} {month_name}: insufficient data")
        return results

    # Run all strategies
    try:
        strategy_decisions = run_strategies(prices_df)
    except Exception as e:
        print(f"  Error running strategies for {ticker} {month_name}: {e}")
        return results

    # Calculate metrics for each strategy
    for strategy_name, decisions_df in strategy_decisions.items():
        try:
            metrics = calculate_metrics(decisions_df, prices_df)
            results.append({
                'ticker': ticker,
                'month': month_name,
                'strategy': strategy_name,
                'CR%': metrics['CR%'],
                'ARR%': metrics['ARR%'],
                'SR': metrics['SR'],
                'MDD%': metrics['MDD%'],
                'win_rate': metrics['win_rate'],
                'total_trades': metrics['total_trades'],
                'start_date': start_date,
                'end_date': end_date
            })
        except Exception as e:
            print(f"  Error calculating metrics for {ticker} {month_name} {strategy_name}: {e}")

    return results


def generate_monthly_report():
    """Generate complete monthly report for all tickers and strategies."""
    all_results = []

    total_tasks = len(TICKERS) * len(MONTHS)
    current_task = 0

    print("=" * 70)
    print("Generating Monthly Baseline Performance Report for 2025")
    print("=" * 70)
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Strategies: {', '.join(STRATEGIES)}")
    print(f"Period: {YEAR}")
    print("=" * 70)

    for ticker in TICKERS:
        print(f"\n[{ticker}]")

        for month_name, start_date, end_date in MONTHS:
            current_task += 1
            progress = current_task / total_tasks * 100
            print(f"  {month_name} ({progress:.0f}%)...", end=" ")

            results = calculate_all_metrics(ticker, month_name, start_date, end_date)
            all_results.extend(results)

            if results:
                print(f"OK ({len(results)} strategies)")
            else:
                print("SKIP")

    return pd.DataFrame(all_results)


def create_excel_report(df: pd.DataFrame, output_file: str):
    """Create Excel report with multiple sheets."""

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

        # Sheet 1: All Raw Data
        df.to_excel(writer, sheet_name='Raw Data', index=False)

        # Sheet 2: CR% by Ticker x Month (pivot table for each strategy)
        for strategy in STRATEGIES:
            strategy_df = df[df['strategy'] == strategy]
            if not strategy_df.empty:
                pivot_cr = strategy_df.pivot_table(
                    values='CR%',
                    index='ticker',
                    columns='month',
                    aggfunc='first'
                )
                # Reorder columns chronologically
                months_order = [m[0] for m in MONTHS if m[0] in pivot_cr.columns]
                pivot_cr = pivot_cr[months_order]
                # Add average column
                pivot_cr['Average'] = pivot_cr.mean(axis=1)
                pivot_cr.loc['Average'] = pivot_cr.mean()
                pivot_cr.to_excel(writer, sheet_name=f'{strategy}_CR%')

        # Sheet 3: SR by Ticker x Month (pivot table for each strategy)
        for strategy in STRATEGIES:
            strategy_df = df[df['strategy'] == strategy]
            if not strategy_df.empty:
                pivot_sr = strategy_df.pivot_table(
                    values='SR',
                    index='ticker',
                    columns='month',
                    aggfunc='first'
                )
                months_order = [m[0] for m in MONTHS if m[0] in pivot_sr.columns]
                pivot_sr = pivot_sr[months_order]
                pivot_sr['Average'] = pivot_sr.mean(axis=1)
                pivot_sr.loc['Average'] = pivot_sr.mean()
                pivot_sr.to_excel(writer, sheet_name=f'{strategy}_SR')

        # Sheet 4: MDD% by Ticker x Month (pivot table for each strategy)
        for strategy in STRATEGIES:
            strategy_df = df[df['strategy'] == strategy]
            if not strategy_df.empty:
                pivot_mdd = strategy_df.pivot_table(
                    values='MDD%',
                    index='ticker',
                    columns='month',
                    aggfunc='first'
                )
                months_order = [m[0] for m in MONTHS if m[0] in pivot_mdd.columns]
                pivot_mdd = pivot_mdd[months_order]
                pivot_mdd['Average'] = pivot_mdd.mean(axis=1)
                pivot_mdd.loc['Average'] = pivot_mdd.mean()
                pivot_mdd.to_excel(writer, sheet_name=f'{strategy}_MDD%')

        # Sheet 5: Summary - Average metrics by strategy
        summary_df = df.groupby('strategy').agg({
            'CR%': 'mean',
            'ARR%': 'mean',
            'SR': 'mean',
            'MDD%': 'mean',
            'win_rate': 'mean',
            'total_trades': 'sum'
        }).round(2)
        summary_df.to_excel(writer, sheet_name='Strategy Summary')

        # Sheet 6: Summary - Average metrics by ticker
        ticker_summary = df.groupby('ticker').agg({
            'CR%': 'mean',
            'ARR%': 'mean',
            'SR': 'mean',
            'MDD%': 'mean',
            'win_rate': 'mean',
            'total_trades': 'sum'
        }).round(2)
        ticker_summary.to_excel(writer, sheet_name='Ticker Summary')

        # Sheet 7: Strategy Comparison Table
        comparison_data = []
        for strategy in STRATEGIES:
            strategy_df = df[df['strategy'] == strategy]
            if not strategy_df.empty:
                comparison_data.append({
                    'Strategy': strategy,
                    'Avg CR%': round(strategy_df['CR%'].mean(), 2),
                    'Avg ARR%': round(strategy_df['ARR%'].mean(), 2),
                    'Avg SR': round(strategy_df['SR'].mean(), 3),
                    'Avg MDD%': round(strategy_df['MDD%'].mean(), 2),
                    'Avg Win Rate': round(strategy_df['win_rate'].mean(), 1),
                    'Total Trades': strategy_df['total_trades'].sum()
                })
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)

    print(f"\nExcel report saved to: {output_file}")


def print_summary(df: pd.DataFrame):
    """Print summary statistics to console."""
    print("\n" + "=" * 70)
    print("SUMMARY BY STRATEGY")
    print("=" * 70)

    for strategy in STRATEGIES:
        strategy_df = df[df['strategy'] == strategy]
        if not strategy_df.empty:
            print(f"\n{strategy}:")
            print(f"  Avg CR%:     {strategy_df['CR%'].mean():>8.2f}%")
            print(f"  Avg ARR%:    {strategy_df['ARR%'].mean():>8.2f}%")
            print(f"  Avg SR:      {strategy_df['SR'].mean():>8.3f}")
            print(f"  Avg MDD%:    {strategy_df['MDD%'].mean():>8.2f}%")
            print(f"  Avg Win Rate:{strategy_df['win_rate'].mean():>8.1f}%")

    print("\n" + "=" * 70)
    print("SUMMARY BY TICKER")
    print("=" * 70)

    for ticker in TICKERS:
        ticker_df = df[df['ticker'] == ticker]
        if not ticker_df.empty:
            print(f"\n{ticker}:")
            print(f"  Avg CR%:     {ticker_df['CR%'].mean():>8.2f}%")
            print(f"  Avg SR:      {ticker_df['SR'].mean():>8.3f}")
            print(f"  Avg MDD%:    {ticker_df['MDD%'].mean():>8.2f}%")


def main():
    # Generate report
    results_df = generate_monthly_report()

    if results_df.empty:
        print("\nNo results generated. Check if price data is available.")
        return

    # Print summary
    print_summary(results_df)

    # Save to Excel
    output_dir = "baseline_reports"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = os.path.join(output_dir, f"baseline_performance_2025_{timestamp}.xlsx")
    csv_file = os.path.join(output_dir, f"baseline_performance_2025_{timestamp}.csv")

    # Save raw data as CSV
    results_df.to_csv(csv_file, index=False)
    print(f"\nCSV saved to: {csv_file}")

    # Create Excel report with multiple sheets
    create_excel_report(results_df, excel_file)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
