"""
Trading Performance Metrics Calculator

Calculates key performance metrics from trading agent results:
- CR%  (Cumulative Return)
- ARR% (Annualized Rate of Return)
- SR   (Sharpe Ratio)
- MDD% (Maximum Drawdown)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import argparse


def parse_decision(decision_text: str) -> str:
    """Parse decision from text to BUY/SELL/HOLD."""
    if pd.isna(decision_text):
        return "UNKNOWN"

    text = str(decision_text).upper()

    # Look for explicit decision keywords
    if "BUY" in text:
        return "BUY"
    elif "SELL" in text:
        return "SELL"
    elif "HOLD" in text:
        return "HOLD"

    return "UNKNOWN"


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch price data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)

    if hist.empty:
        raise ValueError(f"No price data found for {ticker}")

    prices_df = pd.DataFrame({
        'date': hist.index.strftime('%Y-%m-%d'),
        'close': hist['Close'].values
    })

    return prices_df


def calculate_metrics(
    decisions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    initial_capital: float = 100000.0,
    risk_free_rate: float = 0.02,
    position_size: float = 1.0
) -> dict:
    """
    Calculate trading performance metrics.

    Args:
        decisions_df: DataFrame with 'date' and 'decision' columns
        prices_df: DataFrame with 'date' and 'close' columns
        initial_capital: Starting capital (default: $100,000)
        risk_free_rate: Annual risk-free rate (default: 2%)
        position_size: Fraction of capital to use per trade (default: 1.0 = 100%)
                       e.g., 0.1 = 10% of capital per trade

    Returns:
        Dictionary with CR%, ARR%, SR, MDD%
    """
    # Merge decisions with prices
    merged = decisions_df.merge(prices_df, on='date', how='inner')
    merged = merged.sort_values('date').reset_index(drop=True)

    if merged.empty:
        return {
            'CR%': 0.0,
            'ARR%': 0.0,
            'SR': 0.0,
            'MDD%': 0.0,
            'total_trades': 0,
            'win_rate': 0.0
        }

    # Simulate trading
    position = 0  # 0: no position, 1: long
    entry_price = 0
    invested_amount = 0  # Amount invested in current position
    capital = initial_capital
    equity_curve = [capital]
    daily_returns = []
    trades = []

    for i, row in merged.iterrows():
        decision = parse_decision(row['decision'])
        price = row['close']

        if decision == "BUY" and position == 0:
            # Enter long position with position_size fraction of capital
            position = 1
            entry_price = price
            invested_amount = capital * position_size
            shares = invested_amount / price

        elif decision == "SELL" and position == 1:
            # Exit long position
            position = 0
            exit_price = price
            trade_return = (exit_price - entry_price) / entry_price
            # Only the invested portion gains/loses
            profit = invested_amount * trade_return
            capital = capital + profit
            trades.append(trade_return)
            invested_amount = 0

        # Track equity (mark-to-market)
        if position == 1:
            # Current value = uninvested cash + current value of position
            uninvested = capital - invested_amount
            current_position_value = (invested_amount / entry_price) * price
            current_value = uninvested + current_position_value
            equity_curve.append(current_value)
        else:
            equity_curve.append(capital)

        # Calculate daily return
        if len(equity_curve) > 1:
            daily_ret = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
            daily_returns.append(daily_ret)

    # Close any open position at the end
    if position == 1:
        final_price = merged.iloc[-1]['close']
        trade_return = (final_price - entry_price) / entry_price
        profit = invested_amount * trade_return
        capital = capital + profit
        trades.append(trade_return)
        equity_curve[-1] = capital

    equity_curve = np.array(equity_curve)
    daily_returns = np.array(daily_returns) if daily_returns else np.array([0.0])

    # Calculate metrics

    # 1. CR% (Cumulative Return)
    cr = (equity_curve[-1] - initial_capital) / initial_capital * 100

    # 2. ARR% (Annualized Rate of Return)
    trading_days = len(merged)
    if trading_days > 0:
        total_return = equity_curve[-1] / initial_capital
        arr = (total_return ** (252 / trading_days) - 1) * 100
    else:
        arr = 0.0

    # 3. SR (Sharpe Ratio)
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        excess_returns = daily_returns - (risk_free_rate / 252)
        sr = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    else:
        sr = 0.0

    # 4. MDD% (Maximum Drawdown)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    mdd = np.min(drawdown) * 100

    # Win rate
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0.0

    return {
        'CR%': round(cr, 2),
        'ARR%': round(arr, 2),
        'SR': round(sr, 3),
        'MDD%': round(mdd, 2),
        'total_trades': len(trades),
        'win_rate': round(win_rate, 1),
        'final_capital': round(equity_curve[-1], 2),
        'trading_days': trading_days
    }


def evaluate_results(results_dir: str, ticker: str = None, position_size: float = 1.0) -> pd.DataFrame:
    """
    Evaluate trading results from CSV files.

    Args:
        results_dir: Directory containing result CSV files
        ticker: Specific ticker to evaluate (None for all)
        position_size: Fraction of capital to use per trade (default: 1.0 = 100%)

    Returns:
        DataFrame with metrics for each ticker
    """
    all_metrics = []

    # Find result files
    if ticker:
        files = [f"{ticker}_results.csv"]
    else:
        files = [f for f in os.listdir(results_dir) if f.endswith('_results.csv') and not f.startswith('all_')]

    for file in files:
        filepath = os.path.join(results_dir, file)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        ticker_name = file.replace('_results.csv', '')
        print(f"\nProcessing {ticker_name}...")

        # Load decisions
        df = pd.read_csv(filepath)

        if 'date' not in df.columns or 'decision' not in df.columns:
            print(f"  Skipping {ticker_name}: missing required columns")
            continue

        # Filter successful decisions only
        if 'status' in df.columns:
            df = df[df['status'] == 'success']

        if df.empty:
            print(f"  Skipping {ticker_name}: no successful decisions")
            continue

        # Get date range from data
        dates = sorted(df['date'].unique())
        start_date = dates[0]
        end_date = dates[-1]

        print(f"  Period: {start_date} to {end_date}")
        print(f"  Decisions: {len(df)}")

        # Fetch prices
        try:
            prices_df = get_price_data(ticker_name, start_date, end_date)
            print(f"  Price data points: {len(prices_df)}")
        except Exception as e:
            print(f"  Error fetching prices: {e}")
            continue

        # Prepare decisions DataFrame
        decisions_df = df[['date', 'decision']].copy()

        # Calculate metrics
        metrics = calculate_metrics(decisions_df, prices_df, position_size=position_size)
        metrics['ticker'] = ticker_name
        metrics['position_size'] = f"{position_size*100:.0f}%"
        metrics['start_date'] = start_date
        metrics['end_date'] = end_date

        all_metrics.append(metrics)

        # Print results
        print(f"  CR%:  {metrics['CR%']:>8.2f}%")
        print(f"  ARR%: {metrics['ARR%']:>8.2f}%")
        print(f"  SR:   {metrics['SR']:>8.3f}")
        print(f"  MDD%: {metrics['MDD%']:>8.2f}%")
        print(f"  Win Rate: {metrics['win_rate']:.1f}% ({metrics['total_trades']} trades)")

    if not all_metrics:
        print("\nNo results to evaluate.")
        return pd.DataFrame()

    # Create summary DataFrame
    results_df = pd.DataFrame(all_metrics)
    cols = ['ticker', 'start_date', 'end_date', 'CR%', 'ARR%', 'SR', 'MDD%', 'win_rate', 'total_trades', 'trading_days', 'final_capital']
    results_df = results_df[[c for c in cols if c in results_df.columns]]

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Calculate trading performance metrics')
    parser.add_argument('--results-dir', '-d', type=str, required=True,
                        help='Directory containing result CSV files')
    parser.add_argument('--ticker', '-t', type=str, default=None,
                        help='Specific ticker to evaluate (default: all)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file for metrics summary')
    parser.add_argument('--position-size', '-p', type=float, default=1.0,
                        help='Position size as fraction of capital (default: 1.0 = 100%%, e.g., 0.1 = 10%%)')

    args = parser.parse_args()

    print("=" * 60)
    print("Trading Performance Metrics Calculator")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Position size: {args.position_size*100:.0f}%")

    # Evaluate
    results_df = evaluate_results(args.results_dir, args.ticker, args.position_size)

    if results_df.empty:
        return

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Calculate averages
    print("\n" + "-" * 60)
    print("AVERAGE METRICS")
    print("-" * 60)
    print(f"  Avg CR%:  {results_df['CR%'].mean():>8.2f}%")
    print(f"  Avg ARR%: {results_df['ARR%'].mean():>8.2f}%")
    print(f"  Avg SR:   {results_df['SR'].mean():>8.3f}")
    print(f"  Avg MDD%: {results_df['MDD%'].mean():>8.2f}%")
    print(f"  Avg Win Rate: {results_df['win_rate'].mean():.1f}%")

    # Save to file
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    else:
        output_file = os.path.join(args.results_dir, 'performance_metrics.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    print("=" * 60)


if __name__ == "__main__":
    main()
