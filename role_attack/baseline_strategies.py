"""
Baseline Trading Strategies

Implements common baseline strategies for comparison:
- B&H     (Buy & Hold)
- MACD    (Moving Average Convergence Divergence)
- KDJ&RSI (Stochastic Oscillator + RSI)
- ZMR     (Zero-Mean Reversion)
- SMA     (Simple Moving Average Crossover)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import argparse


class BaselineStrategies:
    """Baseline trading strategies implementation."""

    def __init__(self, prices_df: pd.DataFrame):
        """
        Initialize with price data.

        Args:
            prices_df: DataFrame with 'date', 'open', 'high', 'low', 'close', 'volume'
        """
        self.df = prices_df.copy()
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self._calculate_indicators()

    def _calculate_indicators(self):
        """Calculate all technical indicators needed for strategies."""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']

        # SMA
        self.df['sma_5'] = close.rolling(window=5).mean()
        self.df['sma_10'] = close.rolling(window=10).mean()
        self.df['sma_20'] = close.rolling(window=20).mean()
        self.df['sma_50'] = close.rolling(window=50).mean()

        # EMA for MACD
        self.df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        self.df['ema_26'] = close.ewm(span=26, adjust=False).mean()
        self.df['macd'] = self.df['ema_12'] - self.df['ema_26']
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        self.df['macd_hist'] = self.df['macd'] - self.df['macd_signal']

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))

        # KDJ (Stochastic Oscillator)
        low_min = low.rolling(window=9).min()
        high_max = high.rolling(window=9).max()
        self.df['rsv'] = (close - low_min) / (high_max - low_min) * 100
        self.df['k'] = self.df['rsv'].ewm(com=2, adjust=False).mean()
        self.df['d'] = self.df['k'].ewm(com=2, adjust=False).mean()
        self.df['j'] = 3 * self.df['k'] - 2 * self.df['d']

        # ZMR (Zero-Mean Reversion) - Z-score
        self.df['zscore'] = (close - close.rolling(window=20).mean()) / close.rolling(window=20).std()

    def buy_and_hold(self) -> pd.DataFrame:
        """
        Buy & Hold strategy: Buy on first day, hold forever.

        Returns:
            DataFrame with date and decision columns
        """
        decisions = []
        for i, row in self.df.iterrows():
            if i == 0:
                decisions.append({'date': row['date'], 'decision': 'BUY'})
            else:
                decisions.append({'date': row['date'], 'decision': 'HOLD'})
        return pd.DataFrame(decisions)

    def macd_strategy(self) -> pd.DataFrame:
        """
        MACD strategy:
        - BUY when MACD crosses above signal line
        - SELL when MACD crosses below signal line

        Returns:
            DataFrame with date and decision columns
        """
        decisions = []
        position = 0  # 0: no position, 1: long

        for i, row in self.df.iterrows():
            if i == 0 or pd.isna(row['macd']) or pd.isna(row['macd_signal']):
                decisions.append({'date': row['date'], 'decision': 'HOLD'})
                continue

            prev = self.df.iloc[i - 1]
            macd_cross_up = prev['macd'] <= prev['macd_signal'] and row['macd'] > row['macd_signal']
            macd_cross_down = prev['macd'] >= prev['macd_signal'] and row['macd'] < row['macd_signal']

            if macd_cross_up and position == 0:
                decisions.append({'date': row['date'], 'decision': 'BUY'})
                position = 1
            elif macd_cross_down and position == 1:
                decisions.append({'date': row['date'], 'decision': 'SELL'})
                position = 0
            else:
                decisions.append({'date': row['date'], 'decision': 'HOLD'})

        return pd.DataFrame(decisions)

    def kdj_rsi_strategy(self, rsi_oversold=30, rsi_overbought=70) -> pd.DataFrame:
        """
        KDJ + RSI combined strategy:
        - BUY when K crosses above D AND RSI < oversold threshold
        - SELL when K crosses below D AND RSI > overbought threshold

        Args:
            rsi_oversold: RSI threshold for oversold (default: 30)
            rsi_overbought: RSI threshold for overbought (default: 70)

        Returns:
            DataFrame with date and decision columns
        """
        decisions = []
        position = 0

        for i, row in self.df.iterrows():
            if i == 0 or pd.isna(row['k']) or pd.isna(row['d']) or pd.isna(row['rsi']):
                decisions.append({'date': row['date'], 'decision': 'HOLD'})
                continue

            prev = self.df.iloc[i - 1]

            # K crosses above D (bullish) + RSI oversold
            k_cross_up = prev['k'] <= prev['d'] and row['k'] > row['d']
            # K crosses below D (bearish) + RSI overbought
            k_cross_down = prev['k'] >= prev['d'] and row['k'] < row['d']

            if k_cross_up and row['rsi'] < rsi_oversold and position == 0:
                decisions.append({'date': row['date'], 'decision': 'BUY'})
                position = 1
            elif k_cross_down and row['rsi'] > rsi_overbought and position == 1:
                decisions.append({'date': row['date'], 'decision': 'SELL'})
                position = 0
            else:
                decisions.append({'date': row['date'], 'decision': 'HOLD'})

        return pd.DataFrame(decisions)

    def zmr_strategy(self, entry_threshold=-1.5, exit_threshold=0.0) -> pd.DataFrame:
        """
        Zero-Mean Reversion strategy:
        - BUY when z-score < entry_threshold (price is below mean)
        - SELL when z-score > exit_threshold (price reverts to mean)

        Args:
            entry_threshold: Z-score threshold for entry (default: -1.5)
            exit_threshold: Z-score threshold for exit (default: 0.0)

        Returns:
            DataFrame with date and decision columns
        """
        decisions = []
        position = 0

        for i, row in self.df.iterrows():
            if pd.isna(row['zscore']):
                decisions.append({'date': row['date'], 'decision': 'HOLD'})
                continue

            zscore = row['zscore']

            if zscore < entry_threshold and position == 0:
                decisions.append({'date': row['date'], 'decision': 'BUY'})
                position = 1
            elif zscore > exit_threshold and position == 1:
                decisions.append({'date': row['date'], 'decision': 'SELL'})
                position = 0
            else:
                decisions.append({'date': row['date'], 'decision': 'HOLD'})

        return pd.DataFrame(decisions)

    def sma_crossover_strategy(self, short_window=10, long_window=50) -> pd.DataFrame:
        """
        SMA Crossover strategy:
        - BUY when short SMA crosses above long SMA (golden cross)
        - SELL when short SMA crosses below long SMA (death cross)

        Args:
            short_window: Short-term SMA period (default: 10)
            long_window: Long-term SMA period (default: 50)

        Returns:
            DataFrame with date and decision columns
        """
        # Use pre-calculated SMAs or calculate new ones
        short_col = f'sma_{short_window}' if f'sma_{short_window}' in self.df.columns else None
        long_col = f'sma_{long_window}' if f'sma_{long_window}' in self.df.columns else None

        if short_col is None:
            self.df[f'sma_{short_window}'] = self.df['close'].rolling(window=short_window).mean()
            short_col = f'sma_{short_window}'
        if long_col is None:
            self.df[f'sma_{long_window}'] = self.df['close'].rolling(window=long_window).mean()
            long_col = f'sma_{long_window}'

        decisions = []
        position = 0

        for i, row in self.df.iterrows():
            if i == 0 or pd.isna(row[short_col]) or pd.isna(row[long_col]):
                decisions.append({'date': row['date'], 'decision': 'HOLD'})
                continue

            prev = self.df.iloc[i - 1]

            # Golden cross: short crosses above long
            golden_cross = prev[short_col] <= prev[long_col] and row[short_col] > row[long_col]
            # Death cross: short crosses below long
            death_cross = prev[short_col] >= prev[long_col] and row[short_col] < row[long_col]

            if golden_cross and position == 0:
                decisions.append({'date': row['date'], 'decision': 'BUY'})
                position = 1
            elif death_cross and position == 1:
                decisions.append({'date': row['date'], 'decision': 'SELL'})
                position = 0
            else:
                decisions.append({'date': row['date'], 'decision': 'HOLD'})

        return pd.DataFrame(decisions)


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLCV price data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)

    if hist.empty:
        raise ValueError(f"No price data found for {ticker}")

    prices_df = pd.DataFrame({
        'date': hist.index.strftime('%Y-%m-%d'),
        'open': hist['Open'].values,
        'high': hist['High'].values,
        'low': hist['Low'].values,
        'close': hist['Close'].values,
        'volume': hist['Volume'].values
    })

    return prices_df


def calculate_metrics(
    decisions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    initial_capital: float = 100000.0,
    risk_free_rate: float = 0.02,
    position_size: float = 1.0
) -> dict:
    """Calculate trading performance metrics.

    Args:
        decisions_df: DataFrame with 'date' and 'decision' columns
        prices_df: DataFrame with 'date' and 'close' columns
        initial_capital: Starting capital (default: $100,000)
        risk_free_rate: Annual risk-free rate (default: 2%)
        position_size: Fraction of capital to use per trade (default: 1.0 = 100%)
    """
    merged = decisions_df.merge(
        prices_df[['date', 'close']],
        on='date',
        how='inner'
    )
    merged = merged.sort_values('date').reset_index(drop=True)

    if merged.empty:
        return {'CR%': 0.0, 'ARR%': 0.0, 'SR': 0.0, 'MDD%': 0.0, 'total_trades': 0, 'win_rate': 0.0}

    position = 0
    entry_price = 0
    invested_amount = 0
    capital = initial_capital
    equity_curve = [capital]
    daily_returns = []
    trades = []

    for i, row in merged.iterrows():
        decision = row['decision'].upper()
        price = row['close']

        if decision == "BUY" and position == 0:
            position = 1
            entry_price = price
            invested_amount = capital * position_size

        elif decision == "SELL" and position == 1:
            position = 0
            trade_return = (price - entry_price) / entry_price
            profit = invested_amount * trade_return
            capital = capital + profit
            trades.append(trade_return)
            invested_amount = 0

        if position == 1:
            uninvested = capital - invested_amount
            current_position_value = (invested_amount / entry_price) * price
            current_value = uninvested + current_position_value
            equity_curve.append(current_value)
        else:
            equity_curve.append(capital)

        if len(equity_curve) > 1:
            daily_ret = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
            daily_returns.append(daily_ret)

    # Close any open position
    if position == 1:
        final_price = merged.iloc[-1]['close']
        trade_return = (final_price - entry_price) / entry_price
        profit = invested_amount * trade_return
        capital = capital + profit
        trades.append(trade_return)
        equity_curve[-1] = capital

    equity_curve = np.array(equity_curve)
    daily_returns = np.array(daily_returns) if daily_returns else np.array([0.0])

    # CR%
    cr = (equity_curve[-1] - initial_capital) / initial_capital * 100

    # ARR%
    trading_days = len(merged)
    if trading_days > 0:
        total_return = equity_curve[-1] / initial_capital
        arr = (total_return ** (252 / trading_days) - 1) * 100
    else:
        arr = 0.0

    # SR
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        excess_returns = daily_returns - (risk_free_rate / 252)
        sr = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    else:
        sr = 0.0

    # MDD%
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    mdd = np.min(drawdown) * 100

    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0.0

    return {
        'CR%': round(cr, 2),
        'ARR%': round(arr, 2),
        'SR': round(sr, 3),
        'MDD%': round(mdd, 2),
        'total_trades': len(trades),
        'win_rate': round(win_rate, 1)
    }


def run_baseline_comparison(
    ticker: str,
    start_date: str,
    end_date: str,
    agent_results_file: str = None,
    position_size: float = 1.0
) -> pd.DataFrame:
    """
    Run all baseline strategies and compare with agent results.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        agent_results_file: Optional path to agent results CSV
        position_size: Fraction of capital to use per trade (default: 1.0 = 100%)

    Returns:
        DataFrame with metrics for all strategies
    """
    print(f"\nFetching price data for {ticker}...")
    prices_df = get_price_data(ticker, start_date, end_date)
    print(f"  Got {len(prices_df)} days of data")

    # Initialize strategies
    strategies = BaselineStrategies(prices_df)

    # Run all baseline strategies
    baseline_configs = {
        'B&H': strategies.buy_and_hold(),
        'MACD': strategies.macd_strategy(),
        'KDJ&RSI': strategies.kdj_rsi_strategy(),
        'ZMR': strategies.zmr_strategy(),
        'SMA': strategies.sma_crossover_strategy(),
    }

    results = []

    for strategy_name, decisions_df in baseline_configs.items():
        print(f"\nEvaluating {strategy_name}...")
        metrics = calculate_metrics(decisions_df, prices_df, position_size=position_size)
        metrics['strategy'] = strategy_name
        metrics['ticker'] = ticker
        results.append(metrics)

        print(f"  CR%:  {metrics['CR%']:>8.2f}%")
        print(f"  ARR%: {metrics['ARR%']:>8.2f}%")
        print(f"  SR:   {metrics['SR']:>8.3f}")
        print(f"  MDD%: {metrics['MDD%']:>8.2f}%")

    # Load and evaluate agent results if provided
    if agent_results_file and os.path.exists(agent_results_file):
        print(f"\nLoading agent results from {agent_results_file}...")
        agent_df = pd.read_csv(agent_results_file)

        if 'status' in agent_df.columns:
            agent_df = agent_df[agent_df['status'] == 'success']

        if not agent_df.empty and 'date' in agent_df.columns and 'decision' in agent_df.columns:
            agent_decisions = agent_df[['date', 'decision']].copy()
            metrics = calculate_metrics(agent_decisions, prices_df, position_size=position_size)
            metrics['strategy'] = 'TradingAgent'
            metrics['ticker'] = ticker
            results.append(metrics)

            print(f"\nTradingAgent Results:")
            print(f"  CR%:  {metrics['CR%']:>8.2f}%")
            print(f"  ARR%: {metrics['ARR%']:>8.2f}%")
            print(f"  SR:   {metrics['SR']:>8.3f}")
            print(f"  MDD%: {metrics['MDD%']:>8.2f}%")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Run baseline trading strategies comparison')
    parser.add_argument('--ticker', '-t', type=str, required=True,
                        help='Stock ticker symbol')
    parser.add_argument('--start', '-s', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--agent-results', '-a', type=str, default=None,
                        help='Path to agent results CSV file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file for comparison results')
    parser.add_argument('--position-size', '-p', type=float, default=1.0,
                        help='Position size as fraction of capital (default: 1.0 = 100%%)')

    args = parser.parse_args()

    print("=" * 60)
    print("Baseline Trading Strategies Comparison")
    print("=" * 60)
    print(f"Ticker: {args.ticker}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Position size: {args.position_size*100:.0f}%")

    # Run comparison
    results_df = run_baseline_comparison(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        agent_results_file=args.agent_results,
        position_size=args.position_size
    )

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    # Reorder columns
    cols = ['strategy', 'ticker', 'CR%', 'ARR%', 'SR', 'MDD%', 'win_rate', 'total_trades']
    results_df = results_df[[c for c in cols if c in results_df.columns]]
    print(results_df.to_string(index=False))

    # Find best strategy for each metric
    print("\n" + "-" * 60)
    print("BEST PERFORMERS")
    print("-" * 60)
    print(f"  Best CR%:  {results_df.loc[results_df['CR%'].idxmax(), 'strategy']} ({results_df['CR%'].max():.2f}%)")
    print(f"  Best ARR%: {results_df.loc[results_df['ARR%'].idxmax(), 'strategy']} ({results_df['ARR%'].max():.2f}%)")
    print(f"  Best SR:   {results_df.loc[results_df['SR'].idxmax(), 'strategy']} ({results_df['SR'].max():.3f})")
    print(f"  Best MDD%: {results_df.loc[results_df['MDD%'].idxmax(), 'strategy']} ({results_df['MDD%'].max():.2f}%)")

    # Save results
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")

    print("=" * 60)


def batch_comparison(
    tickers: list,
    start_date: str,
    end_date: str,
    results_dir: str = None,
    output_file: str = None,
    position_size: float = 1.0
):
    """
    Run baseline comparison for multiple tickers.

    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        results_dir: Directory containing agent results (optional)
        output_file: Output CSV file
        position_size: Fraction of capital to use per trade (default: 1.0 = 100%)
    """
    all_results = []

    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Processing {ticker}")
        print("=" * 60)

        agent_file = None
        if results_dir:
            agent_file = os.path.join(results_dir, f"{ticker}_results.csv")
            if not os.path.exists(agent_file):
                agent_file = None

        try:
            results_df = run_baseline_comparison(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                agent_results_file=agent_file,
                position_size=position_size
            )
            all_results.append(results_df)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Summary by strategy
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY (Average across all tickers)")
        print("=" * 60)

        summary = combined_df.groupby('strategy').agg({
            'CR%': 'mean',
            'ARR%': 'mean',
            'SR': 'mean',
            'MDD%': 'mean',
            'win_rate': 'mean',
            'total_trades': 'sum'
        }).round(2)

        print(summary.to_string())

        if output_file:
            combined_df.to_csv(output_file, index=False)
            print(f"\nDetailed results saved to: {output_file}")

        return combined_df

    return pd.DataFrame()


if __name__ == "__main__":
    main()
