"""
Portfolio Simulator for Trading Agent Results

Simulates actual portfolio value changes based on trading decisions.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os


class PortfolioSimulator:
    """Simulates portfolio performance based on trading decisions"""

    def __init__(
        self,
        seed_money: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1% per trade
        slippage_rate: float = 0.0005,   # 0.05% slippage
    ):
        """
        Initialize simulator.

        Args:
            seed_money: Initial investment amount ($)
            commission_rate: Commission per trade (0.001 = 0.1%)
            slippage_rate: Slippage per trade (0.0005 = 0.05%)
        """
        self.seed_money = seed_money
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    def load_decisions(self, csv_path: str) -> pd.DataFrame:
        """Load trading decisions from CSV file."""
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')

    def get_price_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_local: bool = True
    ) -> pd.DataFrame:
        """
        Get price data for simulation.

        Args:
            ticker: Stock ticker
            start_date: Start date (yyyy-mm-dd)
            end_date: End date (yyyy-mm-dd)
            use_local: Use local data if available
        """
        if use_local:
            # Try local data first
            local_path = f"tradingagents/dataflows/local_data/market_data/price_data/{ticker}-YFin-data-2015-01-01-2025-03-25.csv"
            if os.path.exists(local_path):
                df = pd.read_csv(local_path)
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                df = df.rename(columns={'Close': 'close', 'Date': 'date'})
                return df[['date', 'close']].sort_values('date')

        # Fallback to yfinance
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        return pd.DataFrame({
            'date': hist.index.tz_localize(None),
            'close': hist['Close'].values
        })

    def simulate(
        self,
        decisions_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        ticker: str
    ) -> Dict:
        """
        Run portfolio simulation.

        Args:
            decisions_df: DataFrame with 'date', 'decision' columns
            prices_df: DataFrame with 'date', 'close' columns
            ticker: Stock ticker symbol

        Returns:
            Simulation results dictionary
        """
        # Filter decisions for this ticker
        ticker_decisions = decisions_df[decisions_df['ticker'] == ticker].copy()
        ticker_decisions['date'] = pd.to_datetime(ticker_decisions['date']).dt.tz_localize(None)

        # Merge with prices
        prices_df['date'] = pd.to_datetime(prices_df['date']).dt.tz_localize(None)
        merged = ticker_decisions.merge(prices_df, on='date', how='inner')

        if len(merged) == 0:
            return {'error': 'No matching dates between decisions and prices'}

        # Initialize portfolio
        cash = self.seed_money
        shares = 0.0
        position = 'OUT'  # OUT or IN

        # Track history
        history = []
        trades = []

        for _, row in merged.iterrows():
            date = row['date']
            decision = row['decision'].upper()
            price = row['close']

            # Calculate portfolio value before action
            portfolio_value = cash + (shares * price)

            # Execute decision
            if decision == 'BUY' and position == 'OUT':
                # Buy with all cash (minus commission)
                effective_price = price * (1 + self.slippage_rate)
                commission = cash * self.commission_rate
                available_cash = cash - commission
                shares = available_cash / effective_price
                cash = 0
                position = 'IN'

                trades.append({
                    'date': date,
                    'action': 'BUY',
                    'price': price,
                    'effective_price': effective_price,
                    'shares': shares,
                    'commission': commission,
                    'portfolio_value': shares * price
                })

            elif decision == 'SELL' and position == 'IN':
                # Sell all shares
                effective_price = price * (1 - self.slippage_rate)
                gross_proceeds = shares * effective_price
                commission = gross_proceeds * self.commission_rate
                cash = gross_proceeds - commission

                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': price,
                    'effective_price': effective_price,
                    'shares': shares,
                    'commission': commission,
                    'portfolio_value': cash
                })

                shares = 0
                position = 'OUT'

            # Record daily value
            current_value = cash + (shares * price)
            history.append({
                'date': date,
                'decision': decision,
                'price': price,
                'cash': cash,
                'shares': shares,
                'position': position,
                'portfolio_value': current_value
            })

        # Final portfolio value
        final_price = merged.iloc[-1]['close']
        final_value = cash + (shares * final_price)

        # Calculate Buy & Hold for comparison
        first_price = merged.iloc[0]['close']
        buy_hold_shares = self.seed_money / first_price
        buy_hold_final = buy_hold_shares * final_price

        # Calculate metrics
        total_return = (final_value - self.seed_money) / self.seed_money * 100
        buy_hold_return = (buy_hold_final - self.seed_money) / self.seed_money * 100

        # Calculate max drawdown
        history_df = pd.DataFrame(history)
        if len(history_df) > 0:
            history_df['peak'] = history_df['portfolio_value'].cummax()
            history_df['drawdown'] = (history_df['portfolio_value'] - history_df['peak']) / history_df['peak'] * 100
            max_drawdown = history_df['drawdown'].min()
        else:
            max_drawdown = 0

        return {
            'ticker': ticker,
            'seed_money': self.seed_money,
            'final_value': final_value,
            'total_return_pct': total_return,
            'buy_hold_final': buy_hold_final,
            'buy_hold_return_pct': buy_hold_return,
            'outperformance_pct': total_return - buy_hold_return,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(trades),
            'total_commission': sum(t['commission'] for t in trades),
            'start_date': str(merged.iloc[0]['date'].date()),
            'end_date': str(merged.iloc[-1]['date'].date()),
            'trading_days': len(merged),
            'final_position': position,
            'trades': trades,
            'history': history_df.to_dict('records') if len(history_df) > 0 else []
        }

    def simulate_all_tickers(
        self,
        csv_path: str,
        use_local: bool = True
    ) -> Dict[str, Dict]:
        """
        Simulate all tickers in the results CSV.

        Args:
            csv_path: Path to results CSV
            use_local: Use local price data

        Returns:
            Dictionary of results per ticker
        """
        decisions_df = self.load_decisions(csv_path)
        tickers = decisions_df['ticker'].unique()

        # Get date range
        start_date = decisions_df['date'].min().strftime('%Y-%m-%d')
        end_date = (decisions_df['date'].max() + timedelta(days=1)).strftime('%Y-%m-%d')

        results = {}
        for ticker in tickers:
            print(f"Simulating {ticker}...")
            prices_df = self.get_price_data(ticker, start_date, end_date, use_local)
            results[ticker] = self.simulate(decisions_df, prices_df, ticker)

        return results

    def generate_report(self, results: Dict[str, Dict]) -> str:
        """Generate summary report from simulation results."""
        lines = [
            "=" * 70,
            "PORTFOLIO SIMULATION REPORT",
            "=" * 70,
            f"Seed Money: ${self.seed_money:,.2f}",
            f"Commission Rate: {self.commission_rate * 100:.2f}%",
            f"Slippage Rate: {self.slippage_rate * 100:.3f}%",
            "=" * 70,
            ""
        ]

        total_final = 0
        total_buy_hold = 0

        for ticker, res in results.items():
            if 'error' in res:
                lines.append(f"{ticker}: Error - {res['error']}")
                continue

            total_final += res['final_value']
            total_buy_hold += res['buy_hold_final']

            lines.extend([
                f"[{ticker}]",
                f"  Period: {res['start_date']} ~ {res['end_date']} ({res['trading_days']} days)",
                f"  Final Value: ${res['final_value']:,.2f} ({res['total_return_pct']:+.2f}%)",
                f"  Buy & Hold:  ${res['buy_hold_final']:,.2f} ({res['buy_hold_return_pct']:+.2f}%)",
                f"  Outperformance: {res['outperformance_pct']:+.2f}%",
                f"  Max Drawdown: {res['max_drawdown_pct']:.2f}%",
                f"  Total Trades: {res['total_trades']}",
                f"  Total Commission: ${res['total_commission']:.2f}",
                ""
            ])

        # Summary
        num_tickers = len([r for r in results.values() if 'error' not in r])
        if num_tickers > 0:
            total_seed = self.seed_money * num_tickers
            total_return = (total_final - total_seed) / total_seed * 100
            total_bh_return = (total_buy_hold - total_seed) / total_seed * 100

            lines.extend([
                "=" * 70,
                "TOTAL PORTFOLIO SUMMARY",
                "=" * 70,
                f"Total Invested: ${total_seed:,.2f}",
                f"Final Value (Strategy): ${total_final:,.2f} ({total_return:+.2f}%)",
                f"Final Value (Buy&Hold): ${total_buy_hold:,.2f} ({total_bh_return:+.2f}%)",
                f"Strategy vs Buy&Hold: {total_return - total_bh_return:+.2f}%",
                "=" * 70
            ])

        return "\n".join(lines)


def main():
    """Run simulation on trading results."""
    import argparse

    parser = argparse.ArgumentParser(description='Portfolio Simulator')
    parser.add_argument('--csv', type=str, default='trading_results/all_tickers_results.csv',
                       help='Path to results CSV')
    parser.add_argument('--seed', type=float, default=10000.0,
                       help='Seed money ($)')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate (0.001 = 0.1%%)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for report')
    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.csv):
        print(f"Error: File not found: {args.csv}")
        print("\nAvailable result files:")
        for f in os.listdir('trading_results'):
            if f.endswith('.csv'):
                print(f"  - trading_results/{f}")
        return

    # Run simulation
    simulator = PortfolioSimulator(
        seed_money=args.seed,
        commission_rate=args.commission
    )

    results = simulator.simulate_all_tickers(args.csv)
    report = simulator.generate_report(results)

    print(report)

    # Save report if output specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")


if __name__ == '__main__':
    main()
