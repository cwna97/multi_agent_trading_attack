"""
Visualization tools for trading agent evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class PerformanceVisualizer:
    """Visualize trading agent performance"""

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer

        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        sns.set_palette("husl")

    def plot_decision_distribution(
        self,
        decisions: pd.Series,
        title: str = "Decision Distribution",
        save_path: Optional[str] = None
    ):
        """
        Plot pie chart of decision distribution

        Args:
            decisions: Series of decisions
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        decision_counts = decisions.value_counts()
        colors = ['#2ecc71', '#e74c3c', '#3498db']

        ax.pie(
            decision_counts.values,
            labels=decision_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors[:len(decision_counts)]
        )

        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_agent_agreement(
        self,
        agreement_rates: Dict[str, float],
        title: str = "Agent Agreement with Final Decision",
        save_path: Optional[str] = None
    ):
        """
        Plot bar chart of agent agreement rates

        Args:
            agreement_rates: Dict mapping agent names to agreement rates
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Clean agent names
        agents = [
            name.replace('_decision', '').replace('_', ' ').title()
            for name in agreement_rates.keys()
        ]
        rates = [rate * 100 for rate in agreement_rates.values()]

        bars = ax.bar(agents, rates, color='steelblue', alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=10
            )

        ax.set_ylabel('Agreement Rate (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.3, label='50% threshold')
        ax.legend()

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_equity_curve(
        self,
        trades: List[Dict],
        initial_capital: float = 10000,
        title: str = "Portfolio Equity Curve",
        save_path: Optional[str] = None
    ):
        """
        Plot equity curve over time

        Args:
            trades: List of trade dictionaries with 'date' and 'return'
            initial_capital: Starting portfolio value
            title: Plot title
            save_path: Optional path to save figure
        """
        # Calculate equity curve
        equity = [initial_capital]
        dates = []

        for trade in trades:
            if 'return' in trade:
                equity.append(equity[-1] * (1 + trade['return']))
                dates.append(trade['date'])

        if len(dates) == 0:
            print("No trades with returns to plot")
            return

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(dates, equity[1:], linewidth=2, color='#2c3e50', label='Portfolio Value')
        ax.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')

        # Fill area
        ax.fill_between(
            dates,
            initial_capital,
            equity[1:],
            where=[e >= initial_capital for e in equity[1:]],
            alpha=0.3,
            color='green',
            label='Profit'
        )
        ax.fill_between(
            dates,
            initial_capital,
            equity[1:],
            where=[e < initial_capital for e in equity[1:]],
            alpha=0.3,
            color='red',
            label='Loss'
        )

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_returns_distribution(
        self,
        returns: List[float],
        title: str = "Returns Distribution",
        save_path: Optional[str] = None
    ):
        """
        Plot histogram of returns

        Args:
            returns: List of trade returns
            title: Plot title
            save_path: Optional path to save figure
        """
        if len(returns) == 0:
            print("No returns to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(returns, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Break-even')
        ax1.axvline(x=np.mean(returns), color='g', linestyle='-', linewidth=2, label='Mean')
        ax1.set_xlabel('Return', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Returns Histogram', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(returns, vert=True)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.5)
        ax2.set_ylabel('Return', fontsize=12)
        ax2.set_title('Returns Box Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_confusion_matrix(
        self,
        confusion_matrix: Dict[str, Dict[str, int]],
        title: str = "Decision Confusion Matrix",
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix heatmap

        Args:
            confusion_matrix: Nested dict of predictions vs actuals
            title: Plot title
            save_path: Optional path to save figure
        """
        # Convert to DataFrame
        df = pd.DataFrame(confusion_matrix).T

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            df,
            annot=True,
            fmt='d',
            cmap='Blues',
            square=True,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )

        ax.set_xlabel('Predicted Decision', fontsize=12)
        ax.set_ylabel('Actual Decision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def create_comprehensive_report(
        self,
        evaluation_results: Dict,
        output_dir: str = "./visualizations"
    ):
        """
        Create comprehensive visualization report

        Args:
            evaluation_results: Results from TradingAgentEvaluator
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        ticker = evaluation_results['ticker']

        # 1. Decision distribution
        if 'decision_distribution' in evaluation_results:
            final_decisions = evaluation_results['decision_distribution']['final_decisions']
            decisions_series = pd.Series(final_decisions)

            self.plot_decision_distribution(
                decisions_series,
                title=f"{ticker} - Decision Distribution",
                save_path=str(output_path / f"{ticker}_decision_distribution.png")
            )

        # 2. Agent agreement
        if 'agent_analysis' in evaluation_results:
            agreement_rates = evaluation_results['agent_analysis']['agreement_rates']

            self.plot_agent_agreement(
                agreement_rates,
                title=f"{ticker} - Agent Agreement Rates",
                save_path=str(output_path / f"{ticker}_agent_agreement.png")
            )

        # 3. Equity curve (if trades available)
        if 'performance_metrics' in evaluation_results:
            metrics = evaluation_results['performance_metrics']

            if 'trades' in metrics and len(metrics['trades']) > 0:
                self.plot_equity_curve(
                    metrics['trades'],
                    title=f"{ticker} - Portfolio Performance",
                    save_path=str(output_path / f"{ticker}_equity_curve.png")
                )

                # 4. Returns distribution
                trade_returns = [
                    t['return'] for t in metrics['trades']
                    if 'return' in t
                ]

                if trade_returns:
                    self.plot_returns_distribution(
                        trade_returns,
                        title=f"{ticker} - Returns Distribution",
                        save_path=str(output_path / f"{ticker}_returns_dist.png")
                    )

        print(f"\nVisualizations saved to: {output_path}")
