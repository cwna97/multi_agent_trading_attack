"""
Trading Agent Evaluator

This module evaluates trading agent performance using various metrics
and provides comprehensive performance reports.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from .metrics import TradingMetrics, AgentContributionAnalyzer


class TradingAgentEvaluator:
    """
    Comprehensive evaluator for trading agent system
    """

    def __init__(self, results_dir: str = "./eval_results"):
        """
        Initialize evaluator

        Args:
            results_dir: Directory containing agent decision logs
        """
        self.results_dir = Path(results_dir)
        self.metrics_calculator = TradingMetrics()
        self.contribution_analyzer = AgentContributionAnalyzer()

    def load_decision_logs(
        self,
        ticker: str,
        strategy_name: str = "TradingAgentsStrategy"
    ) -> List[Dict]:
        """
        Load decision logs from JSON files

        Args:
            ticker: Stock ticker symbol
            strategy_name: Strategy name

        Returns:
            List of decision log dictionaries
        """
        log_dir = self.results_dir / ticker / f"{strategy_name}_logs"

        if not log_dir.exists():
            raise ValueError(f"Log directory not found: {log_dir}")

        logs = []
        for json_file in sorted(log_dir.glob("full_states_log_*.json")):
            with open(json_file, 'r') as f:
                data = json.load(f)
                logs.extend(data.values())  # Extract daily logs

        return logs

    def extract_decisions(self, logs: List[Dict]) -> pd.DataFrame:
        """
        Extract decisions from logs into structured DataFrame

        Args:
            logs: List of decision logs

        Returns:
            DataFrame with structured decision data
        """
        records = []

        for log in logs:
            record = {
                'date': log.get('trade_date'),
                'ticker': log.get('company_of_interest'),
                'trader_decision': self._parse_decision(
                    log.get('trader_investment_decision', '')
                ),
                'final_decision': self._parse_decision(
                    log.get('final_trade_decision', '')
                ),
                'market_report_decision': self._parse_decision(
                    log.get('market_report', '')
                ),
                'sentiment_report_decision': self._parse_decision(
                    log.get('sentiment_report', '')
                ),
                'news_report_decision': self._parse_decision(
                    log.get('news_report', '')
                ),
                'fundamentals_report_decision': self._parse_decision(
                    log.get('fundamentals_report', '')
                ),
            }

            # Extract debate information
            if 'investment_debate_state' in log:
                debate = log['investment_debate_state']
                record['judge_decision'] = self._parse_decision(
                    debate.get('judge_decision', '')
                )

            records.append(record)

        return pd.DataFrame(records)

    def _parse_decision(self, text: str) -> str:
        """
        Parse decision from text report

        Args:
            text: Report text containing decision

        Returns:
            Decision string (BUY, SELL, HOLD)
        """
        text_upper = text.upper()

        # Look for final transaction proposal
        if "FINAL TRANSACTION PROPOSAL:" in text_upper:
            if "BUY" in text_upper.split("FINAL TRANSACTION PROPOSAL:")[-1]:
                return "BUY"
            elif "SELL" in text_upper.split("FINAL TRANSACTION PROPOSAL:")[-1]:
                return "SELL"
            elif "HOLD" in text_upper.split("FINAL TRANSACTION PROPOSAL:")[-1]:
                return "HOLD"

        # Fallback: simple keyword search
        if "BUY" in text_upper:
            return "BUY"
        elif "SELL" in text_upper:
            return "SELL"
        elif "HOLD" in text_upper:
            return "HOLD"

        return "UNKNOWN"

    def evaluate_performance(
        self,
        ticker: str,
        actual_prices: Optional[pd.DataFrame] = None,
        strategy_name: str = "TradingAgentsStrategy"
    ) -> Dict:
        """
        Evaluate overall trading performance

        Args:
            ticker: Stock ticker symbol
            actual_prices: DataFrame with actual stock prices (date, close)
            strategy_name: Strategy name

        Returns:
            Comprehensive evaluation results
        """
        # Load decision logs
        logs = self.load_decision_logs(ticker, strategy_name)
        decisions_df = self.extract_decisions(logs)

        results = {
            'ticker': ticker,
            'evaluation_date': datetime.now().isoformat(),
            'total_decisions': len(decisions_df),
            'decision_distribution': {},
            'agent_analysis': {},
            'performance_metrics': {}
        }

        # Decision distribution
        results['decision_distribution'] = {
            'final_decisions': decisions_df['final_decision'].value_counts().to_dict(),
            'trader_decisions': decisions_df['trader_decision'].value_counts().to_dict()
        }

        # Agent contribution analysis
        agent_columns = [
            'market_report_decision',
            'sentiment_report_decision',
            'news_report_decision',
            'fundamentals_report_decision'
        ]

        agent_agreements = {}
        for _, row in decisions_df.iterrows():
            for col in agent_columns:
                if col not in agent_agreements:
                    agent_agreements[col] = []

                agent_decision = row.get(col, 'UNKNOWN')
                final_decision = row['final_decision']
                agent_agreements[col].append(
                    agent_decision == final_decision
                )

        results['agent_analysis']['agreement_rates'] = {
            agent: np.mean(agreements) if agreements else 0.0
            for agent, agreements in agent_agreements.items()
        }

        # If actual prices provided, calculate financial metrics
        if actual_prices is not None:
            decisions_df = decisions_df.merge(
                actual_prices,
                left_on='date',
                right_on='date',
                how='inner'
            )

            if 'close' in decisions_df.columns:
                return_metrics = self.metrics_calculator.calculate_returns(
                    decisions=decisions_df['final_decision'].tolist(),
                    actual_prices=decisions_df['close'].tolist(),
                    dates=decisions_df['date'].tolist()
                )

                results['performance_metrics'] = return_metrics

                # Calculate Sharpe ratio if we have returns
                if 'trades' in return_metrics and len(return_metrics['trades']) > 0:
                    trade_returns = [
                        t['return'] for t in return_metrics['trades']
                        if 'return' in t
                    ]

                    if trade_returns:
                        results['performance_metrics']['sharpe_ratio'] = \
                            self.metrics_calculator.calculate_sharpe_ratio(trade_returns)

        # Decision consistency
        results['consistency_score'] = \
            self.metrics_calculator.calculate_consistency_score(
                decisions_df['final_decision'].tolist()
            )

        return results

    def generate_report(
        self,
        ticker: str,
        actual_prices: Optional[pd.DataFrame] = None,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive evaluation report

        Args:
            ticker: Stock ticker symbol
            actual_prices: DataFrame with actual prices
            output_file: Optional output file path for report

        Returns:
            Report as formatted string
        """
        results = self.evaluate_performance(ticker, actual_prices)

        report_lines = [
            "=" * 80,
            f"Trading Agent Evaluation Report",
            f"Ticker: {ticker}",
            f"Evaluation Date: {results['evaluation_date']}",
            "=" * 80,
            "",
            f"Total Decisions Analyzed: {results['total_decisions']}",
            "",
            "Decision Distribution:",
            "-" * 40
        ]

        # Final decisions
        report_lines.append("Final Decisions:")
        for decision, count in results['decision_distribution']['final_decisions'].items():
            pct = (count / results['total_decisions']) * 100
            report_lines.append(f"  {decision}: {count} ({pct:.1f}%)")

        report_lines.append("")
        report_lines.append("Agent Agreement Rates:")
        report_lines.append("-" * 40)

        for agent, rate in results['agent_analysis']['agreement_rates'].items():
            agent_name = agent.replace('_decision', '').replace('_', ' ').title()
            report_lines.append(f"  {agent_name}: {rate*100:.1f}%")

        report_lines.append("")
        report_lines.append(f"Decision Consistency Score: {results['consistency_score']:.3f}")

        # Performance metrics
        if results['performance_metrics']:
            report_lines.extend([
                "",
                "Financial Performance Metrics:",
                "-" * 40,
                f"  Total Return: {results['performance_metrics'].get('total_return', 0)*100:.2f}%",
                f"  Average Return per Trade: {results['performance_metrics'].get('avg_return', 0)*100:.2f}%",
                f"  Win Rate: {results['performance_metrics'].get('win_rate', 0)*100:.1f}%",
                f"  Total Trades: {results['performance_metrics'].get('total_trades', 0)}",
            ])

            if 'sharpe_ratio' in results['performance_metrics']:
                report_lines.append(
                    f"  Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.3f}"
                )

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
                f.write("\n\n")
                f.write("Detailed Results (JSON):\n")
                json.dump(results, f, indent=2)

        return report

    def compare_strategies(
        self,
        ticker: str,
        strategy_names: List[str],
        actual_prices: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compare multiple strategies

        Args:
            ticker: Stock ticker
            strategy_names: List of strategy names to compare
            actual_prices: Actual price data

        Returns:
            Comparison DataFrame
        """
        comparison_data = []

        for strategy in strategy_names:
            try:
                results = self.evaluate_performance(
                    ticker,
                    actual_prices,
                    strategy_name=strategy
                )

                comparison_data.append({
                    'strategy': strategy,
                    'total_decisions': results['total_decisions'],
                    'consistency': results['consistency_score'],
                    'total_return': results['performance_metrics'].get('total_return', 0),
                    'win_rate': results['performance_metrics'].get('win_rate', 0),
                    'sharpe_ratio': results['performance_metrics'].get('sharpe_ratio', 0),
                })
            except Exception as e:
                print(f"Error evaluating {strategy}: {e}")

        return pd.DataFrame(comparison_data)
