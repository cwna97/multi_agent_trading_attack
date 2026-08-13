"""
Trading Agent Performance Metrics

This module provides various metrics to evaluate trading agent performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime


class TradingMetrics:
    """Calculate trading performance metrics"""

    @staticmethod
    def calculate_returns(
        decisions: List[str],
        actual_prices: List[float],
        dates: List[str]
    ) -> Dict[str, float]:
        """
        Calculate returns based on trading decisions

        Args:
            decisions: List of trading decisions (BUY, SELL, HOLD)
            actual_prices: List of actual stock prices
            dates: List of trading dates

        Returns:
            Dictionary containing return metrics
        """
        if len(decisions) != len(actual_prices) or len(decisions) != len(dates):
            raise ValueError("All input lists must have the same length")

        # Initialize position and portfolio
        position = 0  # 0: no position, 1: long position, -1: short position
        entry_price = 0
        returns = []
        trades = []

        for i in range(len(decisions)):
            decision = decisions[i].upper()
            current_price = actual_prices[i]

            # Execute trading logic
            if decision == "BUY" and position == 0:
                position = 1
                entry_price = current_price
                trades.append({
                    'date': dates[i],
                    'action': 'BUY',
                    'price': current_price
                })

            elif decision == "SELL" and position == 1:
                position = 0
                trade_return = (current_price - entry_price) / entry_price
                returns.append(trade_return)
                trades.append({
                    'date': dates[i],
                    'action': 'SELL',
                    'price': current_price,
                    'return': trade_return
                })

            elif decision == "HOLD":
                # If holding a position, track unrealized P&L
                if position == 1:
                    unrealized_return = (current_price - entry_price) / entry_price

        # Calculate metrics
        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'avg_return': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'trades': trades
            }

        return {
            'total_return': np.sum(returns),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'total_trades': len(returns),
            'max_return': np.max(returns),
            'min_return': np.min(returns),
            'trades': trades
        }

    @staticmethod
    def calculate_sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe Ratio

        Args:
            returns: List of returns
            risk_free_rate: Annual risk-free rate (default: 2%)

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = np.array(returns) - (risk_free_rate / 252)  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    @staticmethod
    def calculate_max_drawdown(
        equity_curve: List[float]
    ) -> Tuple[float, int, int]:
        """
        Calculate Maximum Drawdown

        Args:
            equity_curve: List of portfolio values over time

        Returns:
            Tuple of (max_drawdown, start_index, end_index)
        """
        if len(equity_curve) == 0:
            return 0.0, 0, 0

        equity_curve = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max

        max_dd = np.min(drawdown)
        end_idx = np.argmin(drawdown)
        start_idx = np.argmax(equity_curve[:end_idx]) if end_idx > 0 else 0

        return max_dd, start_idx, end_idx

    @staticmethod
    def calculate_decision_accuracy(
        predicted_decisions: List[str],
        optimal_decisions: List[str]
    ) -> Dict[str, float]:
        """
        Calculate decision accuracy against optimal strategy

        Args:
            predicted_decisions: Agent's decisions
            optimal_decisions: Optimal decisions (e.g., from backtesting)

        Returns:
            Dictionary with accuracy metrics
        """
        if len(predicted_decisions) != len(optimal_decisions):
            raise ValueError("Decision lists must have same length")

        correct = sum(1 for p, o in zip(predicted_decisions, optimal_decisions) if p == o)
        total = len(predicted_decisions)

        # Confusion matrix
        confusion = {
            'BUY': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
            'SELL': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
            'HOLD': {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        }

        for pred, opt in zip(predicted_decisions, optimal_decisions):
            confusion[opt.upper()][pred.upper()] += 1

        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'confusion_matrix': confusion
        }

    @staticmethod
    def calculate_consistency_score(
        decisions_sequence: List[str]
    ) -> float:
        """
        Calculate decision consistency (avoid excessive trading)

        Args:
            decisions_sequence: Sequence of decisions

        Returns:
            Consistency score (0-1, higher is more consistent)
        """
        if len(decisions_sequence) <= 1:
            return 1.0

        # Count decision changes
        changes = sum(1 for i in range(1, len(decisions_sequence))
                     if decisions_sequence[i] != decisions_sequence[i-1])

        # Normalize by length
        consistency = 1 - (changes / (len(decisions_sequence) - 1))

        return consistency


class AgentContributionAnalyzer:
    """Analyze individual agent contributions to final decision"""

    @staticmethod
    def analyze_agent_agreement(
        agent_decisions: Dict[str, str],
        final_decision: str
    ) -> Dict[str, any]:
        """
        Analyze how each agent's decision aligns with final decision

        Args:
            agent_decisions: Dict mapping agent names to their decisions
            final_decision: The final trading decision

        Returns:
            Analysis of agent contributions
        """
        agreements = {}
        for agent, decision in agent_decisions.items():
            agreements[agent] = decision.upper() == final_decision.upper()

        agreement_rate = sum(agreements.values()) / len(agreements) if agreements else 0

        return {
            'individual_agreements': agreements,
            'overall_agreement_rate': agreement_rate,
            'majority_decision': max(set(agent_decisions.values()),
                                   key=list(agent_decisions.values()).count)
        }

    @staticmethod
    def track_agent_accuracy(
        agent_decisions_history: Dict[str, List[str]],
        actual_outcomes: List[bool]
    ) -> Dict[str, float]:
        """
        Track each agent's accuracy over time

        Args:
            agent_decisions_history: Dict of agent name to list of decisions
            actual_outcomes: List of True/False for profitable outcomes

        Returns:
            Dict of agent accuracies
        """
        accuracies = {}

        for agent, decisions in agent_decisions_history.items():
            if len(decisions) != len(actual_outcomes):
                continue

            # Simple accuracy: BUY decisions that led to profit
            correct = sum(1 for d, o in zip(decisions, actual_outcomes)
                         if (d == 'BUY' and o) or (d == 'SELL' and not o))

            accuracies[agent] = correct / len(decisions) if decisions else 0.0

        return accuracies
