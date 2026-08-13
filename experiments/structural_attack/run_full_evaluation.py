"""
Comprehensive Trading Agent Evaluation with Visualization

This script runs a complete evaluation of the trading agent system,
including performance metrics, decision analysis, and visualizations.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
import json

from tradingagents.evaluation import TradingAgentEvaluator
from tradingagents.evaluation.visualizer import PerformanceVisualizer


def get_actual_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch actual stock prices"""
    print(f"  Fetching price data for {ticker}...")

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)

        prices_df = pd.DataFrame({
            'date': hist.index.strftime('%Y-%m-%d'),
            'close': hist['Close'].values
        })

        print(f"  ✓ Retrieved {len(prices_df)} price points")
        return prices_df

    except Exception as e:
        print(f"  ✗ Error fetching prices: {e}")
        return None


def run_comprehensive_evaluation(
    ticker: str,
    start_date: str,
    end_date: str,
    results_dir: str = "./eval_results",
    output_dir: str = "./evaluation_output"
):
    """
    Run comprehensive evaluation with all metrics and visualizations

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for price data
        end_date: End date for price data
        results_dir: Directory containing agent logs
        output_dir: Directory for output files
    """
    print("\n" + "=" * 80)
    print(f"COMPREHENSIVE TRADING AGENT EVALUATION")
    print(f"Ticker: {ticker}")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Initialize evaluator and visualizer
    evaluator = TradingAgentEvaluator(results_dir=results_dir)
    visualizer = PerformanceVisualizer()

    # Step 1: Decision Analysis
    print("\n[1/4] Analyzing trading decisions...")
    print("-" * 80)

    try:
        results = evaluator.evaluate_performance(ticker=ticker, actual_prices=None)

        print(f"✓ Analyzed {results['total_decisions']} decisions")
        print(f"✓ Decision consistency score: {results['consistency_score']:.3f}")

        # Display decision distribution
        print("\nDecision Distribution:")
        for decision, count in results['decision_distribution']['final_decisions'].items():
            pct = (count / results['total_decisions']) * 100
            print(f"  {decision:6s}: {count:3d} ({pct:5.1f}%)")

    except Exception as e:
        print(f"✗ Error in decision analysis: {e}")
        return

    # Step 2: Fetch Price Data
    print("\n[2/4] Fetching actual price data...")
    print("-" * 80)

    actual_prices = get_actual_prices(ticker, start_date, end_date)

    # Step 3: Performance Metrics
    print("\n[3/4] Calculating performance metrics...")
    print("-" * 80)

    if actual_prices is not None and len(actual_prices) > 0:
        try:
            results = evaluator.evaluate_performance(
                ticker=ticker,
                actual_prices=actual_prices
            )

            metrics = results.get('performance_metrics', {})

            if metrics:
                print("✓ Performance metrics calculated")
                print(f"\n  Total Return:      {metrics.get('total_return', 0)*100:7.2f}%")
                print(f"  Avg Return/Trade:  {metrics.get('avg_return', 0)*100:7.2f}%")
                print(f"  Win Rate:          {metrics.get('win_rate', 0)*100:7.1f}%")
                print(f"  Total Trades:      {metrics.get('total_trades', 0):7d}")

                if 'sharpe_ratio' in metrics:
                    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:7.3f}")

                if 'std_return' in metrics:
                    print(f"  Return Volatility: {metrics['std_return']*100:7.2f}%")

        except Exception as e:
            print(f"✗ Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠ No price data available, skipping performance metrics")

    # Step 4: Generate Visualizations
    print("\n[4/4] Generating visualizations...")
    print("-" * 80)

    try:
        viz_dir = output_path / "visualizations"
        visualizer.create_comprehensive_report(
            evaluation_results=results,
            output_dir=str(viz_dir)
        )

        print("✓ Visualizations created successfully")

    except Exception as e:
        print(f"✗ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

    # Step 5: Generate Text Report
    print("\n[5/5] Generating comprehensive report...")
    print("-" * 80)

    try:
        report_file = output_path / f"evaluation_report_{ticker}.txt"
        report = evaluator.generate_report(
            ticker=ticker,
            actual_prices=actual_prices,
            output_file=str(report_file)
        )

        print(f"✓ Report saved to: {report_file}")

        # Save JSON results
        json_file = output_path / f"evaluation_results_{ticker}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ JSON results saved to: {json_file}")

    except Exception as e:
        print(f"✗ Error generating report: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_path}")
    print("\nGenerated files:")
    print(f"  - Text Report:     evaluation_report_{ticker}.txt")
    print(f"  - JSON Results:    evaluation_results_{ticker}.json")
    print(f"  - Visualizations:  visualizations/{ticker}_*.png")
    print("\n" + "=" * 80)


def batch_evaluation_with_comparison(
    tickers: list,
    start_date: str,
    end_date: str,
    results_dir: str = "./eval_results",
    output_dir: str = "./evaluation_output"
):
    """
    Evaluate multiple tickers and create comparison report

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date for price data
        end_date: End date for price data
        results_dir: Directory containing agent logs
        output_dir: Directory for output files
    """
    print("\n" + "=" * 80)
    print("BATCH EVALUATION - MULTIPLE TICKERS")
    print("=" * 80)

    evaluator = TradingAgentEvaluator(results_dir=results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    summary_results = []

    for ticker in tickers:
        print(f"\n>>> Evaluating {ticker}...")

        try:
            # Get price data
            actual_prices = get_actual_prices(ticker, start_date, end_date)

            # Evaluate
            results = evaluator.evaluate_performance(
                ticker=ticker,
                actual_prices=actual_prices
            )

            # Extract key metrics
            metrics = results.get('performance_metrics', {})
            decisions = results.get('decision_distribution', {}).get('final_decisions', {})

            summary_results.append({
                'Ticker': ticker,
                'Total Decisions': results['total_decisions'],
                'Consistency': f"{results['consistency_score']:.3f}",
                'Total Return (%)': f"{metrics.get('total_return', 0)*100:.2f}",
                'Win Rate (%)': f"{metrics.get('win_rate', 0)*100:.1f}",
                'Total Trades': metrics.get('total_trades', 0),
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.3f}",
                'BUY': decisions.get('BUY', 0),
                'SELL': decisions.get('SELL', 0),
                'HOLD': decisions.get('HOLD', 0),
            })

            print(f"    ✓ Completed")

        except Exception as e:
            print(f"    ✗ Error: {e}")

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_results)

    print("\n" + "=" * 80)
    print("SUMMARY - ALL TICKERS")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_file = output_path / "batch_evaluation_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\n✓ Summary saved to: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    # Example 1: Single ticker comprehensive evaluation
    print("\n*** EXAMPLE 1: Single Ticker Evaluation ***")
    run_comprehensive_evaluation(
        ticker="NVDA",
        start_date="2025-11-01",
        end_date="2025-12-10",
        results_dir="./eval_results",
        output_dir="./evaluation_output"
    )

    # Example 2: Batch evaluation (uncomment to run)
    # print("\n\n*** EXAMPLE 2: Batch Evaluation ***")
    # batch_evaluation_with_comparison(
    #     tickers=["NVDA", "AAPL", "MSFT"],
    #     start_date="2025-11-01",
    #     end_date="2025-12-10",
    #     results_dir="./eval_results",
    #     output_dir="./evaluation_output"
    # )
