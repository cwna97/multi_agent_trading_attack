"""
Example script to evaluate trading agent performance

This script demonstrates how to use the TradingAgentEvaluator
to assess the performance of your trading agents.
"""

import pandas as pd
import yfinance as yf
from tradingagents.evaluation import TradingAgentEvaluator


def get_actual_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch actual stock prices from Yahoo Finance

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with date and close price
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)

    prices_df = pd.DataFrame({
        'date': hist.index.strftime('%Y-%m-%d'),
        'close': hist['Close'].values
    })

    return prices_df


def main():
    """Main evaluation function"""

    # Configuration
    TICKER = "NVDA"
    START_DATE = "2025-12-01"
    END_DATE = "2025-12-10"
    RESULTS_DIR = "./eval_results"

    print(f"Evaluating trading agent performance for {TICKER}")
    print("=" * 60)

    # Initialize evaluator
    evaluator = TradingAgentEvaluator(results_dir=RESULTS_DIR)

    # Option 1: Evaluate without actual prices (only decision analysis)
    print("\n1. Evaluating decision-making patterns...")
    print("-" * 60)

    try:
        results_no_prices = evaluator.evaluate_performance(
            ticker=TICKER,
            actual_prices=None
        )

        print(f"Total decisions analyzed: {results_no_prices['total_decisions']}")
        print(f"Decision consistency: {results_no_prices['consistency_score']:.3f}")
        print("\nDecision distribution:")
        for decision, count in results_no_prices['decision_distribution']['final_decisions'].items():
            print(f"  {decision}: {count}")

        print("\nAgent agreement rates:")
        for agent, rate in results_no_prices['agent_analysis']['agreement_rates'].items():
            agent_name = agent.replace('_decision', '').replace('_', ' ').title()
            print(f"  {agent_name}: {rate*100:.1f}%")

    except Exception as e:
        print(f"Error in evaluation: {e}")

    # Option 2: Evaluate with actual prices (full performance metrics)
    print("\n\n2. Evaluating with actual price data...")
    print("-" * 60)

    try:
        # Fetch actual prices
        actual_prices = get_actual_prices(TICKER, START_DATE, END_DATE)
        print(f"Fetched {len(actual_prices)} price data points")

        # Generate comprehensive report
        report = evaluator.generate_report(
            ticker=TICKER,
            actual_prices=actual_prices,
            output_file=f"{RESULTS_DIR}/evaluation_report_{TICKER}.txt"
        )

        print(report)

        print(f"\nDetailed report saved to: {RESULTS_DIR}/evaluation_report_{TICKER}.txt")

    except Exception as e:
        print(f"Error in price-based evaluation: {e}")
        import traceback
        traceback.print_exc()


def batch_evaluation():
    """
    Example: Batch evaluation for multiple tickers
    """
    TICKERS = ["NVDA", "AAPL", "MSFT"]
    RESULTS_DIR = "./eval_results"

    evaluator = TradingAgentEvaluator(results_dir=RESULTS_DIR)

    summary_results = []

    for ticker in TICKERS:
        try:
            print(f"\nEvaluating {ticker}...")

            results = evaluator.evaluate_performance(ticker=ticker)

            summary_results.append({
                'ticker': ticker,
                'total_decisions': results['total_decisions'],
                'consistency': results['consistency_score'],
                'buy_decisions': results['decision_distribution']['final_decisions'].get('BUY', 0),
                'sell_decisions': results['decision_distribution']['final_decisions'].get('SELL', 0),
                'hold_decisions': results['decision_distribution']['final_decisions'].get('HOLD', 0),
            })

        except Exception as e:
            print(f"  Error: {e}")

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_results)
    print("\n" + "=" * 60)
    print("Summary Across All Tickers:")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_df.to_csv(f"{RESULTS_DIR}/batch_evaluation_summary.csv", index=False)
    print(f"\nSummary saved to: {RESULTS_DIR}/batch_evaluation_summary.csv")


def compare_debate_impact():
    """
    Example: Compare performance with different debate rounds
    """
    # This would require running the agent with different configurations
    # and then comparing the results

    evaluator = TradingAgentEvaluator(results_dir="./eval_results")

    # Hypothetical strategy names based on config
    strategies = [
        "TradingAgentsStrategy_1round",
        "TradingAgentsStrategy_2rounds",
        "TradingAgentsStrategy_3rounds"
    ]

    comparison_df = evaluator.compare_strategies(
        ticker="NVDA",
        strategy_names=strategies,
        actual_prices=get_actual_prices("NVDA", "2025-12-01", "2025-12-10")
    )

    print("\nStrategy Comparison:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    # Run main evaluation
    main()

    # Uncomment to run batch evaluation
    # batch_evaluation()

    # Uncomment to compare strategies
    # compare_debate_impact()
