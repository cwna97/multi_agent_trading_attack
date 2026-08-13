"""
Quick test of the evaluator functionality
"""

from tradingagents.evaluation import TradingAgentEvaluator

# Initialize evaluator
evaluator = TradingAgentEvaluator(results_dir="./eval_results")

print("Testing TradingAgentEvaluator...")
print("=" * 60)

# Test loading and basic evaluation
try:
    print("\n1. Loading decision logs for NVDA...")
    logs = evaluator.load_decision_logs(ticker="NVDA")
    print(f"   ✓ Loaded {len(logs)} decision logs")

    print("\n2. Extracting decisions into DataFrame...")
    decisions_df = evaluator.extract_decisions(logs)
    print(f"   ✓ Extracted {len(decisions_df)} decision records")
    print(f"\n   Sample decisions:")
    print(decisions_df[['date', 'final_decision', 'trader_decision']].head())

    print("\n3. Running performance evaluation (without prices)...")
    results = evaluator.evaluate_performance(ticker="NVDA", actual_prices=None)

    print(f"\n   ✓ Evaluation completed!")
    print(f"   - Total decisions: {results['total_decisions']}")
    print(f"   - Consistency score: {results['consistency_score']:.3f}")
    print(f"\n   Decision distribution:")
    for decision, count in results['decision_distribution']['final_decisions'].items():
        pct = (count / results['total_decisions']) * 100
        print(f"     {decision}: {count} ({pct:.1f}%)")

    print(f"\n   Agent agreement rates:")
    for agent, rate in results['agent_analysis']['agreement_rates'].items():
        agent_name = agent.replace('_decision', '').replace('_', ' ').title()
        print(f"     {agent_name}: {rate*100:.1f}%")

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
