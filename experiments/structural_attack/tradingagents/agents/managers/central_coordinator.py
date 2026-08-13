"""
Centralized Architecture: Single coordinator agent that synthesizes all analyst reports
and makes the final trading decision without debate or intermediate agents.
"""


def create_central_coordinator(llm):
    """
    Central coordinator for the Centralized architecture.
    Receives all 4 analyst reports and directly outputs the final trading decision.
    Replaces the entire Bull/Bear debate + Research Manager + Trader + Risk debate + Risk Judge chain.
    """

    def coordinator_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        market_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        prompt = f"""You are a central trading coordinator with full authority over investment decisions for {ticker} on {trade_date}.

You have received comprehensive analysis from 4 independent specialized analysts:

=== MARKET & TECHNICAL ANALYSIS ===
{market_report}

=== SOCIAL MEDIA SENTIMENT ANALYSIS ===
{sentiment_report}

=== NEWS & MACROECONOMIC ANALYSIS ===
{news_report}

=== FUNDAMENTALS ANALYSIS ===
{fundamentals_report}

As the central coordinator, synthesize all the above information holistically and make a definitive trading decision. Your response should include:

1. **Key Signal Summary**: Highlight the most important signals from each analyst (2-3 sentences each)
2. **Convergence/Divergence**: Note where analysts agree or conflict
3. **Risk Assessment**: Evaluate the main risks to your decision
4. **Final Decision**: A clear, decisive recommendation

You MUST conclude with exactly: FINAL TRANSACTION PROPOSAL: **BUY** or FINAL TRANSACTION PROPOSAL: **HOLD** or FINAL TRANSACTION PROPOSAL: **SELL**
"""

        response = llm.invoke(prompt)

        # Initialize required debate states with empty values
        empty_invest_debate = {
            "bull_history": "",
            "bear_history": "",
            "history": "",
            "current_response": response.content,
            "judge_decision": response.content,
            "count": 0,
        }
        empty_risk_debate = {
            "risky_history": "",
            "safe_history": "",
            "neutral_history": "",
            "history": "",
            "latest_speaker": "",
            "current_risky_response": "",
            "current_safe_response": "",
            "current_neutral_response": "",
            "judge_decision": response.content,
            "count": 0,
        }

        return {
            "investment_plan": response.content,
            "trader_investment_plan": response.content,
            "investment_debate_state": empty_invest_debate,
            "risk_debate_state": empty_risk_debate,
            "final_trade_decision": response.content,
        }

    return coordinator_node
