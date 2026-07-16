"""
MoA (Mixture of Agents) Architecture — Aggregator agent.

The aggregator receives independent proposals from all proposer agents
and synthesizes them into a final, high-quality trading decision.
This is the key insight from the MoA paper: an aggregator with access to
diverse independent perspectives consistently outperforms any single proposer.
"""


def create_moa_aggregator(llm):
    """
    MoA Aggregator: synthesizes independent proposer recommendations into
    the final trading decision. Operates at a higher level than any single
    proposer, leveraging the diversity of independent analysis.
    """

    def aggregator_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        technical_proposal  = state.get("market_report", "")
        news_proposal       = state.get("news_report", "")
        social_proposal     = state.get("sentiment_report", "")
        fundamental_proposal = state.get("fundamentals_report", "")

        prompt = f"""You are a senior portfolio manager synthesizing independent expert opinions on {ticker} for {trade_date}.

Four independent analysts — each working without knowledge of the others — have submitted their analyses:

=== MARKET ANALYST (Price Action, Indicators & Insider Activity) ===
{technical_proposal}

=== NEWS ANALYST (Company News & Macro Context) ===
{news_proposal}

=== SOCIAL MEDIA ANALYST (Reddit & Retail Sentiment) ===
{social_proposal}

=== FUNDAMENTALS ANALYST (Financial Statements & Valuation) ===
{fundamental_proposal}

Your task as the aggregator is to:
1. **Identify convergence**: Where do analysts agree? Strong consensus amplifies confidence.
2. **Resolve divergence**: Where they disagree, weigh the arguments by quality of reasoning.
3. **Synthesize a superior decision**: Combine the best insights from all four perspectives.
4. **Make a final call**: A decisive BUY, HOLD, or SELL recommendation.

Research shows that aggregating diverse independent expert opinions produces better decisions than any single expert alone. Leverage this diversity.

You MUST conclude with exactly: FINAL TRANSACTION PROPOSAL: **BUY** or FINAL TRANSACTION PROPOSAL: **HOLD** or FINAL TRANSACTION PROPOSAL: **SELL**
"""

        response = llm.invoke(prompt)

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

    return aggregator_node
