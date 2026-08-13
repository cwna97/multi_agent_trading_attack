"""
Hybrid Architecture — Layer 3 Aggregator.

Receives four peer-informed revised analyst reports (each analyst has already
read and integrated their peers' perspectives) and issues the final trading decision.
"""


def create_hybrid_aggregator(llm):

    def hybrid_aggregator_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        market_report      = state.get("market_report", "")
        social_report      = state.get("sentiment_report", "")
        news_report        = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        prompt = f"""You are a senior portfolio manager making the final trading decision on {ticker} for {trade_date}.

Unlike a standard committee, each analyst below has already reviewed and integrated their peers' findings before finalizing their perspective. These are peer-informed revised reports, not isolated silos.

=== MARKET ANALYST (Peer-Revised) ===
{market_report}

=== SOCIAL MEDIA ANALYST (Peer-Revised) ===
{social_report}

=== NEWS ANALYST (Peer-Revised) ===
{news_report}

=== FUNDAMENTALS ANALYST (Peer-Revised) ===
{fundamentals_report}

Because each analyst has already cross-referenced their peers, residual disagreements carry strong signal — they represent genuine divergence after informed deliberation, not just siloed perspectives.

Your task:
1. **Identify the consensus**: Where do peer-informed analysts align? Strong post-deliberation consensus amplifies confidence.
2. **Resolve residual divergence**: Where they still disagree, weigh arguments by quality of reasoning and domain authority.
3. **Make the final call**: Issue a decisive recommendation backed by the integrated multi-domain analysis.

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

    return hybrid_aggregator_node
