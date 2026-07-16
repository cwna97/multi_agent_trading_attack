"""
Decentralized Architecture: Each analyst votes independently.
This aggregator collects all 4 analyst reports and determines the majority vote
without a central authority making the final call — the decision emerges from consensus.
"""


def create_vote_aggregator(llm):
    """
    Vote aggregator for the Decentralized architecture.
    Extracts a BUY/HOLD/SELL vote from each analyst's report and determines
    the majority outcome. No single authority — consensus decides.
    """

    def vote_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        market_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        prompt = f"""You are a neutral vote counter for a decentralized trading committee on {ticker} ({trade_date}).

Four independent analysts have each submitted their analysis. Extract one vote (BUY, HOLD, or SELL) from each report based on the overall sentiment and conclusions expressed.

--- Market Analyst Report ---
{market_report}

--- Sentiment Analyst Report ---
{sentiment_report}

--- News Analyst Report ---
{news_report}

--- Fundamentals Analyst Report ---
{fundamentals_report}

Instructions:
1. For each analyst, identify whether their report leans BUY, HOLD, or SELL
2. Count the votes
3. The majority vote wins (if 2-2 tie, default to HOLD)

Format your response as:
Market Vote: BUY/HOLD/SELL — [one-line rationale]
Sentiment Vote: BUY/HOLD/SELL — [one-line rationale]
News Vote: BUY/HOLD/SELL — [one-line rationale]
Fundamentals Vote: BUY/HOLD/SELL — [one-line rationale]

Vote Tally: BUY=X, HOLD=Y, SELL=Z
Majority Decision: [BUY/HOLD/SELL]

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

    return vote_node
