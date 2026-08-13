"""
Linear Architecture: Sequential accumulation synthesizer.

Processes analyst reports one-by-one in a fixed order, updating its running
analysis after each report. The final decision emerges from the last step.

Key characteristic: ordered accumulation — each report adds to the
prior understanding rather than being considered simultaneously.

Contrast with:
  - Centralized:   all reports read simultaneously, holistic synthesis
  - Decentralized: each report gets an independent vote, majority wins
  - MoA:           diversity-aware synthesis, resolves convergence/divergence
"""


def create_linear_synthesizer(llm):
    """
    Linear synthesizer for the Linear architecture.
    Accumulates analyst reports sequentially and makes the final decision
    after integrating all information in order.
    """

    def synthesizer_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        market_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        prompt = f"""You are a trading analyst making a decision on {ticker} for {trade_date}.

You will read four analyst reports one by one. After each report, update your running analysis.
Your final decision is made after integrating all reports in order.

---
STEP 1 — Market & Technical Analysis:
{market_report}

After reading the above, form an initial view (1-2 sentences).

---
STEP 2 — Social Media Sentiment:
{sentiment_report}

Update your view given this new information (1-2 sentences).

---
STEP 3 — News & Macroeconomic Context:
{news_report}

Update your view given this new information (1-2 sentences).

---
STEP 4 — Fundamentals Analysis:
{fundamentals_report}

Final integration: reconcile all accumulated evidence and make a definitive trading decision.
Provide your final reasoning (3-5 sentences) and conclude.

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

    return synthesizer_node
