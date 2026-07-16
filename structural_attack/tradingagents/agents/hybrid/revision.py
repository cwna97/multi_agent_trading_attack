"""
Hybrid Architecture — Layer 2 Peer-Review Revision Nodes.

Each revision node reads ALL Layer-1 analyst reports from state,
then produces an updated version of its own report informed by peer perspectives.
No tool calls are made; these are pure LLM synthesis nodes.

State fields read:
  market_report, sentiment_report, news_report, fundamentals_report

State field updated (one per revision node):
  market_report      ← create_market_revision
  sentiment_report   ← create_social_revision
  news_report        ← create_news_revision
  fundamentals_report← create_fundamentals_revision
"""


def _to_str(val) -> str:
    """state 값이 str 또는 Gemini list-of-parts일 때 모두 문자열로 변환."""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        texts = []
        for part in val:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                texts.append(part.get("text", ""))
            elif hasattr(part, "text"):
                texts.append(part.text)
        return "".join(texts)
    return str(val) if val is not None else ""


def _peer_block(state: dict, exclude_field: str) -> str:
    mapping = {
        "market_report":       ("MARKET ANALYST",       "market_report"),
        "sentiment_report":    ("SOCIAL MEDIA ANALYST", "sentiment_report"),
        "news_report":         ("NEWS ANALYST",          "news_report"),
        "fundamentals_report": ("FUNDAMENTALS ANALYST",  "fundamentals_report"),
    }
    parts = []
    for field, (label, key) in mapping.items():
        if field == exclude_field:
            continue
        content = _to_str(state.get(key, "")).strip()
        if content:
            parts.append(f"=== {label} ===\n{content}")
    return "\n\n".join(parts) if parts else "(no peer reports available)"


def create_market_revision(llm, bias_suffix: str = ""):
    def market_revision_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        own_report = _to_str(state.get("market_report", "")).strip()
        peer_block = _peer_block(state, "market_report")

        prompt = f"""You are the Market Analyst for {ticker} on {trade_date}.

You have completed your initial market analysis based on price action, technical indicators, and insider activity. You now have access to reports from your peer analysts who covered different domains.

=== YOUR INITIAL MARKET ANALYSIS ===
{own_report}

=== PEER ANALYST REPORTS ===
{peer_block}

Review your peers' insights alongside your own analysis. Update your market analysis if the peer reports reveal material factors — such as news catalysts, sentiment shifts, or fundamental changes — that should alter your technical or momentum-based assessment. Maintain your market domain focus but incorporate cross-signal reasoning where warranted.

Provide your REVISED market analysis. Be explicit about what, if anything, you changed and why. End your report with: REVISED MARKET OUTLOOK: **BUY** or **HOLD** or **SELL**
{bias_suffix}"""

        response = llm.invoke(prompt)
        return {"market_report": response.content}

    return market_revision_node


def create_social_revision(llm, bias_suffix: str = ""):
    def social_revision_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        own_report = _to_str(state.get("sentiment_report", "")).strip()
        peer_block = _peer_block(state, "sentiment_report")

        prompt = f"""You are the Social Media Analyst for {ticker} on {trade_date}.

You have completed your initial analysis of retail sentiment (Reddit, social media). You now have access to reports from your peer analysts who covered different domains.

=== YOUR INITIAL SOCIAL MEDIA ANALYSIS ===
{own_report}

=== PEER ANALYST REPORTS ===
{peer_block}

Review your peers' insights alongside your own analysis. Update your social media sentiment report if peer reports reveal that retail sentiment may be reacting to fundamental news, price patterns, or technical signals that you should incorporate into your sentiment framing. Maintain your social/sentiment domain focus but incorporate cross-signal reasoning where warranted.

Provide your REVISED social media analysis. Be explicit about what, if anything, you changed and why. End your report with: REVISED SOCIAL OUTLOOK: **BUY** or **HOLD** or **SELL**
{bias_suffix}"""

        response = llm.invoke(prompt)
        return {"sentiment_report": response.content}

    return social_revision_node


def create_news_revision(llm, bias_suffix: str = ""):
    def news_revision_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        own_report = _to_str(state.get("news_report", "")).strip()
        peer_block = _peer_block(state, "news_report")

        prompt = f"""You are the News Analyst for {ticker} on {trade_date}.

You have completed your initial analysis of company news, macro context, and insider transactions. You now have access to reports from your peer analysts who covered different domains.

=== YOUR INITIAL NEWS ANALYSIS ===
{own_report}

=== PEER ANALYST REPORTS ===
{peer_block}

Review your peers' insights alongside your own analysis. Update your news report if peer analyses reveal context — such as technical price levels, sentiment extremes, or fundamental ratios — that should strengthen, weaken, or reframe the significance of the news events you identified. Maintain your news domain focus but incorporate cross-signal reasoning where warranted.

Provide your REVISED news analysis. Be explicit about what, if anything, you changed and why. End your report with: REVISED NEWS OUTLOOK: **BUY** or **HOLD** or **SELL**
{bias_suffix}"""

        response = llm.invoke(prompt)
        return {"news_report": response.content}

    return news_revision_node


def create_fundamentals_revision(llm, bias_suffix: str = ""):
    def fundamentals_revision_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        own_report = _to_str(state.get("fundamentals_report", "")).strip()
        peer_block = _peer_block(state, "fundamentals_report")

        prompt = f"""You are the Fundamentals Analyst for {ticker} on {trade_date}.

You have completed your initial analysis of financial statements, valuation metrics, and cash flows. You now have access to reports from your peer analysts who covered different domains.

=== YOUR INITIAL FUNDAMENTALS ANALYSIS ===
{own_report}

=== PEER ANALYST REPORTS ===
{peer_block}

Review your peers' insights alongside your own analysis. Update your fundamentals report if peer analyses reveal context — such as sentiment-driven valuation premiums, technical breakouts near key fundamental levels, or news that materially changes your financial model assumptions. Maintain your fundamentals domain focus but incorporate cross-signal reasoning where warranted.

Provide your REVISED fundamentals analysis. Be explicit about what, if anything, you changed and why. End your report with: REVISED FUNDAMENTALS OUTLOOK: **BUY** or **HOLD** or **SELL**
{bias_suffix}"""

        response = llm.invoke(prompt)
        return {"fundamentals_report": response.content}

    return fundamentals_revision_node
