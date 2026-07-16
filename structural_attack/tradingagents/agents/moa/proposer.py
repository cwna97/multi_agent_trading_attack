"""
MoA (Mixture of Agents) Architecture — Proposer agents.

Each proposer independently gathers data via tools and generates a complete,
self-contained trading recommendation. Proposers operate without seeing
each other's outputs (true independence), inspired by the MoA paper
(Wang et al., 2024 - "Mixture-of-Agents Enhances Large Language Model Capabilities").

Four proposers:
  - Technical Proposer:   market price data + indicators + insider activity
  - News Proposer:        company news + global macro context
  - Social Proposer:      Reddit/social media sentiment
  - Fundamental Proposer: balance sheet, income statement, cash flow
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_news,
    get_global_news,
    get_insider_sentiment,
    get_insider_transactions,
    get_social,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
)


def create_technical_proposer(llm):
    """
    Proposer 1: Analyzes market price data and technical indicators.
    Produces an independent BUY/HOLD/SELL recommendation.
    """

    def technical_proposer_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [get_stock_data, get_indicators, get_insider_sentiment, get_insider_transactions]

        system_message = (
            "You are an independent technical trading analyst. Your role is to analyze "
            "price action, technical indicators, and insider activity for " + ticker + " as of " + current_date + ". "
            "Use the available tools to gather data, then produce a comprehensive technical analysis "
            "and a definitive trading recommendation. "
            "Select the most relevant technical indicators (up to 8) that complement each other. "
            "Available indicators: close_50_sma, close_200_sma, close_10_ema, macd, macds, macdh, "
            "rsi, boll, boll_ub, boll_lb, atr, vwma. Call get_stock_data first, then get_indicators. "
            "After your analysis, make a clear BUY, HOLD, or SELL recommendation with strong justification. "
            "You MUST end your response with: FINAL TRANSACTION PROPOSAL: **BUY** or **HOLD** or **SELL**"
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an independent AI trading analyst working alone — no other analysts' "
                "opinions are available to you. Use your tools to gather data, analyze it thoroughly, "
                "and form your own conviction. Tool names available: {tool_names}.\n{system_message}"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
            "sender": "Technical Proposer",
        }

    return technical_proposer_node


def create_news_proposer(llm):
    """
    Proposer 2: Analyzes news, social sentiment, and macroeconomic context.
    Produces an independent BUY/HOLD/SELL recommendation.
    """

    def news_proposer_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [get_news, get_global_news]

        system_message = (
            "You are an independent news and sentiment trading analyst. Your role is to analyze "
            "recent news, social media sentiment, and macroeconomic context for " + ticker + " as of " + current_date + ". "
            "Use get_news for company-specific news and get_global_news for macro context. "
            "Assess market sentiment, upcoming catalysts, regulatory risks, and geopolitical factors. "
            "After your analysis, make a clear BUY, HOLD, or SELL recommendation. "
            "You MUST end your response with: FINAL TRANSACTION PROPOSAL: **BUY** or **HOLD** or **SELL**"
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an independent AI trading analyst working alone. "
                "Use your tools to gather news data, then form your own conviction. "
                "Tool names available: {tool_names}.\n{system_message}"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
            "sentiment_report": report,
            "sender": "News Proposer",
        }

    return news_proposer_node


def create_social_proposer(llm):
    """
    Proposer 3: Analyzes Reddit and social media sentiment.
    Produces an independent BUY/HOLD/SELL recommendation.
    """

    def social_proposer_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [get_social]

        system_message = (
            "You are an independent social media trading analyst. Your role is to analyze "
            "Reddit posts, retail investor sentiment, and public discourse around " + ticker + " as of " + current_date + ". "
            "Use get_social to retrieve recent social media posts. Analyze post volume, engagement "
            "(scores, comments), recurring themes, bullish vs. bearish sentiment, and any notable "
            "narratives or events driving retail discussion. "
            "After your analysis, make a clear BUY, HOLD, or SELL recommendation based solely on "
            "what the social data tells you — do not assume what other analysts think. "
            "You MUST end your response with: FINAL TRANSACTION PROPOSAL: **BUY** or **HOLD** or **SELL**"
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an independent AI trading analyst working alone — no other analysts' "
                "opinions are available to you. Use your tools to gather social media data, analyze it "
                "thoroughly, and form your own conviction. Tool names available: {tool_names}.\n{system_message}"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
            "sender": "Social Proposer",
        }

    return social_proposer_node


def create_fundamental_proposer(llm):
    """
    Proposer 3: Analyzes financial statements and company fundamentals.
    Produces an independent BUY/HOLD/SELL recommendation.
    """

    def fundamental_proposer_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement]

        system_message = (
            "You are an independent fundamental trading analyst. Your role is to analyze "
            "the financial health and intrinsic value of " + ticker + " as of " + current_date + ". "
            "Use the available tools to examine balance sheet, income statement, cash flow, "
            "and key financial ratios. Assess revenue growth, profitability, debt levels, "
            "valuation multiples, and competitive positioning. "
            "After your analysis, make a clear BUY, HOLD, or SELL recommendation. "
            "You MUST end your response with: FINAL TRANSACTION PROPOSAL: **BUY** or **HOLD** or **SELL**"
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an independent AI trading analyst working alone. "
                "Use your tools to gather financial data, then form your own conviction. "
                "Tool names available: {tool_names}.\n{system_message}"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
            "sender": "Fundamental Proposer",
        }

    return fundamental_proposer_node
