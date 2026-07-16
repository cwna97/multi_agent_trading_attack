from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_social
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm, bias_suffix: str = ""):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_social,
        ]

        system_message = (
            "You are a social media analyst tasked with analyzing Reddit and X (Twitter) posts and public sentiment for a specific company over the past week. You will be given a company's ticker symbol. Your objective is to write a comprehensive report detailing your analysis, insights, and implications for traders and investors based on what retail investors and the public are saying on social media about this company. Use the get_social(ticker, start_date, end_date) tool to retrieve social media posts. Analyze post volume, engagement (scores, comments), recurring themes, bullish vs. bearish sentiment, and any notable events or narratives driving discussion. Do not simply state that trends are mixed — provide detailed, fine-grained analysis and insights that may help traders make decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + bias_suffix
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The current company we want to analyze is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node
