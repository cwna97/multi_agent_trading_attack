"""
Decentralized Architecture Setup.

Topology:
  START → [Market → Social → News → Fundamentals (sequential)] → Vote Aggregator → END

Each analyst independently generates their report. The Vote Aggregator extracts
one vote per analyst (BUY/HOLD/SELL) and determines the majority. No central
authority overrides the democratic outcome.

Key difference from Centralized:
  - Centralized: One authority synthesizes everything (top-down)
  - Decentralized: Each agent has equal voting power (bottom-up consensus)

LLM calls per run: ~4 (4 analysts only - the vote aggregator does pure string
parsing on the votes the analysts already state in their own reports, no LLM call)
Expected speedup vs Linear: ~2-3x
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.managers.vote_aggregator import create_vote_aggregator
from .conditional_logic import ConditionalLogic

# Appended to each analyst's system prompt ONLY in the decentralized topology.
# Every analyst already knows how to write its report; this just asks it to also
# state its own BUY/HOLD/SELL vote in a fixed, greppable format so the Vote
# Aggregator can tally votes with plain string parsing instead of another LLM call.
VOTE_INSTRUCTION = """

Additionally, since your report will be tallied against three other independent \
analysts' votes to reach a committee decision, end your report with exactly one \
line, on its own, in this exact format:
MY VOTE: BUY
(or MY VOTE: HOLD, or MY VOTE: SELL)
This line records your own independent recommendation based solely on your domain's analysis."""


class DecentralizedGraphSetup:

    def __init__(
        self,
        quick_thinking_llm,
        deep_thinking_llm,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
    ):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic

    def _get_analyst_node(self, analyst_type: str, analyst_overrides: dict):
        if analyst_overrides and analyst_type in analyst_overrides:
            return analyst_overrides[analyst_type]
        creators = {
            "market":       create_market_analyst,
            "social":       create_social_media_analyst,
            "news":         create_news_analyst,
            "fundamentals": create_fundamentals_analyst,
        }
        return creators[analyst_type](self.quick_thinking_llm, bias_suffix=VOTE_INSTRUCTION)

    def setup_graph(self, selected_analysts=["market", "social", "news", "fundamentals"], analyst_overrides=None, decision_maker_override=None):
        if len(selected_analysts) == 0:
            raise ValueError("Decentralized Graph Setup Error: no analysts selected!")

        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        for analyst_type in ["market", "social", "news", "fundamentals"]:
            if analyst_type in selected_analysts:
                analyst_nodes[analyst_type] = self._get_analyst_node(analyst_type, analyst_overrides)
                delete_nodes[analyst_type] = create_msg_delete()
                tool_nodes[analyst_type] = self.tool_nodes[analyst_type]

        # Vote aggregator does pure string parsing on each analyst's own "MY VOTE:" line
        # (see VOTE_INSTRUCTION above) - no LLM call, so the `llm` arg is unused but kept
        # for interface parity with the other topologies' decision-maker factories.
        # selected_analysts is passed through so an analyst that isn't part of this run
        # (e.g. fundamentals, when only market/social/news are selected) doesn't get
        # counted as a guaranteed phantom vote defaulting to HOLD.
        vote_aggregator_node = decision_maker_override or create_vote_aggregator(
            self.deep_thinking_llm, selected_analysts=selected_analysts
        )

        workflow = StateGraph(AgentState)

        # Add analyst nodes
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type])
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        workflow.add_node("Vote Aggregator", vote_aggregator_node)

        # Wire analysts sequentially, then vote
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, "Vote Aggregator")

        workflow.add_edge("Vote Aggregator", END)

        return workflow.compile()
