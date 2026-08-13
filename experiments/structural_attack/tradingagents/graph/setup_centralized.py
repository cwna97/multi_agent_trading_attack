"""
Centralized Architecture Setup.

Topology:
  START → [Market → Social → News → Fundamentals (sequential)] → Central Coordinator → END

The Central Coordinator has full authority — it reads all 4 analyst reports and
directly produces the final BUY/HOLD/SELL decision.
No debate, no intermediate managers, no risk committee.

LLM calls per run: ~5 (4 analysts + 1 coordinator)
Expected speedup vs Linear: ~2-3x
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.managers.central_coordinator import create_central_coordinator
from .conditional_logic import ConditionalLogic


class CentralizedGraphSetup:

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
        return creators[analyst_type](self.quick_thinking_llm)

    def setup_graph(self, selected_analysts=["market", "social", "news", "fundamentals"], analyst_overrides=None, decision_maker_override=None):
        if len(selected_analysts) == 0:
            raise ValueError("Centralized Graph Setup Error: no analysts selected!")

        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        for analyst_type in ["market", "social", "news", "fundamentals"]:
            if analyst_type in selected_analysts:
                analyst_nodes[analyst_type] = self._get_analyst_node(analyst_type, analyst_overrides)
                delete_nodes[analyst_type] = create_msg_delete()
                tool_nodes[analyst_type] = self.tool_nodes[analyst_type]

        # Central coordinator uses deep thinking LLM (full authority)
        coordinator_node = decision_maker_override or create_central_coordinator(self.deep_thinking_llm)

        workflow = StateGraph(AgentState)

        # Add analyst nodes
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type])
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        workflow.add_node("Central Coordinator", coordinator_node)

        # Wire analysts sequentially
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
                # Last analyst → Central Coordinator (not Bull Researcher)
                workflow.add_edge(current_clear, "Central Coordinator")

        workflow.add_edge("Central Coordinator", END)

        return workflow.compile()
