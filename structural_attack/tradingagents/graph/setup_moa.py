"""
MoA (Mixture of Agents) Architecture Setup.

Topology:
  START → [Market → Social → News → Fundamentals (sequential)] → MoA Aggregator → END

Inspired by Wang et al. (2024) "Mixture-of-Agents Enhances Large Language Model Capabilities".

Fair comparison design (same 1st layer as Centralized/Decentralized):
  - Layer 1: identical 4 Analysts (Market, Social, News, Fundamentals) with same prompts/LLM
  - Each analyst writes to a dedicated state field; Msg Clear isolates their message context
  - Layer 2 (Aggregator): synthesizes 4 independent reports leveraging diversity

The architectural variable is ONLY the aggregation strategy:
  - Centralized:   single authority with full synthesis power
  - Decentralized: majority vote, no override
  - MoA:           diversity-aware aggregation (convergence + divergence reasoning)

LLM calls per run: ~5 (4 analysts + 1 aggregator)
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.moa.aggregator import create_moa_aggregator
from .conditional_logic import ConditionalLogic


class MoAGraphSetup:

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
            raise ValueError("MoA Graph Setup Error: no analysts selected!")

        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        for analyst_type in ["market", "social", "news", "fundamentals"]:
            if analyst_type in selected_analysts:
                analyst_nodes[analyst_type] = self._get_analyst_node(analyst_type, analyst_overrides)
                delete_nodes[analyst_type] = create_msg_delete()
                tool_nodes[analyst_type] = self.tool_nodes[analyst_type]

        aggregator = decision_maker_override or create_moa_aggregator(self.deep_thinking_llm)

        workflow = StateGraph(AgentState)

        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type])
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        workflow.add_node("MoA Aggregator", aggregator)

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
                workflow.add_edge(current_clear, "MoA Aggregator")

        workflow.add_edge("MoA Aggregator", END)

        return workflow.compile()
