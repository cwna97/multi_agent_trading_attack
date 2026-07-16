"""
Linear Architecture Setup.

Topology:
  START → [Market → Social → News → Fundamentals (sequential)] → Linear Synthesizer → END

Fair comparison design (identical 1st layer as Centralized/Decentralized/MoA):
  - Layer 1: same 4 Analysts (Market, Social, News, Fundamentals), same prompts/LLM
  - Layer 2: Linear Synthesizer — processes reports one-by-one in fixed order,
             accumulating understanding sequentially before making the final call

The architectural variable vs other structures:
  - Linear:       ordered accumulation (report 1 → update → report 2 → update → ...)
  - Centralized:  simultaneous holistic synthesis of all reports
  - Decentralized: independent vote per report, majority wins
  - MoA:          diversity-aware aggregation

LLM calls per run: ~5 (4 analysts + 1 synthesizer)
"""

from typing import Dict
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.managers.linear_synthesizer import create_linear_synthesizer
from .conditional_logic import ConditionalLogic


class LinearGraphSetup:

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
            raise ValueError("Linear Graph Setup Error: no analysts selected!")

        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        for analyst_type in ["market", "social", "news", "fundamentals"]:
            if analyst_type in selected_analysts:
                analyst_nodes[analyst_type] = self._get_analyst_node(analyst_type, analyst_overrides)
                delete_nodes[analyst_type] = create_msg_delete()
                tool_nodes[analyst_type] = self.tool_nodes[analyst_type]

        synthesizer = decision_maker_override or create_linear_synthesizer(self.deep_thinking_llm)

        workflow = StateGraph(AgentState)

        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type])
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        workflow.add_node("Linear Synthesizer", synthesizer)

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
                workflow.add_edge(current_clear, "Linear Synthesizer")

        workflow.add_edge("Linear Synthesizer", END)

        return workflow.compile()
