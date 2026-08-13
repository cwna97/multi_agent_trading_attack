"""
Null Architecture Setup (Solo / No-Communication Baseline).

Topology:
  START → {Analyst} → Solo Decision Maker → END

Exactly ONE analyst is selected; its report goes directly to a decision.
No other analyst is present in the graph — no synthesis, no coordination,
no voting, no peer-review.

Purpose: a per-model, per-analyst compliance-control baseline. Run this
structure once per analyst (market / social / news) under each attack to
measure the underlying LLM's raw susceptibility to that attack, with the
aggregation-structure confound (Linear/Centralized/Decentralized/Hybrid)
removed entirely. Compare against those structures' ASR to separate
"structure effect" from "model compliance effect".

LLM calls per run: ~2 (1 analyst + 1 solo decision maker)
"""

from typing import Dict
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.managers.solo_decision_maker import create_solo_decision_maker
from .conditional_logic import ConditionalLogic


class NullGraphSetup:

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

    def setup_graph(self, selected_analysts=["market"], analyst_overrides=None, decision_maker_override=None):
        if len(selected_analysts) != 1:
            raise ValueError(
                "Null Graph Setup Error: expected exactly 1 analyst "
                f"(no-communication design), got {len(selected_analysts)}: {selected_analysts}"
            )
        analyst_type = selected_analysts[0]

        analyst_node = self._get_analyst_node(analyst_type, analyst_overrides)
        clear_node = create_msg_delete()
        tool_node = self.tool_nodes[analyst_type]

        decision_maker = decision_maker_override or create_solo_decision_maker(
            self.deep_thinking_llm, analyst_type
        )

        workflow = StateGraph(AgentState)

        analyst_label = f"{analyst_type.capitalize()} Analyst"
        clear_label = f"Msg Clear {analyst_type.capitalize()}"
        tools_label = f"tools_{analyst_type}"

        workflow.add_node(analyst_label, analyst_node)
        workflow.add_node(clear_label, clear_node)
        workflow.add_node(tools_label, tool_node)
        workflow.add_node("Solo Decision Maker", decision_maker)

        workflow.add_edge(START, analyst_label)
        workflow.add_conditional_edges(
            analyst_label,
            getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
            [tools_label, clear_label],
        )
        workflow.add_edge(tools_label, analyst_label)
        workflow.add_edge(clear_label, "Solo Decision Maker")
        workflow.add_edge("Solo Decision Maker", END)

        return workflow.compile()
