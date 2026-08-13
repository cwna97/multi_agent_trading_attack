"""
Hybrid Architecture Setup.

Topology:
  Layer 1 — Data Gathering (sequential, tool-using, identical to other structures):
    START → Market Analyst → Msg Clear → Social Analyst → Msg Clear
          → News Analyst   → Msg Clear → Fundamentals Analyst → Msg Clear

  Layer 2 — Peer Review (sequential, no tool calls, new):
    → Market Revision → Social Revision → News Revision → Fundamentals Revision
    Each revision node reads ALL Layer-1 reports from state and updates its own report.

  Layer 3 — Aggregation:
    → Hybrid Aggregator → END

Architectural variable vs other structures:
  - Linear/Centralized/Decentralized/MoA: no inter-analyst communication
  - Hybrid: each analyst revises their report after reading peers' Layer-1 outputs

LLM calls per run: ~9 (4 L1 analysts + 4 L2 revisions + 1 aggregator)
"""

from typing import Dict
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.hybrid.revision import (
    create_market_revision,
    create_social_revision,
    create_news_revision,
    create_fundamentals_revision,
)
from tradingagents.agents.hybrid.aggregator import create_hybrid_aggregator
from .conditional_logic import ConditionalLogic

_REVISION_FACTORIES = {
    "market":       create_market_revision,
    "social":       create_social_revision,
    "news":         create_news_revision,
    "fundamentals": create_fundamentals_revision,
}


class HybridGraphSetup:

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

    def setup_graph(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        analyst_overrides=None,
        decision_maker_override=None,
    ):
        if not selected_analysts:
            raise ValueError("Hybrid Graph Setup Error: no analysts selected!")

        # ── Layer 1 nodes ──────────────────────────────────────────────────────
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        for analyst_type in ["market", "social", "news", "fundamentals"]:
            if analyst_type in selected_analysts:
                analyst_nodes[analyst_type] = self._get_analyst_node(analyst_type, analyst_overrides)
                delete_nodes[analyst_type]  = create_msg_delete()
                tool_nodes[analyst_type]    = self.tool_nodes[analyst_type]

        # ── Layer 2 revision nodes (only for selected analysts) ────────────────
        revision_nodes = {
            analyst_type: _REVISION_FACTORIES[analyst_type](self.quick_thinking_llm)
            for analyst_type in selected_analysts
            if analyst_type in _REVISION_FACTORIES
        }

        # ── Layer 3 aggregator ─────────────────────────────────────────────────
        aggregator = decision_maker_override or create_hybrid_aggregator(self.deep_thinking_llm)

        # ── Build graph ────────────────────────────────────────────────────────
        workflow = StateGraph(AgentState)

        # Layer 1: analyst + tool + msg-clear nodes
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type])
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Layer 2: revision nodes
        for analyst_type, node in revision_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Revision", node)

        # Layer 3: aggregator
        workflow.add_node("Hybrid Aggregator", aggregator)

        # ── Wire Layer 1 (sequential chain with tool loops and Msg Clear) ──────
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools   = f"tools_{analyst_type}"
            current_clear   = f"Msg Clear {analyst_type.capitalize()}"

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
                # Last L1 analyst → first L2 revision node
                first_revision = f"{selected_analysts[0].capitalize()} Revision"
                workflow.add_edge(current_clear, first_revision)

        # ── Wire Layer 2 (sequential revision chain, direct edges) ────────────
        for i, analyst_type in enumerate(selected_analysts):
            current_revision = f"{analyst_type.capitalize()} Revision"
            if i < len(selected_analysts) - 1:
                next_revision = f"{selected_analysts[i+1].capitalize()} Revision"
                workflow.add_edge(current_revision, next_revision)
            else:
                workflow.add_edge(current_revision, "Hybrid Aggregator")

        # ── Wire Layer 3 ───────────────────────────────────────────────────────
        workflow.add_edge("Hybrid Aggregator", END)

        return workflow.compile()
