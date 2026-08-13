# TradingAgents/graph/attack_setup.py
"""
Attack Graph Setup Module

This module provides graph setup for adversarial attack experiments.
It allows substituting normal agents with poisoned versions.
"""

from typing import Dict, Any, Optional, Callable
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.attacks.attack_03_persuasive_adversary import (
    create_adversary_bull_researcher,
    create_adversary_bear_researcher,
)
from tradingagents.agents.attacks.attack_04_objective_hijacking import (
    create_hijacked_trader,
)
from tradingagents.agents.attacks.attack_05_jailbreaking import (
    create_jailbroken_risk_manager,
)

from .conditional_logic import ConditionalLogic


class AttackGraphSetup:
    """Handles the setup and configuration of the agent graph with attack capabilities."""

    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
        attack_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with required components.

        Args:
            quick_thinking_llm: LLM for quick analysis tasks
            deep_thinking_llm: LLM for deep reasoning tasks
            tool_nodes: Dictionary of tool nodes for each analyst
            bull_memory: Memory for bull researcher
            bear_memory: Memory for bear researcher
            trader_memory: Memory for trader
            invest_judge_memory: Memory for investment judge
            risk_manager_memory: Memory for risk manager
            conditional_logic: Logic for conditional graph edges
            attack_config: Configuration for attacks. Example:
                {
                    "attack_type": "data_poisoning",
                    "target_agent": "sentiment",
                    "enabled": True
                }
        """
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic
        self.attack_config = attack_config or {}

    def _get_analyst_node(self, analyst_type: str) -> Callable:
        """Get the appropriate analyst node, potentially with attack modifications.

        Args:
            analyst_type: Type of analyst ("market", "social", "news", "fundamentals")

        Returns:
            The analyst node function (normal or poisoned)

        Note:
            - data_poisoning attack is handled via poisoned news data files,
              not by replacing the analyst node. See config["poisoned_news"] flag.
            - persuasive_adversary attack targets researchers, not analysts,
              and is handled in setup_graph() method.
        """
        # Return normal analyst
        if analyst_type == "market":
            return create_market_analyst(self.quick_thinking_llm)
        elif analyst_type == "social":
            # Pass the exact tool list registered in this graph's "social"
            # ToolNode (see setup.py's GraphSetup for the same reasoning).
            return create_social_media_analyst(
                self.quick_thinking_llm,
                tools=list(self.tool_nodes["social"].tools_by_name.values()),
            )
        elif analyst_type == "news":
            return create_news_analyst(self.quick_thinking_llm)
        elif analyst_type == "fundamentals":
            return create_fundamentals_analyst(self.quick_thinking_llm)
        else:
            raise ValueError(f"Unknown analyst type: {analyst_type}")

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph with potential attack modifications.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst (can be poisoned)
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        for analyst_type in selected_analysts:
            if analyst_type in ["market", "social", "news", "fundamentals"]:
                analyst_nodes[analyst_type] = self._get_analyst_node(analyst_type)
                delete_nodes[analyst_type] = create_msg_delete()
                tool_nodes[analyst_type] = self.tool_nodes[analyst_type]

        # Create researcher nodes - check for persuasive attack
        attack_enabled = self.attack_config.get("enabled", False)

        # Support both single attack_type and multiple attack_types list
        attack_types = self.attack_config.get("attack_types", [])
        if not attack_types:
            single_type = self.attack_config.get("attack_type", "")
            attack_types = [single_type] if single_type else []

        target_agent = self.attack_config.get("target_agent", "")
        persuasion_strength = self.attack_config.get("persuasion_strength", "strong")

        if attack_enabled and "persuasive_adversary" in attack_types:
            if target_agent == "bull_researcher":
                # Replace Bull Researcher with Adversary
                bull_researcher_node = create_adversary_bull_researcher(
                    self.quick_thinking_llm, self.bull_memory, persuasion_strength
                )
                bear_researcher_node = create_bear_researcher(
                    self.quick_thinking_llm, self.bear_memory
                )
            elif target_agent == "bear_researcher":
                # Replace Bear Researcher with Adversary
                bull_researcher_node = create_bull_researcher(
                    self.quick_thinking_llm, self.bull_memory
                )
                bear_researcher_node = create_adversary_bear_researcher(
                    self.quick_thinking_llm, self.bear_memory, persuasion_strength
                )
            else:
                # Default: normal researchers
                bull_researcher_node = create_bull_researcher(
                    self.quick_thinking_llm, self.bull_memory
                )
                bear_researcher_node = create_bear_researcher(
                    self.quick_thinking_llm, self.bear_memory
                )
        else:
            # Normal setup
            bull_researcher_node = create_bull_researcher(
                self.quick_thinking_llm, self.bull_memory
            )
            bear_researcher_node = create_bear_researcher(
                self.quick_thinking_llm, self.bear_memory
            )

        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory
        )

        # Create trader node - check for objective hijacking attack
        if attack_enabled and "objective_hijacking" in attack_types:
            # Replace Trader with Hijacked (Contrarian) Trader
            trader_node = create_hijacked_trader(
                self.quick_thinking_llm, self.trader_memory
            )
        else:
            trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)

        # Create risk analysis nodes
        risky_analyst = create_risky_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        safe_analyst = create_safe_debator(self.quick_thinking_llm)

        # Create risk manager node - check for jailbreaking attack
        if attack_enabled and "jailbreaking" in attack_types:
            jailbreak_method = self.attack_config.get("jailbreak_method", "authority_override")
            jailbreak_target = self.attack_config.get("target_action", "BUY")
            risk_manager_node = create_jailbroken_risk_manager(
                self.deep_thinking_llm, self.risk_manager_memory, jailbreak_method, jailbreak_target
            )
        else:
            risk_manager_node = create_risk_manager(
                self.deep_thinking_llm, self.risk_manager_memory
            )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Risky Analyst", risky_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Safe Analyst", safe_analyst)
        workflow.add_node("Risk Judge", risk_manager_node)

        # Define edges
        # Start with the first analyst
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        # Which researcher opens the debate. Defaults to Bull (original behavior).
        # should_continue_debate() below already alternates generically based on
        # who spoke last, so flipping just this entry point is enough to reverse
        # the whole debate order - e.g. persuasive_bearish sets this to "bear" so
        # the (genuine, non-adversarial) Bull Researcher gets the last word instead
        # of always ceding it to whichever side is Bear, which otherwise gives Bear
        # a free structural advantage regardless of which side is the adversary.
        debate_first_speaker = self.attack_config.get("debate_first_speaker", "bull")
        first_debate_node = "Bear Researcher" if debate_first_speaker == "bear" else "Bull Researcher"

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to the debate's first speaker if this is the last analyst
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, first_debate_node)

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Risky Analyst")
        workflow.add_conditional_edges(
            "Risky Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Safe Analyst": "Safe Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Safe Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Risky Analyst": "Risky Analyst",
                "Risk Judge": "Risk Judge",
            },
        )

        workflow.add_edge("Risk Judge", END)

        # Compile and return
        return workflow.compile()
