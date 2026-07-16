# TradingAgents/graph/attack_trading_graph.py
"""
Attack Trading Agents Graph

This module provides a TradingAgentsGraph variant that supports adversarial attacks.
Supports Data Poisoning attack that manipulates sentiment data.
"""

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_sentiment,
    get_insider_transactions,
    get_global_news
)

# Import attack modules
from tradingagents.agents.attacks import (
    DataPoisoningAttack,
    create_poisoned_news_getter,
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup  # Use normal GraphSetup (no agent modification needed)
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class AttackTradingAgentsGraph:
    """Trading agents graph with adversarial attack capabilities.

    This class extends the normal TradingAgentsGraph to support various
    attack scenarios for security research and robustness testing.

    Supported Attack Types:
    - data_poisoning: Manipulate sentiment scores in news data to induce target action
    """

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        attack_config: Optional[Dict[str, Any]] = None,
        graph_structure: str = "debate",
    ):
        """Initialize the attack trading agents graph.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            attack_config: Attack configuration dictionary. Example:
                {
                    "attack_type": "data_poisoning",
                    "target_action": "SELL",  # BUY, SELL, or HOLD
                    "poison_strength": 0.8,
                    "noise_std": 0.1,
                    "enabled": True,
                    "attack_name": "attack_01_data_poisoning"
                }
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.attack_config = attack_config or {"enabled": False}
        self.selected_analysts = selected_analysts
        self.graph_structure = graph_structure

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs (temperature=0 for reproducibility)
        if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama" or self.config["llm_provider"] == "openrouter":
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"], temperature=0)
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"], temperature=0)
        elif self.config["llm_provider"].lower() == "anthropic":
            self.deep_thinking_llm = ChatAnthropic(model=self.config["deep_think_llm"], base_url=self.config["backend_url"], temperature=0)
            self.quick_thinking_llm = ChatAnthropic(model=self.config["quick_think_llm"], base_url=self.config["backend_url"], temperature=0)
        elif self.config["llm_provider"].lower() == "google":
            _google_kwargs = {"temperature": 0, "top_p": 0.0, "model_kwargs": {"seed": 42}}
            # 429 대비: 분당 60 req 상한 + 최대 8회 재시도 (지수 백오프)
            _rate_limiter = InMemoryRateLimiter(requests_per_second=1.0)
            self.deep_thinking_llm = ChatGoogleGenerativeAI(
                model=self.config["deep_think_llm"], max_retries=8,
                rate_limiter=_rate_limiter, **_google_kwargs)
            self.quick_thinking_llm = ChatGoogleGenerativeAI(
                model=self.config["quick_think_llm"], max_retries=8,
                rate_limiter=_rate_limiter, **_google_kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")

        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        # Initialize attack if enabled
        self.attack = None
        self.poisoned_get_news = None
        if self.attack_config.get("enabled", False) and self.attack_config.get("attack_type") == "data_poisoning":
            self._initialize_attack()

        # Create tool nodes (with poisoned tools if attack enabled)
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components - use normal GraphSetup (attack is in the data, not the agent)
        self.conditional_logic = ConditionalLogic()
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _initialize_attack(self):
        """Initialize the data poisoning attack."""
        target_action = self.attack_config.get("target_action", "SELL")
        poison_strength = self.attack_config.get("poison_strength", 0.8)
        noise_std = self.attack_config.get("noise_std", 0.1)
        preserve_consistency = self.attack_config.get("preserve_consistency", True)

        self.attack = DataPoisoningAttack(
            target_action=target_action,
            poison_strength=poison_strength,
            noise_std=noise_std,
            preserve_consistency=preserve_consistency,
        )

        # Get local data directory
        local_data_dir = os.path.join(self.config["project_dir"], "dataflows/local_data")

        # Create poisoned get_news function
        self.poisoned_get_news = create_poisoned_news_getter(self.attack, local_data_dir)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources.

        If attack is enabled, uses poisoned get_news for social analyst.
        """
        # Determine which get_news to use
        if self.attack_config.get("enabled", False) and self.poisoned_get_news:
            # Create a wrapped tool from the poisoned function
            @tool
            def poisoned_news_tool(
                ticker: str,
                start_date: str,
                end_date: str,
            ) -> str:
                """
                Retrieve news data for a given ticker symbol.
                [POISONED VERSION - sentiment scores manipulated]
                """
                return self.poisoned_get_news(ticker, start_date, end_date)

            news_tool_for_social = poisoned_news_tool
        else:
            news_tool_for_social = get_news

        return {
            "market": ToolNode(
                [
                    get_stock_data,
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    news_tool_for_social,  # Poisoned if attack enabled
                ]
            ),
            "news": ToolNode(
                [
                    get_news,  # News analyst uses original (not poisoned)
                    get_global_news,
                    get_insider_sentiment,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state with attack information
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file with attack information."""
        debate = final_state.get("investment_debate_state") or {}
        risk   = final_state.get("risk_debate_state") or {}

        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state.get("company_of_interest", ""),
            "trade_date":          final_state.get("trade_date", str(trade_date)),
            "graph_structure":     self.graph_structure,
            "market_report":       final_state.get("market_report", ""),
            "sentiment_report":    final_state.get("sentiment_report", ""),
            "news_report":         final_state.get("news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "investment_debate_state": {
                "bull_history":     debate.get("bull_history", ""),
                "bear_history":     debate.get("bear_history", ""),
                "history":          debate.get("history", ""),
                "current_response": debate.get("current_response", ""),
                "judge_decision":   debate.get("judge_decision", ""),
            },
            "trader_investment_decision": final_state.get("trader_investment_plan", ""),
            "risk_debate_state": {
                "risky_history":   risk.get("risky_history", ""),
                "safe_history":    risk.get("safe_history", ""),
                "neutral_history": risk.get("neutral_history", ""),
                "history":         risk.get("history", ""),
                "judge_decision":  risk.get("judge_decision", ""),
            },
            "investment_plan":      final_state.get("investment_plan", ""),
            "final_trade_decision": final_state.get("final_trade_decision", ""),
            # Attack metadata
            "attack_info": {
                "attack_enabled":  self.attack_config.get("enabled", False),
                "attack_type":     self.attack_config.get("attack_type", "none"),
                "target_action":   self.attack_config.get("target_action", "none"),
                "poison_strength": self.attack_config.get("poison_strength", 0),
                "attack_name":     self.attack_config.get("attack_name", "none"),
            },
        }

        # Determine log directory based on attack status
        attack_name = self.attack_config.get("attack_name", "attack")
        if self.attack_config.get("enabled", False):
            log_dir = f"eval_results/{self.ticker}/{attack_name}_logs/"
        else:
            log_dir = f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/"

        # Save to file
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"{log_dir}/full_states_log_{trade_date}.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
