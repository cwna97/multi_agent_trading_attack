# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter

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
    get_global_news,
    get_social
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup          # debate (Bull/Bear + Risk debate pipeline)
from .setup_linear import LinearGraphSetup
from .setup_centralized import CentralizedGraphSetup
from .setup_decentralized import DecentralizedGraphSetup, VOTE_INSTRUCTION
from .setup_hybrid import HybridGraphSetup
from .setup_null import NullGraphSetup
from .attack_setup import AttackGraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor
from tradingagents.agents.attacks.attack_analyst_compromise import build_analyst_overrides
from tradingagents.agents.attacks.attack_decision_maker_hijacking import build_decision_maker_override

GRAPH_STRUCTURES = {
    "linear":        LinearGraphSetup,   # sequential accumulation (fair comparison set)
    "centralized":   CentralizedGraphSetup,
    "decentralized": DecentralizedGraphSetup,
    "hybrid":        HybridGraphSetup,   # peer-review layer before aggregation
    "null":          NullGraphSetup,     # solo analyst, no communication (compliance control)
    "debate":        GraphSetup,         # multi-stage Bull/Bear + Risk debate pipeline
}


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        attack_config: Dict[str, Any] = None,
        graph_structure: str = "linear",  # linear | centralized | decentralized | hybrid | null | debate
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            attack_config: Attack configuration for adversarial experiments.
                If None or empty, normal agents are used.
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.attack_config = attack_config or {}
        self.graph_structure = graph_structure

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs (temperature=0 for reproducibility)
        if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama":
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"], temperature=0)
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"], temperature=0)
        elif self.config["llm_provider"].lower() == "openrouter":
            _openrouter_key = os.getenv("OPENROUTER_API_KEY")
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"], api_key=_openrouter_key, temperature=0)
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"], api_key=_openrouter_key, temperature=0)
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

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic()

        # Determine attack types
        attack_enabled = self.attack_config.get("enabled", False)
        attack_types = self.attack_config.get("attack_types", [])
        if not attack_types and self.attack_config.get("attack_type"):
            attack_types = [self.attack_config["attack_type"]]

        # debate-specific attacks (persuasive_adversary / objective_hijacking / jailbreaking)
        # always route to AttackGraphSetup which hardwires the debate pipeline
        DEBATE_ONLY_ATTACKS = {"persuasive_adversary", "objective_hijacking", "jailbreaking"}
        uses_debate_attack = attack_enabled and bool(set(attack_types) & DEBATE_ONLY_ATTACKS)

        if uses_debate_attack:
            self.graph_setup = AttackGraphSetup(
                self.quick_thinking_llm,
                self.deep_thinking_llm,
                self.tool_nodes,
                self.bull_memory,
                self.bear_memory,
                self.trader_memory,
                self.invest_judge_memory,
                self.risk_manager_memory,
                self.conditional_logic,
                self.attack_config,
            )
        else:
            SetupClass = GRAPH_STRUCTURES.get(graph_structure, GraphSetup)
            self.graph_setup = SetupClass(
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

        # Build analyst overrides for analyst_compromise attack (works for all architectures).
        # Decentralized needs its VOTE_INSTRUCTION preserved on compromised analysts too -
        # otherwise their bias_suffix is fully replaced by the attack directive, they never
        # emit a "MY VOTE: ..." line, and the vote aggregator silently defaults them to HOLD.
        vote_extra_suffix = VOTE_INSTRUCTION if graph_structure == "decentralized" else ""
        analyst_overrides = build_analyst_overrides(
            self.attack_config, selected_analysts, self.quick_thinking_llm,
            extra_suffix=vote_extra_suffix,
        ) if attack_enabled and not uses_debate_attack else None

        # Build decision-maker override for decision_maker_hijacking attack
        dm_override = build_decision_maker_override(
            self.attack_config, graph_structure, self.deep_thinking_llm
        ) if attack_enabled and not uses_debate_attack else None

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(
            selected_analysts,
            analyst_overrides=analyst_overrides,
            decision_maker_override=dm_override,
        )

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # Social media tools - must match what create_social_media_analyst binds
                    get_social,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_sentiment,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
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

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
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
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
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
