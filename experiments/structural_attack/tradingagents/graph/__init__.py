# TradingAgents/graph/__init__.py

from .trading_graph import TradingAgentsGraph
from .conditional_logic import ConditionalLogic
from .setup import GraphSetup                      # debate architecture
from .setup_linear import LinearGraphSetup
from .setup_centralized import CentralizedGraphSetup
from .setup_decentralized import DecentralizedGraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor

# Attack modules
from .attack_trading_graph import AttackTradingAgentsGraph

__all__ = [
    "TradingAgentsGraph",
    "ConditionalLogic",
    "GraphSetup",
    "LinearGraphSetup",
    "CentralizedGraphSetup",
    "DecentralizedGraphSetup",
    "Propagator",
    "Reflector",
    "SignalProcessor",
    # Attack modules
    "AttackTradingAgentsGraph",
]
