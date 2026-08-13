"""
Attack Modules for Adversarial Experiments

This package contains various attack implementations for testing
the robustness of the Trading Agents system.

Attack Scenarios:
- Attack 01: Data Poisoning (Sentiment Data Manipulation)
- Attack 02: Indirect Prompt Injection (News Analyst)
- Attack 03: Persuasive Adversary (Bull/Bear Researcher)
- Attack 04: Objective Hijacking (Trader Agent)
- Attack 05: [Reserved for future]
- Attack 06: [Reserved for future]
"""

# Attack 01: Data Poisoning - Sentiment Data Manipulation
from .attack_01_data_poisoning import (
    DataPoisoningAttack,
    AttackResult,
    PoisonedNewsDataLoader,
    create_poisoned_news_getter,
)

# Attack 03: Persuasive Adversary
from .attack_03_persuasive_adversary import (
    create_adversary_bull_researcher,
    create_adversary_bear_researcher,
)

# Attack 04: Objective Hijacking
from .attack_04_objective_hijacking import (
    create_hijacked_trader,
)

__all__ = [
    # Attack 01
    "DataPoisoningAttack",
    "AttackResult",
    "PoisonedNewsDataLoader",
    "create_poisoned_news_getter",
    # Attack 03
    "create_adversary_bull_researcher",
    "create_adversary_bear_researcher",
    # Attack 04
    "create_hijacked_trader",
]
