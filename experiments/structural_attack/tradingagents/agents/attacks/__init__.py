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

# Attack Analyst Compromise (works across all architectures)
from .attack_analyst_compromise import (
    create_compromised_analyst,
    build_analyst_overrides,
)

# D1/D2: News Text Poisoning
from .attack_news_text_poisoning import (
    prepare_text_poisoned_news,
    build_news_poison_config,
)

# A3: Decision-Maker Hijacking (linear/centralized/decentralized/hybrid)
from .attack_decision_maker_hijacking import (
    build_decision_maker_override,
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
    # Analyst Compromise (all architectures)
    "create_compromised_analyst",
    "build_analyst_overrides",
    # D1/D2: News Text Poisoning
    "prepare_text_poisoned_news",
    "build_news_poison_config",
    # A3: Decision-Maker Hijacking
    "build_decision_maker_override",
]
