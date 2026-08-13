# MAT-Attack — Poisoning Agentic Alpha

Code and data for the paper **"Poisoning Agentic Alpha: Adversarial Vulnerabilities
Across Roles and Architectures in Multi-Agent Trading Systems"** (ICAIF '26).

A research framework for studying the **robustness and vulnerability of multi-agent
LLM trading systems** under adversarial prompt-injection attacks. It is built on top of
the [TradingAgents](https://github.com/TauricResearch/TradingAgents) framework and adds
an attack layer, an evaluation pipeline, and a reproducible analysis notebook.

> ⚠️ **Research use only.** This code is intended for security and robustness research on
> LLM-based agent systems. It is **not** financial, investment, or trading advice, and the
> attack techniques must only be used against systems you own or are authorized to test.

---

## Overview

Multi-agent trading systems decompose decisions across specialized LLM agents (analysts,
researchers, a trader, and a risk-management team). Because these agents exchange
natural-language reports and consume untrusted external data (news, social media,
fundamentals), each hand-off is a potential injection point. Crucially, we adopt a
**democratized threat model**: the adversary is restricted to what is practically
reachable — the source data and prompts agents consume — with **no** access to model
weights, orchestration state, or reasoning traces.

We analyze adversarial failure along two complementary axes:

1. **Role axis** — which agent *role* an attack targets, and which attacks are viable given
   that role's task and interface.
2. **Architecture axis** — how far a compromised signal *survives* toward the final
   decision, which depends on the inter-agent communication topology.

A central finding is that **no architecture is inherently robust**: adversarial signals
survive deliberation and still reach the final trading decision.

## Experiments

The repository contains two self-contained experiment packages, each a fork of
TradingAgents with its own attack framework, data pipeline, and results.

### `experiments/role_attack/` — Role-targeted injection

Each functional role is paired with the attack that best fits its interface, and we measure
how often the final decision is flipped toward an attacker-chosen target (BUY / SELL).

| # | Attack | Target role | Level |
|---|--------|-------------|-------|
| 1 | Data Poisoning | Analyst (news / sentiment) | data-level |
| 2 | Indirect Prompt Injection | Analyst (social) | data-level |
| 3 | Persuasive Adversary | Researcher (bull/bear debate) | agent-level |
| 4 | Objective Hijacking | Trader | agent-level |
| 5 | Jailbreaking | Risk Manager | agent-level |

### `experiments/structural_attack/` — Topology & architecture

Runs the analyst-layer attacks across four representative communication topologies —
**Decentralized, Hybrid, Linear, Centralized** — and asks whether a structure's design
predicts its vulnerability. To interpret the ordering we use the **Adversarial Signal
Preservation Score (APS)** as a post-hoc analytical lens: a closed-form score for how much
of a compromised signal survives aggregation (Decentralized 0.00 · Hybrid 0.33 · Linear
0.50 · Centralized 1.00), from most robust to most vulnerable.

## Metrics

| Metric | Meaning |
|--------|---------|
| **ASR** — Attack Success Rate | Fraction of *attackable* days (whose clean decision ≠ target) that flip to the attacker's target |
| **CR** — Cumulative Return | Portfolio cumulative return; compared against the clean run to measure financial impact |

*(The structural analysis additionally reports **APS** as an analytical lens and **DCR** —
Decision Change Rate — the fraction of days whose decision differs from the clean run.)*

## Experimental scope

- **Assets (5):** AAPL, MSFT, NVDA, TSLA, BTC-USD
- **Period:** 2026 Q1 (Jan 1 – Mar 31), 61 trading days (N = 305 asset-days per configuration)
- **Backbones:** GPT-4.1 (+ `gpt-4.1-mini`) and Qwen3-235B-A22B (+ Qwen3-30B-A3B)
- **Directions:** BUY-targeted and SELL-targeted
- **Poison-ratio sweep:** 10 % (default) · 40 % · 80 %
- **Data sources:** Alpha Vantage & yfinance (prices/fundamentals/news), Reddit (social)

## Repository structure

```
mat_attack/
├── experiments/
│   ├── role_attack/             # Role axis — 5 role-matched attacks
│   │   ├── tradingagents/        #   TradingAgents package (LangGraph agents)
│   │   ├── cli/                  #   CLI entry point (python -m cli.main)
│   │   ├── adversarial_attack.py #   Attack framework (payloads, targets, goals)
│   │   ├── scripts/              #   Data prep + per-attack evaluation
│   │   └── .env.example
│   └── structural_attack/        # Architecture axis — 4 topologies + APS
│       ├── tradingagents/
│       ├── figures/  tables/     #   Pre-generated figures & tables
│       └── .env.example
└── analysis/                     # Aggregated run outputs (2026 Q1)
    ├── paper_results_analysis.ipynb   # ← reproduces the paper's tables & figures
    ├── role_attack/              #   clean / attack_results / poison_ratio
    └── structural_attack/        #   clean / attack_results  (GPT-4.1 & Qwen3)
```

## Reproducing the paper's results

[`analysis/paper_results_analysis.ipynb`](analysis/paper_results_analysis.ipynb) recomputes
the headline results **directly from the raw run outputs** in `analysis/` — no external state
required. It covers:

1. **Role-specific failure modes** — Table 1 (ASR by attack × direction) and a per-asset breakdown
2. **Poison-ratio sweep** — ASR at 10 / 40 / 80 %
3. **Architecture-level analysis** — Table 3 (ASR by topology) and the ASR-vs-APS ordering
4. A short summary of findings

```bash
cd analysis
jupyter lab paper_results_analysis.ipynb          # or open in VS Code
# only needs: pandas, numpy, matplotlib
```

## Setup

Each experiment is installed independently. Example for `role_attack` (same steps for
`structural_attack`):

```bash
cd experiments/role_attack

# 1. Create an environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env      # then edit .env with your own keys
```

### Required API keys

Set these in the experiment's `.env` file (never commit real keys — `.env` is git-ignored):

- `OPENAI_API_KEY` — LLM agents
- `ALPHA_VANTAGE_API_KEY` — fundamentals / news data
- `OPENROUTER_API_KEY` — optional, for models routed via OpenRouter (e.g. Qwen3)
- `FINNHUB_API_KEY` — optional, additional news data (structural experiment)

## Usage

Run the base trading agents interactively:

```bash
cd experiments/role_attack
python -m cli.main
```

Run an attack + evaluation (see each experiment's `scripts/` for the full pipeline):

```bash
python run_adversarial_attack.py     # execute attacks
python calculate_metrics.py          # compute ASR / DCR / finance metrics
```

## Responsible use

This framework demonstrates concrete attacks against LLM agent pipelines. All manipulated
content is synthetically generated and confined to the experimental environment (sandboxed
backtests, no live capital or market interaction). Use it only to evaluate and harden
systems you own or are authorized to test.

## Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{poisoning_agentic_alpha_2026,
  title     = {Poisoning Agentic Alpha: Adversarial Vulnerabilities Across Roles and
               Architectures in Multi-Agent Trading Systems},
  booktitle = {Proceedings of the ACM International Conference on AI in Finance (ICAIF '26)},
  year      = {2026}
}
```

## Acknowledgements

Built on the [TradingAgents](https://github.com/TauricResearch/TradingAgents) multi-agent
trading framework by Tauric Research:

```bibtex
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework},
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}
}
```
