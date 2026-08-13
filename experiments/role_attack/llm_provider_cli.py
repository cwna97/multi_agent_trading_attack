"""
Shared --llm-provider CLI wiring for main.py and every scripts/run_attack_0N_*.py.

Single source of truth for how each script picks its backbone LLM, so the
clean baseline (main.py) and all 5 attack runners always support the same
set of providers/models. Add a new backbone here once instead of editing
six files.

Usage in a script:
    from llm_provider_cli import add_llm_provider_args, apply_llm_provider

    parser = argparse.ArgumentParser(...)
    add_llm_provider_args(parser)
    ... (script's own --tickers/--start-date/etc.) ...
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1
    config["data_vendors"] = {...}
    apply_llm_provider(config, args)   # sets llm_provider/backend_url/deep_think_llm/quick_think_llm
"""

import os
import sys

# Named OpenRouter presets so `--llm-provider gemini` / `--llm-provider qwen`
# behave like first-class backbones instead of making every caller remember
# and pass a raw OpenRouter model id via --openrouter-model each time.
PRESET_OPENROUTER_MODELS = {
    "gemini": "google/gemini-2.5-flash",
    "qwen":   "qwen/qwen3-235b-a22b-2507",
}

# Per-model OpenRouter list price (as of setup time) so the throughput-routing
# price cap actually admits at least one endpoint. Qwen3-235B is far cheaper
# than Gemini 2.5 Flash, so a single fixed cap tuned for one starves the other
# (routing then finds "no endpoints" and every call 404s).
PRESET_MAX_PRICE = {
    "qwen":   {"prompt": 0.15, "completion": 0.8},
    "gemini": {"prompt": 0.5,  "completion": 3.0},
}
_DEFAULT_MAX_PRICE = {"prompt": 0.15, "completion": 0.8}  # generic --llm-provider openrouter fallback


def add_llm_provider_args(parser):
    """Add --llm-provider and its OpenRouter-specific options to `parser`."""
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "ollama", "openrouter", "gemini", "qwen"],
        default="openai",
        help="LLM backend to use. 'openai' = gpt-4.1 / gpt-4.1-mini (default). "
             "'ollama' = local Ollama server serving Qwen3-32B. "
             "'openrouter' = OpenRouter with an arbitrary model via --openrouter-model. "
             "'gemini' = OpenRouter preset for google/gemini-2.5-flash. "
             "'qwen' = OpenRouter preset for qwen/qwen3-235b-a22b-2507 (Qwen3-235B-A22B-Instruct-2507).",
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="qwen/qwen3-235b-a22b-2507",
        help="OpenRouter model id to use for both deep/quick thinking when "
             "--llm-provider openrouter (ignored for the gemini/qwen presets; "
             "default: qwen/qwen3-235b-a22b-2507)",
    )
    parser.add_argument(
        "--openrouter-max-price-prompt",
        type=float,
        default=None,
        help="Max $/M prompt tokens for OpenRouter provider routing "
             "(default: preset per-model value, e.g. 0.15 for qwen, 0.5 for gemini)",
    )
    parser.add_argument(
        "--openrouter-max-price-completion",
        type=float,
        default=None,
        help="Max $/M completion tokens for OpenRouter provider routing "
             "(default: preset per-model value, e.g. 0.8 for qwen, 3.0 for gemini)",
    )
    parser.add_argument(
        "--openrouter-no-throughput-routing",
        action="store_true",
        help="Disable throughput-based provider routing and just let OpenRouter pick its own "
             "default (by default, routes to the fastest provider under the price cap above).",
    )
    return parser


def apply_llm_provider(config: dict, cli_args) -> None:
    """Mutate `config` in place to match cli_args.llm_provider.

    Exits the process with an error message if a required API key is missing.
    """
    provider = cli_args.llm_provider

    if provider == "openai":
        config["llm_provider"] = "openai"
        config["backend_url"] = "https://api.openai.com/v1"
        config["deep_think_llm"] = "gpt-4.1"
        config["quick_think_llm"] = "gpt-4.1-mini"
        print(f"LLM provider: openai ({config['deep_think_llm']} / {config['quick_think_llm']})\n")
        return

    if provider == "ollama":
        config["llm_provider"] = "ollama"
        config["backend_url"] = "http://localhost:11434/v1"
        config["deep_think_llm"] = "qwen3:32b"
        config["quick_think_llm"] = "qwen3:32b"
        print(f"LLM provider: ollama ({config['deep_think_llm']})\n")
        return

    # openrouter / gemini / qwen all ride OpenRouter's OpenAI-compatible endpoint.
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        print(f"ERROR: --llm-provider {provider} requires OPENROUTER_API_KEY in .env (or the environment).")
        print("Get a key at https://openrouter.ai/settings/keys and add:")
        print("  OPENROUTER_API_KEY=sk-or-v1-...")
        sys.exit(1)

    model = PRESET_OPENROUTER_MODELS.get(provider, cli_args.openrouter_model)

    config["llm_provider"] = "openrouter"
    config["backend_url"] = "https://openrouter.ai/api/v1"
    config["llm_api_key"] = openrouter_key
    config["deep_think_llm"] = model
    config["quick_think_llm"] = model

    if not cli_args.openrouter_no_throughput_routing:
        preset_price = PRESET_MAX_PRICE.get(provider, _DEFAULT_MAX_PRICE)
        max_price_prompt = (
            cli_args.openrouter_max_price_prompt
            if cli_args.openrouter_max_price_prompt is not None
            else preset_price["prompt"]
        )
        max_price_completion = (
            cli_args.openrouter_max_price_completion
            if cli_args.openrouter_max_price_completion is not None
            else preset_price["completion"]
        )
        config["openrouter_provider_routing"] = {
            "sort": "throughput",
            "max_price": {
                "prompt": max_price_prompt,
                "completion": max_price_completion,
            },
        }
        print(
            f"OpenRouter provider routing: sort=throughput, "
            f"max_price=(prompt<=${max_price_prompt}/M, "
            f"completion<=${max_price_completion}/M)"
        )

    print(f"LLM provider: {provider} → openrouter ({config['deep_think_llm']})\n")
