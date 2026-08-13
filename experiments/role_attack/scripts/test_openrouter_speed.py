"""
Quick benchmark: run one ticker/date through the full TradingAgentsGraph
pipeline via OpenRouter, with optional throughput-based provider routing,
and report wall-clock time + which upstream provider actually served the
chat completions.

This exists so speed comparisons (e.g. "does provider routing actually help
the real pipeline, not just a single raw API call?") don't have to be
one-off `python -c "..."` snippets - run it, get a timed result, done.

Usage:
    python scripts/test_openrouter_speed.py
    python scripts/test_openrouter_speed.py --ticker MSFT --date 2026-01-07
    python scripts/test_openrouter_speed.py --no-routing   # compare against OpenRouter's own default routing
    python scripts/test_openrouter_speed.py --max-price-prompt 0.2 --max-price-completion 1.0
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import LOCAL_ONLY_CONFIG


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark one ticker/date through the full pipeline via OpenRouter.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker to run (default: AAPL)")
    parser.add_argument("--date", type=str, default="2026-01-07", help="Trade date, YYYY-MM-DD (default: 2026-01-07)")
    parser.add_argument(
        "--model", type=str, default="qwen/qwen3-235b-a22b-2507",
        help="OpenRouter model id (default: qwen/qwen3-235b-a22b-2507)",
    )
    parser.add_argument("--max-price-prompt", type=float, default=0.15, help="Max $/M prompt tokens (default: 0.15)")
    parser.add_argument("--max-price-completion", type=float, default=0.8, help="Max $/M completion tokens (default: 0.8)")
    parser.add_argument(
        "--no-routing", action="store_true",
        help="Skip throughput-based provider routing; let OpenRouter use its own default selection instead.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: OPENROUTER_API_KEY not set in .env")
        sys.exit(1)

    config = dict(LOCAL_ONLY_CONFIG)
    config["llm_provider"] = "openrouter"
    config["backend_url"] = "https://openrouter.ai/api/v1"
    config["llm_api_key"] = key
    config["deep_think_llm"] = args.model
    config["quick_think_llm"] = args.model

    if not args.no_routing:
        config["openrouter_provider_routing"] = {
            "sort": "throughput",
            "max_price": {"prompt": args.max_price_prompt, "completion": args.max_price_completion},
        }

    print("=" * 70)
    print(f"Ticker: {args.ticker}  Date: {args.date}  Model: {args.model}")
    if args.no_routing:
        print("Provider routing: OFF (OpenRouter default)")
    else:
        print(f"Provider routing: sort=throughput, max_price=(prompt<=${args.max_price_prompt}/M, "
              f"completion<=${args.max_price_completion}/M)")
    print("=" * 70)

    t0 = time.time()
    ta = TradingAgentsGraph(debug=False, config=config)
    final_state, decision = ta.propagate(args.ticker, args.date)
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"RESULT: {decision}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
