import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.1-2025-11-13",
    "quick_think_llm": "gpt-5-mini-2025-08-07",
    "backend_url": "https://api.openai.com/v1",
    # "backend_url": api_key,
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: yfinance, alpha_vantage, local
        "technical_indicators": "yfinance",  # Options: yfinance, alpha_vantage, local
        "fundamental_data": "alpha_vantage", # Options: openai, alpha_vantage, local
        "news_data": "google",               # Options: openai, alpha_vantage, google, local
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
        # Example: "get_news": "openai",               # Override category default
    },
}

# Local-only configuration (uses prefetched data from local_data directory)
# Run scripts/prefetch_data.py first to download data
LOCAL_ONLY_CONFIG = {
    **DEFAULT_CONFIG,
    "data_vendors": {
        "core_stock_apis": "local",         # Uses local_data/market_data/price_data/
        "technical_indicators": "local",    # Uses local_data/indicators/
        "fundamental_data": "local",        # Uses local_data/fundamental_data/{ticker}/
        "news_data": "local",               # Uses local_data/news_data/
    },
}
