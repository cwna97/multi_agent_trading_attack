# TradingAgents/graph/analyst_selection.py

DEFAULT_ANALYSTS = ["market", "social", "news", "fundamentals"]

# Tickers with no company fundamentals available (e.g. cryptocurrencies)
NO_FUNDAMENTALS_TICKERS = {"BTC-USD"}


def get_selected_analysts(ticker, base_analysts=DEFAULT_ANALYSTS):
    """Return the analyst team for a ticker, dropping fundamentals when unavailable."""
    analysts = list(base_analysts)
    if ticker in NO_FUNDAMENTALS_TICKERS and "fundamentals" in analysts:
        analysts.remove("fundamentals")
    return analysts
