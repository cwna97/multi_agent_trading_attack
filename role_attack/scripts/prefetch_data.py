#!/usr/bin/env python3
"""
Prefetch and cache all data for specified tickers.
Downloads stock data, technical indicators, fundamentals, and news.

Usage:
    uv run python scripts/prefetch_data.py
"""

import os
import sys
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import yfinance as yf
from stockstats import wrap
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.googlenews_utils import getNewsData
from tradingagents.dataflows.config import get_config

# ===== Configuration =====
TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "TSLA", "GOOGL", "META"]
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

# Technical indicators to prefetch
INDICATORS = [
    "close_50_sma", "close_200_sma", "close_10_ema",
    "macd", "macds", "macdh",
    "rsi",
    "boll", "boll_ub", "boll_lb",
    "atr", "vwma", "mfi"
]


def get_local_data_dir():
    """Get or create local_data directory."""
    config = get_config()
    base_dir = config.get("data_cache_dir", "tradingagents/dataflows/data_cache")
    local_dir = os.path.join(os.path.dirname(base_dir), "local_data")
    os.makedirs(local_dir, exist_ok=True)
    return local_dir


def prefetch_stock_data(ticker: str, start_date: str, end_date: str, data_dir: str) -> bool:
    """
    Download and save stock price data (OHLCV).

    Saves to: local_data/market_data/price_data/{ticker}-YFin-data-{start}-{end}.csv
    """
    tqdm.write(f"  [Stock Data] Downloading {ticker}...")

    try:
        # Extend date range for technical indicator calculation
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        extended_start = (start_dt - relativedelta(years=1)).strftime("%Y-%m-%d")

        data = yf.download(
            ticker,
            start=extended_start,
            end=end_date,
            progress=False,
            auto_adjust=True
        )

        if data.empty:
            tqdm.write(f"    WARNING: No data found for {ticker}")
            return False

        # Reset index to have Date as column
        data = data.reset_index()

        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]

        # Save to CSV
        save_dir = os.path.join(data_dir, "market_data", "price_data")
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{ticker}-YFin-data-{extended_start}-{end_date}.csv"
        filepath = os.path.join(save_dir, filename)
        data.to_csv(filepath, index=False)

        tqdm.write(f"    ✓ Saved {len(data)} rows to {filename}")
        return True

    except Exception as e:
        tqdm.write(f"    ERROR: {e}")
        return False


def prefetch_technical_indicators(ticker: str, start_date: str, end_date: str, data_dir: str) -> bool:
    """
    Calculate and save technical indicators.

    Saves to: local_data/indicators/{ticker}_indicators.json
    """
    tqdm.write(f"  [Indicators] Calculating for {ticker}...")

    try:
        # Load stock data first
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        extended_start = (start_dt - relativedelta(years=1)).strftime("%Y-%m-%d")

        price_file = os.path.join(
            data_dir, "market_data", "price_data",
            f"{ticker}-YFin-data-{extended_start}-{end_date}.csv"
        )

        if not os.path.exists(price_file):
            tqdm.write(f"    WARNING: Price data not found, skipping indicators")
            return False

        data = pd.read_csv(price_file)
        data["Date"] = pd.to_datetime(data["Date"])

        # Wrap with stockstats
        df = wrap(data)

        # Calculate all indicators
        indicators_data = {}
        for indicator in INDICATORS:
            try:
                df[indicator]  # Trigger calculation
            except Exception as e:
                tqdm.write(f"    WARNING: Could not calculate {indicator}: {e}")

        # Convert to date-indexed dict
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        for _, row in df.iterrows():
            date_str = row["Date"]
            if date_str < start_date or date_str > end_date:
                continue

            indicators_data[date_str] = {}
            for indicator in INDICATORS:
                if indicator in df.columns:
                    val = row[indicator]
                    indicators_data[date_str][indicator] = None if pd.isna(val) else float(val)

        # Save to JSON
        save_dir = os.path.join(data_dir, "indicators")
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{ticker}_indicators.json"
        filepath = os.path.join(save_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(indicators_data, f, indent=2)

        tqdm.write(f"    ✓ Saved {len(indicators_data)} days of indicators to {filename}")
        return True

    except Exception as e:
        tqdm.write(f"    ERROR: {e}")
        return False


def prefetch_fundamentals(ticker: str, data_dir: str) -> bool:
    """
    Download and save fundamental data (balance sheet, cash flow, income statement).

    Saves to: local_data/fundamental_data/{ticker}/
    """
    tqdm.write(f"  [Fundamentals] Downloading for {ticker}...")

    try:
        ticker_obj = yf.Ticker(ticker)
        save_dir = os.path.join(data_dir, "fundamental_data", ticker)
        os.makedirs(save_dir, exist_ok=True)

        saved_files = []

        # Balance Sheet
        for freq, suffix in [("quarterly", "quarterly"), ("annual", "annual")]:
            try:
                if freq == "quarterly":
                    bs = ticker_obj.quarterly_balance_sheet
                else:
                    bs = ticker_obj.balance_sheet

                if not bs.empty:
                    filepath = os.path.join(save_dir, f"balance_sheet_{suffix}.csv")
                    bs.to_csv(filepath)
                    saved_files.append(f"balance_sheet_{suffix}")
            except Exception as e:
                tqdm.write(f"    WARNING: balance_sheet_{suffix}: {e}")

        # Cash Flow
        for freq, suffix in [("quarterly", "quarterly"), ("annual", "annual")]:
            try:
                if freq == "quarterly":
                    cf = ticker_obj.quarterly_cashflow
                else:
                    cf = ticker_obj.cashflow

                if not cf.empty:
                    filepath = os.path.join(save_dir, f"cashflow_{suffix}.csv")
                    cf.to_csv(filepath)
                    saved_files.append(f"cashflow_{suffix}")
            except Exception as e:
                tqdm.write(f"    WARNING: cashflow_{suffix}: {e}")

        # Income Statement
        for freq, suffix in [("quarterly", "quarterly"), ("annual", "annual")]:
            try:
                if freq == "quarterly":
                    inc = ticker_obj.quarterly_income_stmt
                else:
                    inc = ticker_obj.income_stmt

                if not inc.empty:
                    filepath = os.path.join(save_dir, f"income_statement_{suffix}.csv")
                    inc.to_csv(filepath)
                    saved_files.append(f"income_statement_{suffix}")
            except Exception as e:
                tqdm.write(f"    WARNING: income_statement_{suffix}: {e}")

        # Insider Transactions
        try:
            insider = ticker_obj.insider_transactions
            if insider is not None and not insider.empty:
                filepath = os.path.join(save_dir, "insider_transactions.csv")
                insider.to_csv(filepath, index=False)
                saved_files.append("insider_transactions")
        except Exception as e:
            tqdm.write(f"    WARNING: insider_transactions: {e}")

        tqdm.write(f"    ✓ Saved {len(saved_files)} files for {ticker}")
        return True

    except Exception as e:
        tqdm.write(f"    ERROR: {e}")
        return False


def prefetch_news(ticker: str, start_date: str, end_date: str, data_dir: str) -> bool:
    """
    Download and save news data from Google News.

    Saves to: local_data/news_data/{ticker}_news.json
    """
    tqdm.write(f"  [News] Downloading for {ticker}...")

    try:
        # Fetch news (day by day via gnews)
        news_results = getNewsData(ticker, start_date, end_date, fetch_full_article=False)

        if not news_results:
            tqdm.write(f"    WARNING: No news found for {ticker}")
            return False

        # Organize by date
        news_by_date = {}
        for article in news_results:
            date_str = article.get("date", "unknown")
            if date_str not in news_by_date:
                news_by_date[date_str] = []
            news_by_date[date_str].append({
                "headline": article.get("title", ""),
                "summary": article.get("snippet", ""),
                "source": article.get("source", ""),
                "link": article.get("link", ""),
            })

        # Save to JSON
        save_dir = os.path.join(data_dir, "news_data")
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{ticker}_news.json"
        filepath = os.path.join(save_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(news_by_date, f, ensure_ascii=False, indent=2)

        tqdm.write(f"    ✓ Saved {len(news_results)} articles ({len(news_by_date)} days) to {filename}")
        return True

    except Exception as e:
        tqdm.write(f"    ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("Data Prefetch Script")
    print("=" * 60)
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Period: {START_DATE} to {END_DATE}")

    data_dir = get_local_data_dir()
    print(f"Data directory: {data_dir}")
    print("=" * 60)

    results = {
        "stock_data": {},
        "indicators": {},
        "fundamentals": {},
        "news": {}
    }

    # Data types to process
    data_types = ["stock_data", "indicators", "fundamentals", "news"]

    # Total tasks = tickers * data_types
    total_tasks = len(TICKERS) * len(data_types)

    # Single progress bar with detailed description
    pbar = tqdm(total=total_tasks, desc="Starting...", unit="task")

    for ticker_idx, ticker in enumerate(TICKERS):
        for dtype_idx, dtype in enumerate(data_types):
            # Update progress bar description
            pbar.set_description(f"{ticker} [{dtype_idx+1}/4] {dtype}")

            if dtype == "stock_data":
                results["stock_data"][ticker] = prefetch_stock_data(
                    ticker, START_DATE, END_DATE, data_dir
                )
            elif dtype == "indicators":
                results["indicators"][ticker] = prefetch_technical_indicators(
                    ticker, START_DATE, END_DATE, data_dir
                )
            elif dtype == "fundamentals":
                results["fundamentals"][ticker] = prefetch_fundamentals(ticker, data_dir)
            elif dtype == "news":
                results["news"][ticker] = prefetch_news(ticker, START_DATE, END_DATE, data_dir)

            pbar.update(1)

        # Show ticker completion
        tqdm.write(f"  ✓ {ticker} completed ({ticker_idx+1}/{len(TICKERS)})")

    pbar.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for category, ticker_results in results.items():
        success = sum(1 for v in ticker_results.values() if v)
        total = len(ticker_results)
        status = "✓" if success == total else "!"
        print(f"  {status} {category}: {success}/{total} successful")

    print(f"\nData saved to: {data_dir}/")
    print("Directory structure:")
    print("  local_data/")
    print("    ├── market_data/price_data/    # Stock OHLCV data")
    print("    ├── indicators/                 # Technical indicators")
    print("    ├── fundamental_data/{ticker}/  # Financial statements")
    print("    └── news_data/                  # News articles")

    print("\nDone!")


if __name__ == "__main__":
    main()
