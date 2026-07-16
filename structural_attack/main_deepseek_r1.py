from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import pandas_market_calendars as mcal
import traceback
import sys
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import time

# 모든 출력 억제
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

load_dotenv()

# ===== 모델 설정 =====
MODEL_NAME = "deepseek-r1"
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "ollama"
config["backend_url"] = "http://localhost:11434/v1"
config["deep_think_llm"] = "deepseek-r1:70b"
config["quick_think_llm"] = "qwen2.5:14b"  # quick_think은 동일하게 유지 (변수 통제)
config["max_debate_rounds"] = 1
config["poisoned_news"] = False

config["data_vendors"] = {
    "core_stock_apis": "local",
    "technical_indicators": "local",
    "fundamental_data": "local",
    "news_data": "local",
}

def process_ticker(args):
    """Process a single ticker for all trade dates."""
    ticker, trade_dates, config, progress_dict = args
    results = []
    errors = []
    total_dates = len(trade_dates)

    ta = TradingAgentsGraph(debug=False, config=config)
    progress_dict[ticker] = {'current': 0, 'total': total_dates, 'status': 'running', 'last_date': ''}

    for idx, trade_date in enumerate(trade_dates, 1):
        date_str = trade_date.strftime("%Y-%m-%d")
        try:
            _, decision = ta.propagate(ticker, date_str)
            results.append({"ticker": ticker, "date": date_str, "decision": decision, "status": "success"})
            status = "success"
        except Exception as e:
            error_trace = traceback.format_exc()
            errors.append({"ticker": ticker, "date": date_str, "error": str(e), "traceback": error_trace})
            results.append({"ticker": ticker, "date": date_str, "decision": f"ERROR: {str(e)}", "status": "error"})
            status = "error"

        progress_dict[ticker] = {
            'current': idx, 'total': total_dates, 'status': 'running',
            'last_date': date_str, 'last_status': status
        }

    progress_dict[ticker] = {
        'current': total_dates, 'total': total_dates,
        'status': 'completed', 'last_date': date_str, 'errors': len(errors)
    }
    return ticker, results, errors


def display_progress(progress_dict, tickers, total_dates):
    print("\033[2J\033[H", end="")
    print("=" * 70)
    print(f"[{MODEL_NAME}] TRADING AGENT ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total_completed = 0
    total_tasks = len(tickers) * total_dates

    for ticker in tickers:
        if ticker in progress_dict:
            info = progress_dict[ticker]
            current, total, status = info['current'], info['total'], info['status']
            pct = current / total * 100 if total > 0 else 0
            bar_width = 30
            filled = int(bar_width * current / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_width - filled)
            if status == 'completed':
                icon, color = "✓", "\033[92m"
            elif status == 'running':
                icon, color = "⏳", "\033[93m"
            else:
                icon, color = "○", "\033[90m"
            last_date = info.get('last_date', '')
            errors = info.get('errors', 0)
            error_str = f" ({errors} errors)" if errors > 0 else ""
            print(f"{color}{icon} {ticker:6s} [{bar}] {current:3d}/{total:3d} ({pct:5.1f}%) {last_date}{error_str}\033[0m")
            total_completed += current
        else:
            print(f"\033[90m○ {ticker:6s} [{'░' * 30}]   0/{total_dates:3d} (  0.0%) waiting...\033[0m")

    overall_pct = total_completed / total_tasks * 100 if total_tasks > 0 else 0
    print("-" * 70)
    print(f"Overall: {total_completed}/{total_tasks} ({overall_pct:.1f}%)")
    print("=" * 70)


# ===== 실험 설정 (Qwen Q3와 동일 조건) =====
tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "PLTR", "TSLA"]
NUM_WORKERS = 1
end_date = datetime(2025, 3, 30)
start_date = datetime(2025, 1, 1)

print("Fetching NYSE trading days (excluding weekends and holidays)...")
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
all_trade_dates = schedule.index
trade_dates = all_trade_dates
print(f"Found {len(all_trade_dates)} trading days, using {len(trade_dates)} days\n")

print(f"[{MODEL_NAME}] deep_think={config['deep_think_llm']} / quick_think={config['quick_think_llm']}")
print(f"Starting analysis for {len(tickers)} tickers over {len(trade_dates)} trading days")
print(f"Using {NUM_WORKERS} parallel workers")
print(f"Period: {start_date.date()} to {end_date.date()}\n")

all_results = []
all_errors = []
total_tasks = len(tickers) * len(trade_dates)

if __name__ == '__main__':
    date_range_str = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
    results_dir = os.path.join("trading_results", f"{MODEL_NAME}_{date_range_str}")
    os.makedirs(results_dir, exist_ok=True)

    total_success = 0
    total_failed = 0
    completed_tickers = []

    manager = Manager()
    progress_dict = manager.dict()
    args_list = [(ticker, trade_dates, config, progress_dict) for ticker in tickers]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_ticker, args): args[0] for args in args_list}
        print("Starting analysis...\n")
        processed_futures = set()

        while len(processed_futures) < len(futures):
            display_progress(progress_dict, tickers, len(trade_dates))

            for future in futures:
                if future.done() and future not in processed_futures:
                    processed_futures.add(future)
                    ticker = futures[future]
                    try:
                        ticker_name, results, errors = future.result()
                        if results:
                            pd.DataFrame(results).to_csv(f"{results_dir}/{ticker_name}_results.csv", index=False)
                        if errors:
                            pd.DataFrame(errors).to_csv(f"{results_dir}/{ticker_name}_errors.csv", index=False)
                        all_results.extend(results)
                        all_errors.extend(errors)
                        completed_tickers.append(ticker_name)
                        total_success += len([r for r in results if r['status'] == 'success'])
                        total_failed += len([r for r in results if r['status'] == 'error'])
                    except Exception as e:
                        print(f"\n[ERROR] Worker failed for {ticker}: {str(e)}\n")

            time.sleep(3)

        display_progress(progress_dict, tickers, len(trade_dates))

    print(f"\nMerging all results...")
    if all_results:
        pd.DataFrame(all_results).to_csv(f"{results_dir}/all_tickers_results.csv", index=False)
    if all_errors:
        pd.DataFrame(all_errors).to_csv(f"{results_dir}/all_errors.csv", index=False)

    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY [{MODEL_NAME}]")
    print(f"{'='*50}")
    print(f"Results directory: {results_dir}/")
    print(f"\nTotal analyses: {len(all_results)}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_failed}")
    if all_errors:
        print(f"\n⚠️  {len(all_errors)} errors occurred")
    print(f"{'='*50}")
