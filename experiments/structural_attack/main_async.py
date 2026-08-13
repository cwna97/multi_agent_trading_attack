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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# 모든 출력 억제
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4.1"
config["quick_think_llm"] = "gpt-4.1-mini"
config["max_debate_rounds"] = 1

# Configure data vendors
config["data_vendors"] = {
    "core_stock_apis": "local",
    "technical_indicators": "local",
    "fundamental_data": "local",
    "news_data": "local",
}

# Thread-local storage for TradingAgentsGraph instances
thread_local = threading.local()

def get_trading_graph():
    """Get or create a TradingAgentsGraph instance for the current thread."""
    if not hasattr(thread_local, 'ta'):
        thread_local.ta = TradingAgentsGraph(debug=False, config=config)
    return thread_local.ta


def process_single_task(task):
    """Process a single (ticker, date) task."""
    ticker, trade_date, progress_dict, lock = task
    date_str = trade_date.strftime("%Y-%m-%d")

    try:
        ta = get_trading_graph()
        _, decision = ta.propagate(ticker, date_str)
        result = {
            "ticker": ticker,
            "date": date_str,
            "decision": decision,
            "status": "success"
        }
        error = None
    except Exception as e:
        error_trace = traceback.format_exc()
        result = {
            "ticker": ticker,
            "date": date_str,
            "decision": f"ERROR: {str(e)}",
            "status": "error"
        }
        error = {
            "ticker": ticker,
            "date": date_str,
            "error": str(e),
            "traceback": error_trace
        }

    # Update progress (thread-safe)
    with lock:
        if ticker not in progress_dict:
            progress_dict[ticker] = {'completed': 0, 'success': 0, 'error': 0, 'last_date': ''}
        progress_dict[ticker]['completed'] += 1
        progress_dict[ticker]['last_date'] = date_str
        if result['status'] == 'success':
            progress_dict[ticker]['success'] += 1
        else:
            progress_dict[ticker]['error'] += 1

    return result, error


def display_progress_async(progress_dict, tickers, total_per_ticker, lock):
    """Display progress for async processing."""
    print("\033[2J\033[H", end="")

    print("=" * 70)
    print(f"TRADING AGENT ANALYSIS (ASYNC) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total_completed = 0
    total_tasks = len(tickers) * total_per_ticker

    with lock:
        for ticker in tickers:
            if ticker in progress_dict:
                info = progress_dict[ticker]
                current = info['completed']
                total = total_per_ticker
                pct = current / total * 100 if total > 0 else 0

                bar_width = 30
                filled = int(bar_width * current / total) if total > 0 else 0
                bar = "█" * filled + "░" * (bar_width - filled)

                if current == total:
                    icon = "✓"
                    color = "\033[92m"
                elif current > 0:
                    icon = "⏳"
                    color = "\033[93m"
                else:
                    icon = "○"
                    color = "\033[90m"

                last_date = info.get('last_date', '')
                errors = info.get('error', 0)
                error_str = f" ({errors} err)" if errors > 0 else ""

                print(f"{color}{icon} {ticker:6s} [{bar}] {current:3d}/{total:3d} ({pct:5.1f}%) {last_date}{error_str}\033[0m")
                total_completed += current
            else:
                print(f"\033[90m○ {ticker:6s} [{'░' * 30}]   0/{total_per_ticker:3d} (  0.0%) waiting...\033[0m")

    overall_pct = total_completed / total_tasks * 100 if total_tasks > 0 else 0
    print("-" * 70)
    print(f"Overall: {total_completed}/{total_tasks} ({overall_pct:.1f}%)")
    print("=" * 70)


# ===== 설정 =====
tickers = ["TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "PLTR"]

# 병렬 처리 worker 수 (ThreadPool - API 호출 위주이므로 더 많이 가능)
NUM_WORKERS = 10  # ThreadPool은 I/O bound 작업에 더 많은 worker 사용 가능

# 기간 설정
end_date = datetime(2025, 12, 31)
start_date = datetime(2025, 10, 1)

# NYSE 캘린더로 실제 거래일만 가져오기
print("Fetching NYSE trading days (excluding weekends and holidays)...")
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
all_trade_dates = schedule.index
trade_dates = all_trade_dates
print(f"Found {len(all_trade_dates)} trading days, using {len(trade_dates)} days\n")

# 총 작업 수 계산
total_tasks = len(tickers) * len(trade_dates)

if __name__ == '__main__':
    # 결과 디렉토리 생성
    date_range_str = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
    results_dir = os.path.join("trading_results", date_range_str)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Starting ASYNC analysis for {len(tickers)} tickers over {len(trade_dates)} trading days")
    print(f"Using {NUM_WORKERS} thread workers")
    print(f"Total tasks: {total_tasks}")
    print(f"Period: {start_date.date()} to {end_date.date()}\n")

    # Progress tracking
    progress_dict = {}
    lock = threading.Lock()

    # 모든 (ticker, date) 조합을 task로 생성
    tasks = []
    for ticker in tickers:
        for trade_date in trade_dates:
            tasks.append((ticker, trade_date, progress_dict, lock))

    all_results = []
    all_errors = []

    # ThreadPoolExecutor로 병렬 처리
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_task, task): task for task in tasks}

        print("Starting analysis...\n")

        # Progress monitoring in separate display loop
        completed_count = 0

        while completed_count < len(futures):
            # Display progress
            display_progress_async(progress_dict, tickers, len(trade_dates), lock)

            # Count completed
            completed_count = sum(1 for f in futures if f.done())

            # Collect results from completed futures
            for future in list(futures.keys()):
                if future.done() and future in futures:
                    try:
                        result, error = future.result()
                        all_results.append(result)
                        if error:
                            all_errors.append(error)
                    except Exception as e:
                        print(f"\n[ERROR] Task failed: {str(e)}\n")
                    del futures[future]

            time.sleep(1)

        # Final progress display
        display_progress_async(progress_dict, tickers, len(trade_dates), lock)

    # ===== 결과 저장 =====
    print(f"\nSaving results...")

    # 티커별 결과 저장
    results_df = pd.DataFrame(all_results)
    for ticker in tickers:
        ticker_results = results_df[results_df['ticker'] == ticker]
        if not ticker_results.empty:
            ticker_results.to_csv(f"{results_dir}/{ticker}_results.csv", index=False)

    # 전체 결과 저장
    if all_results:
        results_df.to_csv(f"{results_dir}/all_tickers_results.csv", index=False)

    # 에러 저장
    if all_errors:
        errors_df = pd.DataFrame(all_errors)
        errors_df.to_csv(f"{results_dir}/all_errors.csv", index=False)

    # ===== 최종 요약 =====
    total_success = len([r for r in all_results if r['status'] == 'success'])
    total_failed = len([r for r in all_results if r['status'] == 'error'])

    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Results directory: {results_dir}/")
    print(f"\nIndividual ticker files:")
    for ticker in tickers:
        print(f"  ✓ {ticker}_results.csv")
    print(f"\nCombined file:")
    print(f"  ✓ all_tickers_results.csv")
    print(f"\nTotal analyses: {len(all_results)}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_failed}")
    if all_errors:
        print(f"\n⚠️  {len(all_errors)} errors occurred")
        print(f"Error details: {results_dir}/all_errors.csv")
    print(f"{'='*50}")
