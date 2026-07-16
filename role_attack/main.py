from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.analyst_selection import get_selected_analysts
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
import argparse

# 모든 출력 억제
warnings.filterwarnings('ignore')  # 경고 메시지 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 로그 억제

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4.1"  # Use a different model
config["quick_think_llm"] = "gpt-4.1-mini"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds
config["poisoned_news"] = False 

# Configure data vendors (default uses yfinance and alpha_vantage)
config["data_vendors"] = {
    "core_stock_apis": "local",           # Options: yfinance, alpha_vantage, local
    "technical_indicators": "local",      # Options: yfinance, alpha_vantage, local
    "fundamental_data": "local",     # Options: openai, alpha_vantage, local
    "news_data": "local",            # Options: openai, alpha_vantage, google, local
}

def process_ticker(args):
    """Process a single ticker for all trade dates."""
    ticker, trade_dates, config, progress_dict = args
    results = []
    errors = []
    total_dates = len(trade_dates)

    # Each worker needs its own TradingAgentsGraph instance
    ta = TradingAgentsGraph(debug=False, config=config, selected_analysts=get_selected_analysts(ticker))

    # Initialize progress
    progress_dict[ticker] = {'current': 0, 'total': total_dates, 'status': 'running', 'last_date': ''}

    for idx, trade_date in enumerate(trade_dates, 1):
        date_str = trade_date.strftime("%Y-%m-%d")

        try:
            _, decision = ta.propagate(ticker, date_str)
            results.append({
                "ticker": ticker,
                "date": date_str,
                "decision": decision,
                "status": "success"
            })
            status = "success"
        except Exception as e:
            error_trace = traceback.format_exc()
            errors.append({
                "ticker": ticker,
                "date": date_str,
                "error": str(e),
                "traceback": error_trace
            })
            results.append({
                "ticker": ticker,
                "date": date_str,
                "decision": f"ERROR: {str(e)}",
                "status": "error"
            })
            status = "error"

        # Update shared progress
        progress_dict[ticker] = {
            'current': idx,
            'total': total_dates,
            'status': 'running',
            'last_date': date_str,
            'last_status': status
        }

    # Mark as completed
    progress_dict[ticker] = {
        'current': total_dates,
        'total': total_dates,
        'status': 'completed',
        'last_date': date_str,
        'errors': len(errors)
    }

    return ticker, results, errors


def display_progress(progress_dict, tickers, total_dates):
    """Display unified progress bar for all tickers."""
    # Clear screen and move cursor to top
    print("\033[2J\033[H", end="")

    print("=" * 70)
    print(f"TRADING AGENT ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total_completed = 0
    total_tasks = len(tickers) * total_dates

    for ticker in tickers:
        if ticker in progress_dict:
            info = progress_dict[ticker]
            current = info['current']
            total = info['total']
            status = info['status']
            pct = current / total * 100 if total > 0 else 0

            # Progress bar
            bar_width = 30
            filled = int(bar_width * current / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_width - filled)

            # Status icon
            if status == 'completed':
                icon = "✓"
                color = "\033[92m"  # Green
            elif status == 'running':
                icon = "⏳"
                color = "\033[93m"  # Yellow
            else:
                icon = "○"
                color = "\033[90m"  # Gray

            last_date = info.get('last_date', '')
            errors = info.get('errors', 0)
            error_str = f" ({errors} errors)" if errors > 0 else ""

            print(f"{color}{icon} {ticker:6s} [{bar}] {current:3d}/{total:3d} ({pct:5.1f}%) {last_date}{error_str}\033[0m")
            total_completed += current
        else:
            print(f"\033[90m○ {ticker:6s} [{'░' * 30}]   0/{total_dates:3d} (  0.0%) waiting...\033[0m")

    # Overall progress
    overall_pct = total_completed / total_tasks * 100 if total_tasks > 0 else 0
    print("-" * 70)
    print(f"Overall: {total_completed}/{total_tasks} ({overall_pct:.1f}%)")
    print("=" * 70)


# ===== 설정 =====
# 분석할 ticker 리스트
tickers = ["AAPL", "AMZN", "GOOGL","META", "MSFT", "NVDA", "TSLA"]  # NVDA "TSLA", "AAPL"


# Alpha Arena
# tickers = ["TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "PLTR"] # NDX
# 10.19 ~ 11.3

# 병렬 처리 worker 수 (API rate limit 고려)
NUM_WORKERS = 3  # 2-3개 권장

# 기간 설정 (1년)
end_date = datetime(2025, 9, 30)
start_date = datetime(2025, 7, 1)

# NYSE 캘린더로 실제 거래일만 가져오기 (공휴일 자동 제외)
print("Fetching NYSE trading days (excluding weekends and holidays)...")
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
all_trade_dates = schedule.index

# 주 단위로 샘플링 (매주 첫 거래일)
# trade_dates = all_trade_dates[::5]  # 주 단위 (약 5 거래일 = 1주)
trade_dates = all_trade_dates
print(f"Found {len(all_trade_dates)} trading days, using {len(trade_dates)} days\n")

# ===== 병렬 실행 =====
print(f"Starting PARALLEL analysis for {len(tickers)} tickers over {len(trade_dates)} trading days")
print(f"Using {NUM_WORKERS} parallel workers")
print(f"Period: {start_date.date()} to {end_date.date()}\n")

# 결과 저장용
all_results = []
all_errors = []

# 총 작업 수 계산
total_tasks = len(tickers) * len(trade_dates)

# 병렬 처리
if __name__ == '__main__':
    # --llm-provider ollama: gpt-4.1/gpt-4.1-mini 대신 로컬 Ollama 서버의
    # Qwen3-32B로 실행 (사전에 `ollama pull qwen3:32b` 필요)
    parser = argparse.ArgumentParser(description="Run TradingAgents batch analysis")
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM backend to use. 'ollama' runs against a local Ollama server serving Qwen3-32B instead of the OpenAI API.",
    )
    cli_args = parser.parse_args()

    if cli_args.llm_provider == "ollama":
        config["llm_provider"] = "ollama"
        config["backend_url"] = "http://localhost:11434/v1"
        config["deep_think_llm"] = "qwen3:32b"
        config["quick_think_llm"] = "qwen3:32b"

    print(f"LLM provider: {config['llm_provider']} ({config['deep_think_llm']})\n")

    # 결과 디렉토리 생성 (날짜 기반 폴더 구조)
    date_range_str = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
    results_dir = os.path.join("trading_results", date_range_str)
    os.makedirs(results_dir, exist_ok=True)

    # 통계 추적용
    total_success = 0
    total_failed = 0
    completed_tickers = []

    # Shared progress dictionary
    manager = Manager()
    progress_dict = manager.dict()

    # Prepare arguments for each ticker
    args_list = [(ticker, trade_dates, config, progress_dict) for ticker in tickers]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all ticker jobs
        futures = {
            executor.submit(process_ticker, args): args[0]
            for args in args_list
        }

        # Progress monitoring loop with immediate result saving
        print("Starting analysis...\n")

        processed_futures = set()

        while len(processed_futures) < len(futures):
            # Display progress
            display_progress(progress_dict, tickers, len(trade_dates))

            # Check for newly completed futures and save immediately
            for future in futures:
                if future.done() and future not in processed_futures:
                    processed_futures.add(future)
                    ticker = futures[future]

                    try:
                        ticker_name, results, errors = future.result()

                        # 즉시 ticker별 결과 저장
                        if results:
                            ticker_df = pd.DataFrame(results)
                            ticker_df.to_csv(f"{results_dir}/{ticker_name}_results.csv", index=False)

                        # 즉시 ticker별 에러 저장
                        if errors:
                            ticker_errors_df = pd.DataFrame(errors)
                            ticker_errors_df.to_csv(f"{results_dir}/{ticker_name}_errors.csv", index=False)

                        # 통계 업데이트
                        all_results.extend(results)
                        all_errors.extend(errors)
                        completed_tickers.append(ticker_name)

                        success_count = len([r for r in results if r['status'] == 'success'])
                        error_count = len([r for r in results if r['status'] == 'error'])
                        total_success += success_count
                        total_failed += error_count

                    except Exception as e:
                        print(f"\n[ERROR] Worker failed for {ticker}: {str(e)}\n")

            time.sleep(3)  # Update every 3 secs

        # Final progress display
        display_progress(progress_dict, tickers, len(trade_dates))

    # ===== 전체 결과 병합 저장 =====
    print(f"\nMerging all results...")

    # 전체 결과를 하나의 파일로 저장
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv(f"{results_dir}/all_tickers_results.csv", index=False)

    # 전체 에러를 하나의 파일로 저장
    if all_errors:
        errors_df = pd.DataFrame(all_errors)
        errors_df.to_csv(f"{results_dir}/all_errors.csv", index=False)

    # ===== 최종 요약 =====
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Results directory: {results_dir}/")
    print(f"\nIndividual ticker files:")
    for ticker in completed_tickers:
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

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
