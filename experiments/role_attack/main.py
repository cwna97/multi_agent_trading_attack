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
from concurrent.futures import ProcessPoolExecutor, CancelledError
from multiprocessing import Manager
import time
import argparse

from resume_utils import filter_pending_dates, load_prior_success, merge_with_prior
from llm_provider_cli import add_llm_provider_args, apply_llm_provider

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
    "social_data": "local",          # Options: local
}

def _is_quota_exhausted(exc: Exception) -> bool:
    """
    True if `exc` signals a permanent, account-wide API budget failure
    (OpenAI's "insufficient_quota") rather than a transient/retryable error
    like a per-request rate limit or a one-off timeout. Retrying or continuing
    to the next date cannot help once this happens - every subsequent call
    will fail identically until billing is fixed.
    """
    code = getattr(exc, "code", None)
    if code == "insufficient_quota":
        return True
    # Fall back to message matching: LangChain may wrap the raw openai
    # exception, and older/other SDKs may not expose `.code`.
    text = str(exc).lower()
    return "insufficient_quota" in text or "exceeded your current quota" in text


def process_ticker(args):
    """Process a single ticker for all trade dates."""
    ticker, trade_dates, config, progress_dict, stop_event = args
    results = []
    errors = []
    total_dates = len(trade_dates)

    # Each worker needs its own TradingAgentsGraph instance
    ta = TradingAgentsGraph(debug=False, config=config, selected_analysts=get_selected_analysts(ticker))

    # Initialize progress
    progress_dict[ticker] = {'current': 0, 'total': total_dates, 'status': 'running', 'last_date': ''}

    date_str = ""
    for idx, trade_date in enumerate(trade_dates, 1):
        if stop_event.is_set():
            # Another worker already hit insufficient_quota - no point burning
            # through the rest of this ticker's dates with the same failure.
            break

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

            if _is_quota_exhausted(e):
                stop_event.set()
                progress_dict[ticker] = {
                    'current': idx,
                    'total': total_dates,
                    'status': 'stopped (quota exhausted)',
                    'last_date': date_str,
                    'last_status': status,
                    'errors': len(errors),
                }
                return ticker, results, errors

        # Update shared progress
        progress_dict[ticker] = {
            'current': idx,
            'total': total_dates,
            'status': 'running',
            'last_date': date_str,
            'last_status': status
        }

    # Mark as completed (or stopped early because of another ticker's quota failure)
    progress_dict[ticker] = {
        'current': len(results),
        'total': total_dates,
        'status': 'stopped (quota exhausted)' if stop_event.is_set() else 'completed',
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


# 병렬 처리
if __name__ == '__main__':
    # --llm-provider ollama: gpt-4.1/gpt-4.1-mini 대신 로컬 Ollama 서버의
    # Qwen3-32B로 실행 (사전에 `ollama pull qwen3:32b` 필요)
    # --llm-provider openrouter/gemini/qwen: OpenRouter API로 실행
    # (사전에 .env에 OPENROUTER_API_KEY=sk-or-... 추가 필요)
    parser = argparse.ArgumentParser(description="Run TradingAgents batch analysis (clean/baseline, no attack)")
    add_llm_provider_args(parser)
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        default=["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"],
        help="List of tickers to analyze (default: AAPL AMZN GOOGL META MSFT NVDA TSLA)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-07-01",
        help="Start date in YYYY-MM-DD format (default: 2025-07-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-09-30",
        help="End date in YYYY-MM-DD format (default: 2025-09-30)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3, 2-3 recommended for API rate limits)",
    )
    cli_args = parser.parse_args()

    apply_llm_provider(config, cli_args)

    # ===== 설정 =====
    tickers = cli_args.tickers
    NUM_WORKERS = cli_args.workers
    start_date = datetime.strptime(cli_args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(cli_args.end_date, "%Y-%m-%d")

    # NYSE 캘린더로 실제 거래일만 가져오기 (공휴일 자동 제외)
    print("Fetching NYSE trading days (excluding weekends and holidays)...")
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    all_trade_dates = schedule.index
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

    # 결과 디렉토리 생성 (날짜 기반 폴더 구조)
    date_range_str = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
    results_dir = os.path.join("trading_results", date_range_str)
    os.makedirs(results_dir, exist_ok=True)

    # ===== Resume: skip (ticker, date) pairs already completed successfully =====
    # (e.g. a previous run of this exact command that stopped on insufficient_quota)
    print("Checking for prior partial results to resume from...")
    date_by_str = {d.strftime("%Y-%m-%d"): d for d in trade_dates}
    ticker_date_strs = {ticker: list(date_by_str.keys()) for ticker in tickers}
    ticker_date_strs = filter_pending_dates(ticker_date_strs, results_dir, suffix="_results.csv")
    ticker_trade_dates = {
        ticker: [date_by_str[s] for s in strs] for ticker, strs in ticker_date_strs.items()
    }

    # 통계 추적용
    total_success = 0
    total_failed = 0
    completed_tickers = []

    # Shared progress dictionary + cross-worker "stop everything" signal
    manager = Manager()
    progress_dict = manager.dict()
    stop_event = manager.Event()

    # Tickers with nothing left to do (fully completed in a prior partial run)
    # don't need a worker - fold their saved rows straight into this run's output.
    for ticker in tickers:
        if not ticker_trade_dates.get(ticker):
            _, prior_rows = load_prior_success(results_dir, ticker, suffix="_results.csv")
            if prior_rows:
                print(f"  {ticker}: already fully completed in a prior run - skipping")
                all_results.extend(prior_rows)
                completed_tickers.append(ticker)
                progress_dict[ticker] = {
                    'current': len(prior_rows), 'total': len(prior_rows),
                    'status': 'completed', 'last_date': prior_rows[-1]['date'], 'errors': 0,
                }

    # Prepare arguments for each ticker
    args_list = [
        (ticker, ticker_trade_dates[ticker], config, progress_dict, stop_event)
        for ticker in tickers if ticker_trade_dates[ticker]
    ]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all ticker jobs
        futures = {
            executor.submit(process_ticker, args): args[0]
            for args in args_list
        }

        # Progress monitoring loop with immediate result saving
        print("Starting analysis...\n")

        processed_futures = set()
        quota_stop_announced = False

        while len(processed_futures) < len(futures):
            # Display progress
            display_progress(progress_dict, tickers, len(trade_dates))

            if stop_event.is_set():
                # A worker hit insufficient_quota. Cancel every ticker job that
                # hasn't started yet - no point spending 3+ more workers' worth
                # of dates confirming the same permanent failure.
                if not quota_stop_announced:
                    quota_stop_announced = True
                    print("\n" + "!" * 70)
                    print("STOPPING: OpenAI API quota exhausted (insufficient_quota).")
                    print("Cancelling all not-yet-started ticker jobs. Already-running")
                    print("workers will finish their current in-flight date and stop.")
                    print("!" * 70 + "\n")
                for future in futures:
                    if not future.done():
                        future.cancel()

            # Check for newly completed futures and save immediately
            for future in futures:
                if future.done() and future not in processed_futures:
                    processed_futures.add(future)
                    ticker = futures[future]

                    try:
                        ticker_name, results, errors = future.result()

                        # 이전 partial run에서 저장된 성공 row와 병합 후 저장
                        combined_results = merge_with_prior(results_dir, ticker_name, results, suffix="_results.csv")
                        if combined_results:
                            ticker_df = pd.DataFrame(combined_results)
                            ticker_df.to_csv(f"{results_dir}/{ticker_name}_results.csv", index=False)

                        # 즉시 ticker별 에러 저장
                        if errors:
                            ticker_errors_df = pd.DataFrame(errors)
                            ticker_errors_df.to_csv(f"{results_dir}/{ticker_name}_errors.csv", index=False)

                        # 통계 업데이트
                        all_results.extend(combined_results)
                        all_errors.extend(errors)
                        completed_tickers.append(ticker_name)

                        success_count = len([r for r in results if r['status'] == 'success'])
                        error_count = len([r for r in results if r['status'] == 'error'])
                        total_success += success_count
                        total_failed += error_count

                    except CancelledError:
                        print(f"\n[CANCELLED] {ticker} was cancelled before it started (quota exhausted).\n")
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
    if stop_event.is_set():
        print(f"\n{'!'*50}")
        print(f"RUN STOPPED EARLY: OpenAI API quota exhausted (insufficient_quota)")
        print(f"Not all tickers/dates were processed. Results above are partial.")
        print(f"{'!'*50}")
    if all_errors:
        print(f"\n⚠️  {len(all_errors)} errors occurred")
        print(f"Error details: {results_dir}/all_errors.csv")
    print(f"{'='*50}")

    if stop_event.is_set():
        sys.exit(1)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
