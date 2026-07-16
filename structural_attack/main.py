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
warnings.filterwarnings('ignore')  # 경고 메시지 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 로그 억제

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

# 환경 변수로 모델/구조 설정 (run_experiments.py에서 주입)
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME      = os.environ.get("MODEL_NAME",       "qwen2.5_72b")
DEEP_THINK      = os.environ.get("DEEP_THINK_LLM",   "qwen2.5:72b")
QUICK_THINK     = os.environ.get("QUICK_THINK_LLM",  "qwen2.5:14b")
BACKEND_URL     = os.environ.get("BACKEND_URL",      "http://localhost:11434/v1")
LLM_PROVIDER    = os.environ.get("LLM_PROVIDER",     "ollama")
START_DATE_STR  = os.environ.get("START_DATE",       "2025-01-01")
END_DATE_STR    = os.environ.get("END_DATE",         "2025-03-31")
GRAPH_STRUCTURE    = os.environ.get("GRAPH_STRUCTURE",  "linear")   # linear | centralized | decentralized | moa | debate
# analysts: market, social, news, fundamentals (BTC-USD는 fundamentals 제외)
SELECTED_ANALYSTS  = os.environ.get("SELECTED_ANALYSTS", "market,social,news").split(",")
RESUME             = os.environ.get("RESUME", "false").lower() == "true"

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = LLM_PROVIDER
config["backend_url"] = BACKEND_URL
config["deep_think_llm"] = DEEP_THINK
config["quick_think_llm"] = QUICK_THINK
config["max_debate_rounds"] = 1
config["poisoned_news"] = False
config["news_source"]   = os.environ.get("NEWS_SOURCE",   "alpha_vantage_filled")
config["social_source"] = os.environ.get("SOCIAL_SOURCE", "social_data_filled")

# Configure data vendors (default uses yfinance and alpha_vantage)
config["data_vendors"] = {
    "core_stock_apis": "local",           # Options: yfinance, alpha_vantage, local
    "technical_indicators": "local",      # Options: yfinance, alpha_vantage, local
    "fundamental_data": "local",     # Options: openai, alpha_vantage, local
    "news_data": "local",            # Options: openai, alpha_vantage, google, local
}

def process_ticker(args):
    """Process a single ticker for all trade dates."""
    ticker, trade_dates, config, progress_dict, skip_dates, done_count = args
    results = []
    errors = []
    total_dates = len(trade_dates)

    # RESUME: skip already-succeeded dates
    pending_dates = [d for d in trade_dates if d.strftime("%Y-%m-%d") not in skip_dates]

    # BTC-USD has no fundamentals data — use per-ticker analyst selection
    if ticker == "BTC-USD":
        analysts = [a for a in SELECTED_ANALYSTS if a != "fundamentals"]
    else:
        analysts = SELECTED_ANALYSTS

    # Each worker needs its own TradingAgentsGraph instance
    ta = TradingAgentsGraph(debug=False, config=config, graph_structure=GRAPH_STRUCTURE,
                            selected_analysts=analysts)

    # Initialize progress
    progress_dict[ticker] = {'current': done_count, 'total': total_dates, 'status': 'running', 'last_date': ''}

    for idx, trade_date in enumerate(pending_dates, done_count + 1):
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
# 분석할 ticker 리스트 (TICKERS 환경변수로 override 가능)
_TICKERS_ENV = os.environ.get("TICKERS", "")
tickers = _TICKERS_ENV.split(",") if _TICKERS_ENV else ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]


#  Alpha Arena
# tickers = ["TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "PLTR"] # NDX
# 10.19 ~ 11.3

# 병렬 처리 worker 수 (API rate limit 고려)
NUM_WORKERS = 1  # Ollama는 단일 GPU에서 직렬 처리, 병렬 시 큐잉 효율 저하

# 기간 설정
start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
end_date   = datetime.strptime(END_DATE_STR,   "%Y-%m-%d")

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
    # 결과 디렉토리 생성 (날짜 기반 폴더 구조)
    date_range_str = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
    results_dir = os.path.join("trading_results", date_range_str, MODEL_NAME)
    os.makedirs(results_dir, exist_ok=True)

    # 통계 추적용
    total_success = 0
    total_failed = 0
    completed_tickers = []

    # RESUME: 기존 성공 결과 로드 → 완료된 날짜 skip
    existing_results = {}
    skip_dates_map   = {}
    done_count_map   = {}

    if RESUME:
        print("[RESUME] 기존 결과 파일 확인 중...")
        for ticker in tickers:
            csv_path = os.path.join(results_dir, f"{ticker}_results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                success_rows = df[df["status"] == "success"]
                skip_dates_map[ticker]   = set(success_rows["date"].astype(str).tolist())
                existing_results[ticker] = success_rows.to_dict("records")
                done_count_map[ticker]   = len(skip_dates_map[ticker])
                print(f"  {ticker}: {done_count_map[ticker]}일 이미 완료 → skip")
            else:
                skip_dates_map[ticker]   = set()
                existing_results[ticker] = []
                done_count_map[ticker]   = 0

    # Shared progress dictionary
    manager = Manager()
    progress_dict = manager.dict()

    # Prepare arguments for each ticker
    args_list = [
        (
            ticker, trade_dates, config, progress_dict,
            skip_dates_map.get(ticker, set()),
            done_count_map.get(ticker, 0),
        )
        for ticker in tickers
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

        # 진행률 JSON 파일 경로 (run_experiments.py가 읽어 터미널 표시)
        progress_file = os.path.join(SCRIPT_DIR, f"logs/.progress_{MODEL_NAME}.json")

        while len(processed_futures) < len(futures):
            # Display progress (로그 파일용)
            display_progress(progress_dict, tickers, len(trade_dates))

            # 진행률을 JSON 파일에 기록 (터미널 모니터링용)
            try:
                snapshot = {
                    "model": MODEL_NAME,
                    "structure": GRAPH_STRUCTURE,
                    "tickers": {
                        t: dict(progress_dict[t]) if t in progress_dict else {"current": 0, "total": len(trade_dates), "status": "waiting", "last_date": ""}
                        for t in tickers
                    },
                    "total_dates": len(trade_dates),
                    "updated_at": datetime.now().strftime("%H:%M:%S"),
                }
                with open(progress_file, "w") as pf:
                    import json as _json
                    _json.dump(snapshot, pf)
            except Exception:
                pass

            # Check for newly completed futures and save immediately
            for future in futures:
                if future.done() and future not in processed_futures:
                    processed_futures.add(future)
                    ticker = futures[future]

                    try:
                        ticker_name, results, errors = future.result()

                        # RESUME: 기존 성공 결과와 병합
                        if RESUME and existing_results.get(ticker_name):
                            results = existing_results[ticker_name] + results

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
# 
