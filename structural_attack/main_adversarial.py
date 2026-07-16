"""
Adversarial version of main.py

run_adversarial_experiments.py에서 환경변수로 attack_config를 주입받아
TradingAgentsGraph를 adversarial 모드로 실행.

환경 변수 (추가):
    ATTACK_ENABLED       : "true" | "false"
    ATTACK_TYPE          : D1=news_text_poison | D2=news_inject |
                           A1/A2=analyst_compromise | A3=decision_maker_hijacking
    ATTACK_SIGNAL        : BUY | SELL | HOLD
    ATTACK_ID            : D1 | D2 | A1 | A2 | A3
    ATTACK_POISON_MODE   : content | inject  (D1/D2용)
    ATTACK_POISON_RATE   : float  (D1/D2용)
    ATTACK_BIAS_STYLE    : subtle | overt  (A1/A2용)
    ATTACK_NUM_COMPROMISED: int  (A1/A2용)
    ATTACK_HIJACK_MODE   : forced | biased  (A3용)
"""

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.attacks.attack_news_text_poisoning import prepare_text_poisoned_news
from tradingagents.utils.cost_tracker import CostTracker
from langchain_community.callbacks import get_openai_callback
from tradingagents.utils.gemini_callback import get_gemini_callback
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import pandas_market_calendars as mcal
import traceback
import sys
import os
import warnings
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import time

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

load_dotenv()

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME      = os.environ.get("MODEL_NAME",        "adv_experiment")
DEEP_THINK      = os.environ.get("DEEP_THINK_LLM",    "qwen2.5:72b")
QUICK_THINK     = os.environ.get("QUICK_THINK_LLM",   "qwen2.5:14b")
BACKEND_URL     = os.environ.get("BACKEND_URL",        "http://localhost:11434/v1")
LLM_PROVIDER    = os.environ.get("LLM_PROVIDER",      "ollama")
START_DATE_STR  = os.environ.get("START_DATE",        "2026-01-01")
END_DATE_STR    = os.environ.get("END_DATE",          "2026-03-31")
GRAPH_STRUCTURE = os.environ.get("GRAPH_STRUCTURE",   "linear")
SELECTED_ANALYSTS = os.environ.get("SELECTED_ANALYSTS", "market,social,news").split(",")
RESUME          = os.environ.get("RESUME", "false").lower() == "true"
MAX_BUDGET_USD  = float(os.environ.get("MAX_BUDGET_USD", "0"))   # 0 = 무제한

# ---------------------------------------------------------------------------
# Attack config from env
# ---------------------------------------------------------------------------
ATTACK_ENABLED  = os.environ.get("ATTACK_ENABLED", "false").lower() == "true"
ATTACK_TYPE     = os.environ.get("ATTACK_TYPE",    "")
ATTACK_SIGNAL   = os.environ.get("ATTACK_SIGNAL",  "SELL")
ATTACK_ID       = os.environ.get("ATTACK_ID",      "")

attack_config = {}
if ATTACK_ENABLED and ATTACK_TYPE:
    attack_config = {
        "enabled":      True,
        "attack_types": [ATTACK_TYPE],
        "attack_type":  ATTACK_TYPE,
        "target_signal": ATTACK_SIGNAL,
        # D1/D2
        "poison_mode":  os.environ.get("ATTACK_POISON_MODE", "content"),
        "poison_rate":  float(os.environ.get("ATTACK_POISON_RATE", "1.0")),
        # A1/A2
        "bias_style":   os.environ.get("ATTACK_BIAS_STYLE",  "subtle"),
        "num_compromised": int(os.environ.get("ATTACK_NUM_COMPROMISED", "1")),
        # A3
        "hijack_mode":  os.environ.get("ATTACK_HIJACK_MODE", "forced"),
        "attack_name":  f"{ATTACK_ID}_{ATTACK_TYPE}_{ATTACK_SIGNAL.lower()}",
    }

# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------
config = DEFAULT_CONFIG.copy()
config["llm_provider"]     = LLM_PROVIDER
config["backend_url"]      = BACKEND_URL
config["deep_think_llm"]   = DEEP_THINK
config["quick_think_llm"]  = QUICK_THINK
config["max_debate_rounds"] = 1
config["poisoned_news"]    = False
config["social_source"]    = os.environ.get("SOCIAL_SOURCE", "social_data_filled")

# news_source: D1/D2는 poisoned 디렉토리로, 나머지는 기본값
BASE_NEWS_SOURCE = os.environ.get("NEWS_SOURCE", "alpha_vantage_filled")
config["news_source"] = BASE_NEWS_SOURCE

config["data_vendors"] = {
    "core_stock_apis":      "local",
    "technical_indicators": "local",
    "fundamental_data":     "local",
    "news_data":            "local",
}

_tickers_env = os.environ.get("TICKERS", "")
tickers = _tickers_env.split(",") if _tickers_env else ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]

# ---------------------------------------------------------------------------
# D1/D2: Pre-generate poisoned news files before process_ticker forks
# SKIP_NEWS_REGEN=true → 파일 생성 생략 (병렬 실행 시 race condition 방지용)
# ---------------------------------------------------------------------------
NEWS_TEXT_ATTACK_TYPES = {"news_text_poison", "news_inject", "news_content_poison", "prompt_inject"}
SKIP_NEWS_REGEN = os.environ.get("SKIP_NEWS_REGEN", "false").lower() == "true"

if ATTACK_ENABLED and ATTACK_TYPE in NEWS_TEXT_ATTACK_TYPES:
    sig  = ATTACK_SIGNAL.upper()
    mode = attack_config["poison_mode"]
    if SKIP_NEWS_REGEN:
        # 파일은 이미 runner가 사전 생성 → 이름만 구성
        poisoned_source = (
            f"{BASE_NEWS_SOURCE}_text_poison_{sig.lower()}"
            if mode == "content"
            else f"{BASE_NEWS_SOURCE}_prompt_inject_{sig.lower()}"
        )
        print(f"[Attack {ATTACK_ID}] Skipping news regen → using: {poisoned_source}")
    else:
        print(f"[Attack {ATTACK_ID}] Generating poisoned news data...")
        local_data_dir = os.path.join(SCRIPT_DIR, "tradingagents/dataflows/local_data")
        poisoned_source = prepare_text_poisoned_news(
            tickers=tickers,
            target_signal=ATTACK_SIGNAL,
            poison_mode=mode,
            poison_rate=attack_config["poison_rate"],
            src_news_source=BASE_NEWS_SOURCE,
            local_data_dir=local_data_dir,
        )
    config["news_source"] = poisoned_source
    print(f"[Attack {ATTACK_ID}] news_source set to: {poisoned_source}")


# ---------------------------------------------------------------------------
# Process ticker (identical to main.py)
# ---------------------------------------------------------------------------

def process_ticker(args):
    ticker, trade_dates, config, attack_config, progress_dict, skip_dates, done_count = args
    results = []
    errors = []
    total_dates = len(trade_dates)

    # RESUME 모드: 이미 성공한 날짜는 건너뜀
    pending_dates = [d for d in trade_dates if d.strftime("%Y-%m-%d") not in skip_dates]

    if ticker == "BTC-USD":
        analysts = [a for a in SELECTED_ANALYSTS if a != "fundamentals"]
    else:
        analysts = SELECTED_ANALYSTS

    ta = TradingAgentsGraph(
        debug=False,
        config=config,
        attack_config=attack_config,
        graph_structure=GRAPH_STRUCTURE,
        selected_analysts=analysts,
    )

    track_cost = LLM_PROVIDER.lower() in ("openai", "openrouter", "google")
    _get_callback = get_gemini_callback if LLM_PROVIDER.lower() == "google" else get_openai_callback
    _cost_log = Path(SCRIPT_DIR) / "logs" / (
        "cost_log_gemini.json" if LLM_PROVIDER.lower() == "google" else "cost_log.json"
    )
    cost_tracker = CostTracker(MODEL_NAME, max_budget_usd=MAX_BUDGET_USD, log_path=_cost_log) if track_cost else None

    progress_dict[ticker] = {'current': done_count, 'total': total_dates, 'status': 'running', 'last_date': ''}

    date_str = ""
    budget_stopped = False
    for trade_date in pending_dates:
        date_str = trade_date.strftime("%Y-%m-%d")

        # 예산 선제 확인 (기록 없이 파일만 읽음)
        if cost_tracker and cost_tracker.check_budget():
            print(f"\n[Budget] ⚠ 예산 한도 ${MAX_BUDGET_USD:.2f} 초과 — {ticker} 중단")
            budget_stopped = True
            break

        try:
            if track_cost:
                with _get_callback() as cb:
                    _, decision = ta.propagate(ticker, date_str)
                over_budget = cost_tracker.record(cb, ticker=ticker, trade_date=date_str)
                day_cost    = cb.total_cost
                day_tokens  = cb.total_tokens
                if over_budget:
                    print(f"\n[Budget] ⚠ 예산 한도 ${MAX_BUDGET_USD:.2f} 초과 — 다음 티커부터 중단")
                    budget_stopped = True
            else:
                _, decision = ta.propagate(ticker, date_str)
                day_cost   = 0.0
                day_tokens = 0
                over_budget = False

            results.append({
                "ticker": ticker, "date": date_str,
                "decision": decision, "status": "success",
                "attack_id": ATTACK_ID, "attack_signal": ATTACK_SIGNAL,
            })
            status = "success"
        except Exception as e:
            error_trace = traceback.format_exc()
            errors.append({
                "ticker": ticker, "date": date_str,
                "error": str(e), "traceback": error_trace,
            })
            results.append({
                "ticker": ticker, "date": date_str,
                "decision": f"ERROR: {str(e)}", "status": "error",
                "attack_id": ATTACK_ID, "attack_signal": ATTACK_SIGNAL,
            })
            status    = "error"
            day_cost  = 0.0
            day_tokens = 0
            over_budget = False

        done_count += 1
        progress_dict[ticker] = {
            'current':      done_count,
            'total':        total_dates,
            'status':       'running',
            'last_date':    date_str,
            'last_status':  status,
            'day_cost':     round(day_cost, 6),
            'day_tokens':   day_tokens,
            'session_cost': round(cost_tracker.session_cost if cost_tracker else 0.0, 6),
        }

        if budget_stopped:
            break

    progress_dict[ticker] = {
        'current': total_dates, 'total': total_dates,
        'status': 'completed' if not budget_stopped else 'budget_stopped',
        'last_date': date_str, 'errors': len(errors),
    }
    return ticker, results, errors


def display_progress(progress_dict, tickers, total_dates):
    from tradingagents.utils.cost_tracker import CostTracker
    print("\033[2J\033[H", end="")
    print("=" * 70)
    print(f"ADVERSARIAL ANALYSIS [{ATTACK_ID}|{ATTACK_TYPE}|{ATTACK_SIGNAL}|{GRAPH_STRUCTURE}]")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    total_completed = 0
    total_tasks = len(tickers) * total_dates
    for ticker in tickers:
        if ticker in progress_dict:
            info         = progress_dict[ticker]
            current      = info['current']
            total        = info['total']
            status       = info['status']
            day_cost     = info.get('day_cost', 0.0)
            session_cost = info.get('session_cost', 0.0)
            day_tokens   = info.get('day_tokens', 0)
            pct    = current / total * 100 if total > 0 else 0
            filled = int(30 * current / total) if total > 0 else 0
            bar    = "█" * filled + "░" * (30 - filled)
            color  = "\033[92m" if status == 'completed' else "\033[93m" if status == 'running' else "\033[90m"
            icon   = "✓" if status == 'completed' else "⏳" if status == 'running' else "○"
            errors = info.get('errors', 0)
            err_str  = f" ({errors} err)" if errors else ""
            cost_str = f"  last=${day_cost:.4f} tok={day_tokens:,}  Σ=${session_cost:.4f}" if session_cost > 0 else ""
            print(f"{color}{icon} {ticker:7s} [{bar}] {current:3d}/{total:3d} ({pct:5.1f}%){err_str}{cost_str}\033[0m")
            total_completed += current
        else:
            print(f"\033[90m○ {ticker:7s} [{'░'*30}]   0/{total_dates:3d} (  0.0%) waiting...\033[0m")
    overall_pct  = total_completed / total_tasks * 100 if total_tasks > 0 else 0
    total_cost   = CostTracker.get_total_cost()
    budget_str   = ""
    if MAX_BUDGET_USD > 0:
        remaining  = max(MAX_BUDGET_USD - total_cost, 0)
        budget_str = f"  예산 잔여: ${remaining:.4f} / ${MAX_BUDGET_USD:.2f}"
    print("-" * 70)
    print(f"Overall: {total_completed}/{total_tasks} ({overall_pct:.1f}%)")
    if total_cost > 0:
        print(f"Cost:    누적 총 ${total_cost:.4f}{budget_str}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print(f"Fetching NYSE trading days...")
start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
end_date   = datetime.strptime(END_DATE_STR,   "%Y-%m-%d")
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
trade_dates = schedule.index
print(f"Found {len(trade_dates)} trading days\n")

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "3"))

if __name__ == '__main__':
    date_range_str = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
    results_dir = os.path.join("trading_results", date_range_str, MODEL_NAME)
    os.makedirs(results_dir, exist_ok=True)

    total_success = 0
    total_failed  = 0
    completed_tickers = []
    all_results = []
    all_errors  = []

    # RESUME: 기존 성공 결과 로드 → 완료된 날짜 skip
    existing_results = {}   # ticker -> list of result dicts
    skip_dates_map   = {}   # ticker -> set of date strings (success + error)
    done_count_map   = {}   # ticker -> int (success only)

    if RESUME:
        print("[RESUME] 기존 결과 파일 확인 중...")
        for ticker in tickers:
            csv_path = os.path.join(results_dir, f"{ticker}_results.csv")
            err_path = os.path.join(results_dir, f"{ticker}_errors.csv")
            skip_dates = set()

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                success_rows = df[df["status"] == "success"]
                success_dates = set(success_rows["date"].astype(str).tolist())
                skip_dates |= success_dates
                existing_results[ticker] = success_rows.to_dict("records")
                done_count_map[ticker]   = len(success_dates)
                print(f"  {ticker}: {done_count_map[ticker]}일 성공 완료 → skip")
            else:
                existing_results[ticker] = []
                done_count_map[ticker]   = 0

            if os.path.exists(err_path):
                try:
                    edf = pd.read_csv(err_path)
                    error_dates = set(edf["date"].astype(str).tolist())
                    new_errors  = error_dates - skip_dates
                    skip_dates |= new_errors
                    if new_errors:
                        print(f"  {ticker}: {len(new_errors)}일 이전 에러 → skip (재시도 원하면 {ticker}_errors.csv 삭제)")
                except Exception:
                    pass

            skip_dates_map[ticker] = skip_dates

    manager = Manager()
    progress_dict = manager.dict()

    args_list = [
        (
            ticker, trade_dates, config, attack_config, progress_dict,
            skip_dates_map.get(ticker, set()),
            done_count_map.get(ticker, 0),
        )
        for ticker in tickers
    ]

    progress_file = os.path.join(SCRIPT_DIR, f"logs/.progress_{MODEL_NAME}.json")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_ticker, args): args[0] for args in args_list}
        processed_futures = set()

        while len(processed_futures) < len(futures):
            display_progress(progress_dict, tickers, len(trade_dates))

            try:
                snapshot = {
                    "model": MODEL_NAME,
                    "structure": GRAPH_STRUCTURE,
                    "attack_id": ATTACK_ID,
                    "attack_type": ATTACK_TYPE,
                    "attack_signal": ATTACK_SIGNAL,
                    "tickers": {
                        t: dict(progress_dict[t]) if t in progress_dict
                        else {"current": 0, "total": len(trade_dates), "status": "waiting", "last_date": ""}
                        for t in tickers
                    },
                    "total_dates": len(trade_dates),
                    "updated_at": datetime.now().strftime("%H:%M:%S"),
                }
                os.makedirs(os.path.dirname(progress_file), exist_ok=True)
                with open(progress_file, "w") as pf:
                    json.dump(snapshot, pf)
            except Exception:
                pass

            for future in futures:
                if future.done() and future not in processed_futures:
                    processed_futures.add(future)
                    ticker = futures[future]
                    try:
                        ticker_name, results, errors = future.result()
                        # RESUME: 기존 성공 결과와 병합
                        if RESUME and existing_results.get(ticker_name):
                            results = existing_results[ticker_name] + results
                        if results:
                            pd.DataFrame(results).to_csv(
                                f"{results_dir}/{ticker_name}_results.csv", index=False)
                        if errors:
                            pd.DataFrame(errors).to_csv(
                                f"{results_dir}/{ticker_name}_errors.csv", index=False)
                        all_results.extend(results)
                        all_errors.extend(errors)
                        completed_tickers.append(ticker_name)
                        total_success += len([r for r in results if r['status'] == 'success'])
                        total_failed  += len([r for r in results if r['status'] == 'error'])
                    except Exception as e:
                        print(f"\n[ERROR] Worker failed for {ticker}: {e}\n")

            time.sleep(3)

        display_progress(progress_dict, tickers, len(trade_dates))

    if all_results:
        pd.DataFrame(all_results).to_csv(f"{results_dir}/all_tickers_results.csv", index=False)
    if all_errors:
        pd.DataFrame(all_errors).to_csv(f"{results_dir}/all_errors.csv", index=False)

    print(f"\n{'='*50}")
    print(f"ADVERSARIAL RESULTS [{ATTACK_ID}|{ATTACK_SIGNAL}|{GRAPH_STRUCTURE}]")
    print(f"{'='*50}")
    print(f"Results directory: {results_dir}/")
    print(f"Total analyses: {len(all_results)}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_failed}")
    if all_errors:
        print(f"⚠ {len(all_errors)} errors — see {results_dir}/all_errors.csv")
    if LLM_PROVIDER.lower() in ("openai", "openrouter"):
        from tradingagents.utils.cost_tracker import CostTracker
        CostTracker.print_summary()
    print(f"{'='*50}")
