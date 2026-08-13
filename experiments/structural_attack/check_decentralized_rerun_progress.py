"""
qwen/gemini/gpt decentralized 재실행 배치 진행률 체크.

Usage:
    .venv/bin/python3 check_decentralized_rerun_progress.py
"""
import json
from pathlib import Path

LOGS = Path(__file__).parent / "logs"

EXPERIMENTS = {
    "qwen": [
        "qwen_q1_decentralized",
        "adv_qwen_q1_D1_buy_decentralized",
        "adv_qwen_q1_D2_buy_decentralized",
        "adv_qwen_q1_A1_buy_decentralized",
        "adv_qwen_q1_A2s_buy_decentralized",
        "qwen_q3_decentralized",
        "adv_qwen_q3_D1_sell_decentralized",
        "adv_qwen_q3_D2_sell_decentralized",
        "adv_qwen_q3_A1_sell_decentralized",
        "adv_qwen_q3_A1o_sell_decentralized",
        "adv_qwen_q3_A2_sell_decentralized",
        "adv_qwen_q3_A2s_sell_decentralized",
    ],
    "gemini": [
        "gemini_q1_decentralized",
        "adv_gemini_q1_D1_buy_decentralized",
        "adv_gemini_q1_D2_buy_decentralized",
        "adv_gemini_q1_A1_buy_decentralized",
        "adv_gemini_q1_A2s_buy_decentralized",
        "gemini_q3_decentralized",
        "adv_gemini_q3_D1_sell_decentralized",
        "adv_gemini_q3_D2_sell_decentralized",
        "adv_gemini_q3_A1_sell_decentralized",
        "adv_gemini_q3_A1o_sell_decentralized",
        "adv_gemini_q3_A2_sell_decentralized",
        "adv_gemini_q3_A2s_sell_decentralized",
    ],
    "gpt": [
        "gpt-4.1_decentralized",
        "adv_gpt_D1_buy_decentralized",
        "adv_gpt_D2_buy_decentralized",
        "adv_gpt_A1_buy_decentralized",
        "adv_gpt_A2_buy_decentralized",
        "adv_gpt_A1overt_buy_decentralized",
        "adv_gpt_A2subtle_buy_decentralized",
        "gpt-4.1_q3_decentralized",
        "adv_gpt_q3_D1_sell_decentralized",
        "adv_gpt_q3_D2_sell_decentralized",
        "adv_gpt_q3_A1_sell_decentralized",
        "adv_gpt_q3_A1o_sell_decentralized",
        "adv_gpt_q3_A2_sell_decentralized",
        "adv_gpt_q3_A2s_sell_decentralized",
    ],
}


def status_for(name: str) -> str:
    pfile = LOGS / f".progress_{name}.json"
    if not pfile.exists():
        return "⏳ 대기중 (아직 시작 안 함)"
    try:
        data = json.loads(pfile.read_text())
    except Exception:
        return "⚠ progress 파일 읽기 오류"

    tickers = data.get("tickers", {})
    total_dates = data.get("total_dates", 1)
    done = sum(v.get("current", 0) for v in tickers.values())
    total = total_dates * len(tickers) if tickers else 0
    pct = done / total * 100 if total else 0
    updated = data.get("updated_at", "")

    statuses = {v.get("status", "") for v in tickers.values()}
    flag = ""
    if "budget_stopped" in statuses:
        flag = "  <-- BUDGET STOPPED!"
    elif len(tickers) and all(v.get("status") == "completed" for v in tickers.values()):
        flag = "  ✓ DONE"

    return f"{done:4d}/{total:4d} ({pct:5.1f}%)  updated={updated}{flag}"


def main():
    for provider, names in EXPERIMENTS.items():
        print(f"\n=== {provider} ===")
        for name in names:
            print(f"  {name:45s} {status_for(name)}")


if __name__ == "__main__":
    main()
