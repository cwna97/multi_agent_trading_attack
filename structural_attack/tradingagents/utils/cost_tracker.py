"""
실험 비용 추적기 (OpenAI API 사용량 기반)

get_openai_callback()으로 수집한 토큰 사용량을 logs/cost_log.json에 누적 기록.
실험 간 공유 상태는 파일 기반(flock으로 동시 쓰기 방지).

Usage:
    from langchain_community.callbacks import get_openai_callback
    from tradingagents.utils.cost_tracker import CostTracker

    tracker = CostTracker(experiment_name="gpt-4.1_hybrid", max_budget_usd=300.0)

    with get_openai_callback() as cb:
        result = graph.invoke(...)

    over_budget = tracker.record(cb)   # True이면 즉시 중단 권장
"""

import json
import fcntl
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

# logs/ 는 스크립트 루트 기준
_DEFAULT_LOG = Path(__file__).parent.parent.parent / "logs" / "cost_log.json"


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    if not path.exists():
        return {"total_cost_usd": 0.0, "experiments": {}}
    try:
        with open(path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        return data
    except Exception:
        return {"total_cost_usd": 0.0, "experiments": {}}


def _save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
    tmp.replace(path)   # atomic rename


# ── 공개 API ──────────────────────────────────────────────────────────────────

class BudgetExceededError(RuntimeError):
    """예산 한도 초과 시 발생하는 예외."""
    def __init__(self, spent: float, limit: float):
        self.spent = spent
        self.limit = limit
        super().__init__(
            f"Budget limit reached: ${spent:.4f} spent / ${limit:.2f} limit"
        )


class CostTracker:
    """
    실험 단위 비용 추적기.

    Args:
        experiment_name : MODEL_NAME 환경변수와 동일한 식별자
        max_budget_usd  : 총 누적 한도 (USD). 0 이하면 무제한
        log_path        : cost_log.json 경로 (기본: logs/cost_log.json)
    """

    def __init__(
        self,
        experiment_name: str,
        max_budget_usd: float = 0.0,
        log_path: Optional[Path] = None,
    ):
        self.experiment_name  = experiment_name
        self.max_budget_usd   = max_budget_usd
        self.log_path         = Path(log_path) if log_path else _DEFAULT_LOG
        self._session_cost    = 0.0   # 이번 프로세스 내 누적

    # ── 핵심 메서드 ────────────────────────────────────────────────────────────

    def record(self, cb, ticker: str = "", trade_date: str = "") -> bool:
        """
        get_openai_callback() 결과를 기록하고 예산 초과 여부를 반환.

        Args:
            cb          : get_openai_callback() 컨텍스트 객체
            ticker      : 현재 티커 (로그용)
            trade_date  : 현재 날짜 (로그용)

        Returns:
            True  → 예산 초과 (호출자가 처리)
            False → 정상
        """
        day_cost = cb.total_cost
        self._session_cost += day_cost

        data = _load(self.log_path)
        exp  = data["experiments"].setdefault(self.experiment_name, {
            "cost_usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "ticker_days_completed": 0,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        exp["cost_usd"]               += day_cost
        exp["prompt_tokens"]          += cb.prompt_tokens
        exp["completion_tokens"]      += cb.completion_tokens
        exp["ticker_days_completed"]  += 1
        exp["last_entry"]              = {
            "ticker": ticker, "date": trade_date,
            "cost_usd": round(day_cost, 6),
            "tokens": cb.total_tokens,
        }
        exp["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        data["total_cost_usd"] = sum(
            e["cost_usd"] for e in data["experiments"].values()
        )
        _save(self.log_path, data)

        if self.max_budget_usd > 0 and data["total_cost_usd"] >= self.max_budget_usd:
            return True   # 예산 초과
        return False

    def check_budget(self) -> bool:
        """현재 누적 총 비용이 예산을 초과했는지 확인 (기록 없이)."""
        if self.max_budget_usd <= 0:
            return False
        total = _load(self.log_path).get("total_cost_usd", 0.0)
        return total >= self.max_budget_usd

    # ── 조회 ───────────────────────────────────────────────────────────────────

    @property
    def session_cost(self) -> float:
        """현재 프로세스 내 누적 비용."""
        return self._session_cost

    @classmethod
    def get_total_cost(cls, log_path: Optional[Path] = None) -> float:
        """전체 누적 총 비용 (모든 실험 합산)."""
        return _load(log_path or _DEFAULT_LOG).get("total_cost_usd", 0.0)

    @classmethod
    def get_summary(cls, log_path: Optional[Path] = None) -> dict:
        """cost_log.json 전체 내용 반환."""
        return _load(log_path or _DEFAULT_LOG)

    @classmethod
    def print_summary(cls, log_path: Optional[Path] = None):
        """콘솔에 비용 요약 출력."""
        data = _load(log_path or _DEFAULT_LOG)
        total = data.get("total_cost_usd", 0.0)
        exps  = data.get("experiments", {})

        print("=" * 60)
        print(f"  누적 총 비용: ${total:.4f}")
        print("=" * 60)
        for name, info in sorted(exps.items()):
            cost   = info.get("cost_usd", 0.0)
            days   = info.get("ticker_days_completed", 0)
            ptok   = info.get("prompt_tokens", 0)
            ctok   = info.get("completion_tokens", 0)
            upd    = info.get("updated_at", "")
            print(f"  {name}")
            print(f"    비용    : ${cost:.4f}   ({days} ticker-days)")
            print(f"    토큰    : input={ptok:,}  output={ctok:,}")
            print(f"    최종갱신: {upd}")
        print("=" * 60)
