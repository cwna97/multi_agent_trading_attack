# Trading Agent Evaluation Guide

이 가이드는 트레이딩 에이전트 시스템의 성능을 평가하는 방법을 설명합니다.

## 📋 목차

1. [개요](#개요)
2. [평가 메트릭](#평가-메트릭)
3. [사용 방법](#사용-방법)
4. [출력 결과](#출력-결과)
5. [고급 사용법](#고급-사용법)

## 개요

Evaluator는 다음 4가지 주요 기능을 제공합니다:

1. **의사결정 분석**: 에이전트의 투자 결정 패턴 분석
2. **성능 메트릭**: 수익률, 승률, 샤프 비율 등 계산
3. **에이전트 기여도**: 각 에이전트의 최종 결정에 대한 기여도 분석
4. **시각화**: 성과를 시각적으로 표현

## 평가 메트릭

### 1. 재무 성과 메트릭

| 메트릭 | 설명 | 계산 방법 |
|--------|------|-----------|
| **Total Return** | 총 수익률 | 모든 거래의 수익률 합계 |
| **Average Return** | 거래당 평균 수익률 | 총 수익률 / 거래 횟수 |
| **Win Rate** | 승률 | 수익 거래 수 / 전체 거래 수 |
| **Sharpe Ratio** | 샤프 비율 | (평균 수익 - 무위험 수익) / 수익 변동성 |
| **Max Drawdown** | 최대 손실폭 | 최고점 대비 최대 하락률 |

### 2. 의사결정 메트릭

| 메트릭 | 설명 |
|--------|------|
| **Decision Distribution** | BUY/SELL/HOLD 결정 비율 |
| **Consistency Score** | 결정의 일관성 (0-1) |
| **Agent Agreement Rate** | 각 에이전트의 최종 결정 일치율 |

### 3. 에이전트 분석

- **Individual Agreement**: 각 에이전트의 최종 결정과의 일치도
- **Contribution Score**: 에이전트별 정확도 추적

## 사용 방법

### 기본 사용법

```python
from tradingagents.evaluation import TradingAgentEvaluator

# Evaluator 초기화
evaluator = TradingAgentEvaluator(results_dir="./eval_results")

# 성능 평가
results = evaluator.evaluate_performance(ticker="NVDA")

# 결과 출력
print(f"Total Decisions: {results['total_decisions']}")
print(f"Consistency: {results['consistency_score']:.3f}")
```

### 가격 데이터를 활용한 평가

```python
import pandas as pd
import yfinance as yf

# 실제 주가 데이터 가져오기
stock = yf.Ticker("NVDA")
hist = stock.history(start="2025-11-01", end="2025-12-10")
prices_df = pd.DataFrame({
    'date': hist.index.strftime('%Y-%m-%d'),
    'close': hist['Close'].values
})

# 가격 데이터를 포함한 평가
results = evaluator.evaluate_performance(
    ticker="NVDA",
    actual_prices=prices_df
)

# 수익률 출력
print(f"Total Return: {results['performance_metrics']['total_return']*100:.2f}%")
print(f"Win Rate: {results['performance_metrics']['win_rate']*100:.1f}%")
```

### 리포트 생성

```python
# 텍스트 리포트 생성
report = evaluator.generate_report(
    ticker="NVDA",
    actual_prices=prices_df,
    output_file="./evaluation_report_NVDA.txt"
)

print(report)
```

### 시각화 생성

```python
from tradingagents.evaluation.visualizer import PerformanceVisualizer

visualizer = PerformanceVisualizer()

# 종합 시각화 리포트 생성
visualizer.create_comprehensive_report(
    evaluation_results=results,
    output_dir="./visualizations"
)
```

### 간편 실행 스크립트

```bash
# 단일 종목 종합 평가
python run_full_evaluation.py

# 또는 Python 코드에서:
from run_full_evaluation import run_comprehensive_evaluation

run_comprehensive_evaluation(
    ticker="NVDA",
    start_date="2025-11-01",
    end_date="2025-12-10"
)
```

## 출력 결과

### 1. 텍스트 리포트 (`evaluation_report_NVDA.txt`)

```
================================================================================
Trading Agent Evaluation Report
Ticker: NVDA
Evaluation Date: 2025-12-13T10:30:00
================================================================================

Total Decisions Analyzed: 5

Decision Distribution:
----------------------------------------
Final Decisions:
  BUY: 3 (60.0%)
  HOLD: 2 (40.0%)

Agent Agreement Rates:
----------------------------------------
  Market Report: 80.0%
  Sentiment Report: 100.0%
  News Report: 100.0%
  Fundamentals Report: 100.0%

Decision Consistency Score: 0.750

Financial Performance Metrics:
----------------------------------------
  Total Return: 5.23%
  Average Return per Trade: 2.62%
  Win Rate: 66.7%
  Total Trades: 2
  Sharpe Ratio: 1.234
================================================================================
```

### 2. JSON 결과 (`evaluation_results_NVDA.json`)

```json
{
  "ticker": "NVDA",
  "evaluation_date": "2025-12-13T10:30:00",
  "total_decisions": 5,
  "decision_distribution": {
    "final_decisions": {"BUY": 3, "HOLD": 2}
  },
  "agent_analysis": {
    "agreement_rates": {
      "market_report_decision": 0.8,
      "sentiment_report_decision": 1.0,
      "news_report_decision": 1.0,
      "fundamentals_report_decision": 1.0
    }
  },
  "performance_metrics": {
    "total_return": 0.0523,
    "avg_return": 0.0262,
    "win_rate": 0.667,
    "total_trades": 2,
    "sharpe_ratio": 1.234
  },
  "consistency_score": 0.75
}
```

### 3. 시각화 파일

- `NVDA_decision_distribution.png`: 결정 분포 파이 차트
- `NVDA_agent_agreement.png`: 에이전트 일치율 막대 그래프
- `NVDA_equity_curve.png`: 포트폴리오 가치 변화 곡선
- `NVDA_returns_dist.png`: 수익률 분포 히스토그램

## 고급 사용법

### 1. 배치 평가 (여러 종목)

```python
from run_full_evaluation import batch_evaluation_with_comparison

batch_evaluation_with_comparison(
    tickers=["NVDA", "AAPL", "MSFT", "GOOGL"],
    start_date="2025-11-01",
    end_date="2025-12-10"
)
```

### 2. 전략 비교

```python
# 다른 설정으로 실행한 전략들 비교
comparison_df = evaluator.compare_strategies(
    ticker="NVDA",
    strategy_names=[
        "TradingAgentsStrategy_1round",
        "TradingAgentsStrategy_2rounds",
        "TradingAgentsStrategy_3rounds"
    ],
    actual_prices=prices_df
)

print(comparison_df)
```

### 3. 커스텀 메트릭 계산

```python
from tradingagents.evaluation.metrics import TradingMetrics

metrics = TradingMetrics()

# 샤프 비율 계산
sharpe = metrics.calculate_sharpe_ratio(
    returns=[0.02, -0.01, 0.03, 0.01],
    risk_free_rate=0.02
)

# 최대 손실폭 계산
max_dd, start_idx, end_idx = metrics.calculate_max_drawdown(
    equity_curve=[10000, 10500, 10200, 10800, 10100]
)

print(f"Sharpe Ratio: {sharpe:.3f}")
print(f"Max Drawdown: {max_dd*100:.2f}%")
```

### 4. 에이전트 기여도 분석

```python
from tradingagents.evaluation.metrics import AgentContributionAnalyzer

analyzer = AgentContributionAnalyzer()

# 에이전트 일치도 분석
agreement = analyzer.analyze_agent_agreement(
    agent_decisions={
        'market_analyst': 'BUY',
        'sentiment_analyst': 'BUY',
        'news_analyst': 'HOLD',
        'fundamentals_analyst': 'BUY'
    },
    final_decision='BUY'
)

print(f"Agreement Rate: {agreement['overall_agreement_rate']*100:.1f}%")
```

## 디렉토리 구조

```
TradingAgents/
├── eval_results/              # 에이전트 결정 로그
│   └── NVDA/
│       └── TradingAgentsStrategy_logs/
│           └── full_states_log_2025-12-03.json
│
├── evaluation_output/         # 평가 결과 출력
│   ├── evaluation_report_NVDA.txt
│   ├── evaluation_results_NVDA.json
│   └── visualizations/
│       ├── NVDA_decision_distribution.png
│       ├── NVDA_agent_agreement.png
│       ├── NVDA_equity_curve.png
│       └── NVDA_returns_dist.png
│
└── tradingagents/
    └── evaluation/
        ├── __init__.py
        ├── metrics.py          # 메트릭 계산
        ├── evaluator.py        # 평가 클래스
        └── visualizer.py       # 시각화 도구
```

## 성능 해석 가이드

### Sharpe Ratio 해석

- **< 1.0**: 위험 대비 수익이 낮음
- **1.0 - 2.0**: 양호한 성과
- **2.0 - 3.0**: 우수한 성과
- **> 3.0**: 매우 우수한 성과

### Win Rate 해석

- **< 40%**: 개선 필요
- **40% - 60%**: 평균적
- **> 60%**: 우수함

### Consistency Score 해석

- **< 0.5**: 결정이 매우 불안정
- **0.5 - 0.7**: 적절한 유연성
- **> 0.7**: 일관성 높음

## 문제 해결

### 문제: "Log directory not found"

**원인**: eval_results 디렉토리가 없거나 잘못된 경로

**해결**:
```python
evaluator = TradingAgentEvaluator(results_dir="./eval_results")
```

### 문제: "No price data available"

**원인**: yfinance에서 데이터를 가져오지 못함

**해결**:
```python
# 날짜 범위 확인
# 주말/공휴일 제외
# 인터넷 연결 확인
```

### 문제: 시각화가 표시되지 않음

**원인**: matplotlib 백엔드 문제

**해결**:
```python
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
```

## 추가 개선 사항

향후 추가할 수 있는 기능들:

1. **실시간 모니터링**: 실시간으로 성능 추적
2. **A/B 테스트**: 다양한 설정 자동 비교
3. **백테스팅**: 과거 데이터로 전략 검증
4. **리스크 메트릭**: VaR, CVaR 등 추가
5. **벤치마크 비교**: S&P 500 등과 비교

## 참고 자료

- [Sharpe Ratio 설명](https://www.investopedia.com/terms/s/sharperatio.asp)
- [Maximum Drawdown](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp)
- [Trading Performance Metrics](https://www.investopedia.com/articles/fundamental-analysis/10/strategy-performance-reports.asp)
