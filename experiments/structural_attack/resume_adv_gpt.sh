#!/bin/bash
# A2_buy_decentralized부터 재실행 (credit 소진 이후 14개 실험)

set -e  # 실험 실패 시 즉시 중단

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python3"
RUNNER="$SCRIPT_DIR/run_adversarial_experiments_gpt.py"

cd "$SCRIPT_DIR"

echo "======================================================"
echo "RESUME: A2_buy_decentralized 부터 재실행"
echo "총 14개 실험"
echo "======================================================"

# A2_buy_decentralized: 149/305 기존 성공 결과 보존 후 나머지만 실행
$PYTHON $RUNNER --attack A2 --signal BUY --structure decentralized --resume
# 이후 실험은 결과 없으므로 일반 실행
$PYTHON $RUNNER --attack A2 --signal BUY --structure moa
$PYTHON $RUNNER --attack A2 --signal SELL
$PYTHON $RUNNER --attack A3 --signal BUY
$PYTHON $RUNNER --attack A3 --signal SELL

echo "======================================================"
echo "모든 재실험 완료"
echo "======================================================"
