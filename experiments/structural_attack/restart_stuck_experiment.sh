#!/bin/bash
# 멈춘 실험(adv_gemini_q3_A2_sell_hybrid) 종료 후 --resume으로 재시작
set -e
cd /home/iislab/workspace/nacw/multi_agents/mat_attack/structural_attack

echo "=== 1. 멈춰있는 main_adversarial.py(A2/hybrid) 프로세스 종료 ==="
for PID in 3751501 3751523 3751528 3751530 3751531; do
  kill -9 "$PID" 2>/dev/null || true
done
sleep 2

echo "=== 2. --resume 으로 해당 실험만 재시작 (완료된 4개 티커 + AAPL 47일치 보존) ==="
mkdir -p logs
.venv/bin/python3 run_q3_adversarial_gemini.py --attack A2 --structure hybrid --resume \
  > logs/adv_gemini_q3_A2_sell_hybrid_resumed.out 2>&1 &

echo "재시작 완료. 로그: tail -f logs/adv_gemini_q3_A2_sell_hybrid_resumed.out"
