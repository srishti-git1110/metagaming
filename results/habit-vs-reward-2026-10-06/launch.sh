#!/usr/bin/env bash
# Exact commands used for the 2026-10-06 habit-vs-reward run (spends OpenRouter credits).
set -u
L=results/habit-vs-reward-2026-10-06/logs
R=(.venv/bin/python -m metagaming.habit_reward)
for p in qwen8-think qwen8-nothink gptoss-high gptoss-low kimi-high; do
  nohup "${R[@]}" single --profile $p --n 200 --connections $([[ $p == kimi* ]] && echo 8 || echo 16) --run >> $L/single-$p.log 2>&1 &
done
for p in qwen8-think gptoss-high; do for o in habit reward; do
  nohup "${R[@]}" single --profile $p --organism $o --n 100 --run >> $L/organism-$p-$o.log 2>&1 &
done; done
for p in qwen8-think gptoss-high kimi-high; do
  nohup "${R[@]}" feedback --profile $p --n 40 --connections 8 --run >> $L/feedback-$p.log 2>&1 &
done
for p in qwen8-nothink gptoss-low kimi-high; do
  nohup "${R[@]}" check --profile $p --n 200 --run >> $L/check-$p.log 2>&1 &
done
