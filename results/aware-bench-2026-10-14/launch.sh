#!/usr/bin/env bash
# First EvalDetectBench-based run (2026-10-14). Spends OpenRouter credits.
set -u
L=results/aware-bench-2026-10-14/logs
R=(.venv/bin/python -m metagaming.aware_bench)
nohup "${R[@]}" detect --connections 16 --run >> $L/detect.log 2>&1 &
nohup bash -c '"$@" behave --targets luna,gptoss-high,gptoss-low,deepseek --samples 3 --connections 16 --run && "$@" judge --connections 16 --run && "$@" report' _ "${R[@]}" >> $L/behave.log 2>&1 &
