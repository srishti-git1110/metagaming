#!/usr/bin/env bash
# Round 5 (2026-10-22), budget-limited first launch: 5 cheap configs, 2 samples, GPT-6-luna judge.
# Add targets/judges/samples by re-running the same stages with different flags; existing work is skipped.
set -u
L=results/round5-2026-10-22/logs
R=(.venv/bin/python -m metagaming.round5)
nohup "${R[@]}" detect --judges luna --connections 12 --run >> $L/detect.log 2>&1 &
nohup bash -c '"$@" behave --targets luna,gptoss-high,gptoss-low,deepseek,deepseek-nothink --samples 2 --connections 20 --run && "$@" judge --judges luna --connections 16 --run && "$@" report' _ "${R[@]}" >> $L/behave.log 2>&1 &
