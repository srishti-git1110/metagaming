#!/usr/bin/env bash
# MASK honesty run on EvalDetectBench items (2026-10-15). Spends OpenRouter credits.
set -u
L=results/mask-2026-10-15/logs
R=(.venv/bin/python -m metagaming.mask_run)
nohup "${R[@]}" detect --connections 12 --run >> $L/detect.log 2>&1 &
nohup bash -c '"$@" behave --samples 5 --connections 16 --run && "$@" judge --connections 16 --run && "$@" report' _ "${R[@]}" >> $L/behave.log 2>&1 &
