#!/usr/bin/env bash
# Metagaming-profile run (2026-10-24). Open-weight models only. Spends OpenRouter credits.
set -u
L=results/profile-2026-10-24/logs
R=(.venv/bin/python -m metagaming.profile)
nohup bash -c '"$@" belief --targets deepseek,gptoss,glm,kimi --repeats 2 --connections 12 --run; "$@" comprehend --targets deepseek,gptoss,glm,kimi --repeats 3 --run' _ "${R[@]}" >> $L/belief.log 2>&1 &
nohup bash -c '"$@" behave --targets deepseek,gptoss,glm --samples 6 --connections 18 --run; "$@" behave --targets kimi --samples 3 --connections 8 --run; "$@" judge --connections 14 --run; "$@" report' _ "${R[@]}" >> $L/behave.log 2>&1 &
