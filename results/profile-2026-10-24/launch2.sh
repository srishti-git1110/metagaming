#!/bin/bash
# 27 Oct: re-judge all replies with DeepSeek (no thinking); judge all chains of thought; extend FORTRESS to all 100 prompts.
cd /Users/srishtigureja/Desktop/Personal/metagaming
L=results/profile-2026-10-24/logs
nohup python3 -m metagaming.profile judge --judge-model deepseek --connections 24 --run > $L/judge-deepseek.log 2>&1 &
nohup python3 -m metagaming.cot_judge --judge deepseek --connections 24 --run > $L/cot-judge-deepseek.log 2>&1 &
nohup python3 -m metagaming.profile build --fortress-per-domain 38 --run > $L/build-fortress-full.log 2>&1 &
wait
