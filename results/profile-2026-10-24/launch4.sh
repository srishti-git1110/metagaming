#!/bin/bash
# Fill in the 15 Fortress items whose realistic rewrite was recovered on retry, then rebuild the report.
cd /Users/srishtigureja/Desktop/Personal/metagaming
L=results/profile-2026-10-24/logs
python3 -m metagaming.profile belief --fortress-per-domain 38 --connections 24 --run > $L/belief-fortress-fill.log 2>&1
python3 -m metagaming.profile behave --fortress-per-domain 38 --targets deepseek,gptoss,glm --samples 6 --connections 24 --run > $L/behave-fortress-fill.log 2>&1 &
python3 -m metagaming.profile behave --fortress-per-domain 38 --targets kimi --samples 3 --connections 8 --run > $L/behave-fortress-fill-kimi.log 2>&1 &
wait
python3 -m metagaming.profile judge --judge-model deepseek --connections 24 --run > $L/judge-deepseek-3.log 2>&1
python3 -m metagaming.cot_judge --judge deepseek --connections 24 --run > $L/cot-judge-deepseek-3.log 2>&1
python3 -m metagaming.profile report --fortress-per-domain 38 --judge-model deepseek > $L/report-deepseek.log 2>&1
echo CHAIN4_DONE
