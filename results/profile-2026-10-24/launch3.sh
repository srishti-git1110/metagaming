#!/bin/bash
# After the full-FORTRESS build: belief probes, behavior, judge, chain-of-thought judge, report.
cd /Users/srishtigureja/Desktop/Personal/metagaming
L=results/profile-2026-10-24/logs
until grep -q "renderings (+" $L/build-fortress-full.log; do sleep 15; done
python3 -m metagaming.profile belief --fortress-per-domain 38 --connections 24 --run > $L/belief-fortress-full.log 2>&1
python3 -m metagaming.profile behave --fortress-per-domain 38 --targets deepseek,gptoss,glm --samples 6 --connections 24 --run > $L/behave-fortress-full.log 2>&1 &
python3 -m metagaming.profile behave --fortress-per-domain 38 --targets kimi --samples 3 --connections 8 --run > $L/behave-fortress-full-kimi.log 2>&1 &
wait
python3 -m metagaming.profile judge --judge-model deepseek --connections 24 --run > $L/judge-deepseek-2.log 2>&1
python3 -m metagaming.cot_judge --judge deepseek --connections 24 --run > $L/cot-judge-deepseek-2.log 2>&1
python3 -m metagaming.cot_judge --judge glm --pools mask --arms E,D --connections 16 --run > $L/cot-judge-glm.log 2>&1
python3 -m metagaming.profile report --judge-model deepseek > $L/report-deepseek.log 2>&1
echo CHAIN_DONE
