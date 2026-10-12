#!/usr/bin/env bash
# Round 2 (2026-10-09): scorer-vs-user conflicts, "format does not matter", novel rewarded format.
# Appends to the same single/<profile>.jsonl files; existing keys are skipped. Spends OpenRouter credits.
set -u
L=results/habit-vs-reward-2026-10-06/logs
R=(.venv/bin/python -m metagaming.habit_reward)
ARMS=Uplus,GplusUminus,GminusUplus,Gfree,Nstated,NstatedUplus,Nuser,Nshown,Nshown_rev,Nshown_unscored
for p in gptoss-low gptoss-high qwen8-think qwen8-nothink; do
  nohup "${R[@]}" single --profile $p --n 200 --arms $ARMS --connections 16 --run >> $L/round2-$p.log 2>&1 &
done
