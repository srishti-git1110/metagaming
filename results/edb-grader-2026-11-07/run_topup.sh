#!/bin/bash
# usage: run_topup.sh <name> <openrouter model id> <ids_key>  -- reruns only the items listed under ids_key in topup-ids.json, all five arms
set -e
NAME=$1; MODEL=openrouter/$2; KEY=$3; OUT=results/edb-grader-2026-11-07/$NAME; T=metagaming/inspect_tasks/edb_grader.py
IDS=$PWD/results/edb-grader-2026-11-07/topup-ids.json
mkdir -p $OUT
.venv/bin/inspect eval $T --model $MODEL -T arm=baseline -T ids_file=$IDS -T ids_key=$KEY --max-connections 12 --log-dir $OUT/logs-baseline-a
BASE=$(ls $PWD/$OUT/logs-baseline-a/*.eval | head -1)
.venv/bin/inspect eval $T --model $MODEL -T arm=baseline -T ids_file=$IDS -T ids_key=$KEY -T baseline_log=$BASE --max-connections 12 --log-dir $OUT/logs-baseline-b
for arm in mentor grader_auto filler; do
  .venv/bin/inspect eval $T --model $MODEL -T arm=$arm -T ids_file=$IDS -T ids_key=$KEY -T baseline_log=$BASE --max-connections 12 --log-dir $OUT/logs-$arm
done
echo "DONE $NAME"
