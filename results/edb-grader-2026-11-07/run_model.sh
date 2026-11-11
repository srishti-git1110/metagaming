#!/bin/bash
# usage: run_model.sh <name> <openrouter model id>   e.g. run_model.sh gptoss openai/gpt-oss-120b
# Five runs on the 425 EvalDetectBench items: baseline twice (second judged against the first = noise floor),
# then mentor / grader_auto / filler, each judged against the first baseline.
set -e
NAME=$1; MODEL=openrouter/$2; OUT=results/edb-grader-2026-11-07/$NAME; T=metagaming/inspect_tasks/edb_grader.py
mkdir -p $OUT
.venv/bin/inspect eval $T --model $MODEL -T arm=baseline --max-connections 16 --log-dir $OUT/logs-baseline-a
BASE=$(ls $PWD/$OUT/logs-baseline-a/*.eval | head -1)
.venv/bin/inspect eval $T --model $MODEL -T arm=baseline -T baseline_log=$BASE --max-connections 16 --log-dir $OUT/logs-baseline-b
for arm in mentor grader_auto filler; do
  .venv/bin/inspect eval $T --model $MODEL -T arm=$arm -T baseline_log=$BASE --max-connections 16 --log-dir $OUT/logs-$arm
done
echo "DONE $NAME"
