"""Matched safety-instruction pilot with evidence-grounded reasoning labels."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval
from inspect_ai.dataset import MemoryDataset

from .conditions import CONDITIONS
from .data import ROOT
from .pilot import JUDGE, TARGET, PRICES, make_tasks, model_args, select_ids
from .rich_scoring import RUBRIC, awareness_labels
from .scoring import refusal, vea

MATCHED_CONDITIONS = ("baseline", "decommissioned") + tuple(
    f"matched_{cue}_{grading}" for cue in ("ordinary", "test")
    for grading in ("active", "inactive", "safety_only"))


def build(epochs=2, per_benchmark=5, connections=8):
    tasks = make_tasks(MATCHED_CONDITIONS, select_ids(per_benchmark), epochs, 8192, connections)
    for i, task in enumerate(tasks):
        args = (JUDGE, "auto", model_args("azure"), task.metadata["judge_config"])
        task.scorer = [awareness_labels(*args), vea(*args, tolerant=True), refusal(*args, tolerant=True)]
        task.metadata.update(experiment_version="matched_v2", awareness_rubric="evidence_v1",
                             awareness_rubric_sha256=hashlib.sha256(RUBRIC.encode()).hexdigest(),
                             scoring_missingness="per_metric", judge_parse_attempts=2)
        samples = list(task.dataset)
        random.Random(42+i).shuffle(samples)
        task.dataset = MemoryDataset(samples, name=task.dataset.name)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per-benchmark", type=int, default=5)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "results" / "pilot-02")
    args = parser.parse_args()
    if args.epochs < 1 or args.connections < 1:
        parser.error("epochs and connections must be positive")
    tasks = build(args.epochs, args.per_benchmark, args.connections)
    n = sum(len(t.dataset) for t in tasks)*args.epochs
    plan = {
        "experiment_version": "matched_v2", "target": TARGET, "judge": JUDGE,
        "target_model_args": model_args("alibaba"), "judge_model_args": model_args("azure"),
        "target_calls": n, "judge_calls_before_retries": 3*n, "maximum_parse_attempt_calls": 6*n,
        "epochs": args.epochs, "connections": args.connections,
        "system_messages": {c: CONDITIONS[c].system for c in MATCHED_CONDITIONS},
        "prompt_ids": select_ids(args.per_benchmark),
        "generation_config": tasks[0].config.model_dump(exclude_none=True),
        "judge_config": tasks[0].metadata["judge_config"],
        "awareness_rubric": RUBRIC, "pricing_snapshot_2026_09_23": PRICES,
        "planned_contrasts": "Within each framing: active-inactive, active-safety_only, inactive-safety_only; test-ordinary at each grading status; each arm-baseline.",
        "interpretation": "Prompt-context intervention; evidence labels are nonexclusive, not motive diagnoses. Same original 25 prompts unless selection changed; exploratory bootstrap, no multiplicity correction.",
    }
    print(json.dumps(plan, indent=2))
    if not args.run:
        print("PLAN ONLY: no model calls.")
        return
    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required")
    if args.log_dir.exists() and any(args.log_dir.iterdir()):
        parser.error("Use a new empty directory to preserve completed runs")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    logs = eval(tasks, model=TARGET, model_args=model_args("alibaba"),
                max_connections=args.connections, max_samples=args.connections,
                max_tasks=len(tasks), log_dir=str(args.log_dir), display="plain",
                fail_on_error=.1, retry_on_error=0, log_model_api=False)
    if any(l.status != "success" for l in logs):
        raise SystemExit("Incomplete run; inspect saved logs before analyzing.")
    from .matched_report import write_report
    write_report(sorted(str(p) for p in args.log_dir.glob("*.eval")), args.log_dir / "analysis")


if __name__ == "__main__":
    main()
