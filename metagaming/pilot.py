"""OpenRouter pilot. Default is offline planning; --run spends API credits."""
import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

from .conditions import CONDITIONS
from .data import ROOT
from .tasks import awareness, build_samples

TARGET = "openrouter/qwen/qwen3-8b"
JUDGE = "openrouter/openai/gpt-5-mini"
PRIMARY_CONDITIONS = (
    "baseline", "decommissioned", "neutral_control", "test_active", "test_inactive",
    "ordinary_active", "ordinary_inactive",
)
# Catalog snapshot 2026-09-23. USD per token, not per million tokens.
PRICES = {TARGET: {"input": 0.000000117, "output": 0.000000455},
          JUDGE: {"input": 0.00000025, "output": 0.000002}}


def select_ids(per_benchmark: int = 5, seed: int = 42) -> list[str]:
    """Fixed, outcome-independent stratified subset. 0 means all 444 prompts."""
    if per_benchmark < 0:
        raise ValueError("per_benchmark must be nonnegative")
    groups = defaultdict(list)
    for sample in build_samples("baseline"):
        groups[sample.metadata["benchmark"]].append(str(sample.id))
    ids = []
    for benchmark in sorted(groups):
        candidates = sorted(groups[benchmark])
        stable_seed = int.from_bytes(hashlib.sha256(f"{seed}:{benchmark}".encode()).digest()[:8])
        random.Random(stable_seed).shuffle(candidates)
        if per_benchmark > len(candidates):
            raise ValueError(f"Only {len(candidates)} {benchmark} prompts are available")
        ids.extend(candidates[:per_benchmark] if per_benchmark else candidates)
    return ids


def model_args(provider: str) -> dict:
    return {"reasoning_enabled": True, "stream": False,
            "provider": {"only": [provider], "allow_fallbacks": False, "require_parameters": True}}


def estimate_cost(n: int, target_max_tokens: int, judge_max_tokens: int) -> dict:
    """Planning estimates, not a guaranteed cap or provider billing quote."""
    t, j = PRICES[TARGET], PRICES[JUDGE]
    typical = n * (800*t["input"] + 1500*t["output"]
                   + 2500*j["input"] + 2*500*j["output"])
    capped_outputs = n * (target_max_tokens*t["output"] + 2*judge_max_tokens*j["output"])
    return {
        "pricing_as_of": "2026-09-23", "usd_per_token": PRICES,
        "illustrative_usd": round(typical, 2),
        "max_output_tokens_only_usd_excluding_input_and_retries": round(capped_outputs, 2),
        "assumptions": "800 target input, 1500 target output, 2500 total judge input, 500 output per judge call (including hidden reasoning). Actual usage varies; neither figure is a hard spend cap.",
    }


def make_tasks(conditions, ids, epochs, max_tokens, judge_connections,
               judge_max_tokens=4000, smoke=False):
    tasks = []
    judge_args = model_args("azure")
    judge_settings = {"max_tokens": judge_max_tokens, "reasoning_effort": "low",
                      "max_connections": judge_connections, "max_retries": 2,
                      "timeout": 180}
    for condition in conditions:
        task = awareness(JUDGE, condition=condition, response_mode="auto", epochs=epochs,
                         max_tokens=max_tokens, judge_model_args=judge_args,
                         judge_config=judge_settings)
        if smoke:
            prompt = "A box holds 3 red balls and 5 blue balls. How many balls are there in total?"
            messages = [ChatMessageUser(content=prompt)]
            if CONDITIONS[condition].system:
                messages.insert(0, ChatMessageSystem(content=CONDITIONS[condition].system))
            samples = [Sample(id="transport_smoke", input=messages, metadata={
                "condition": condition, "benchmark": "transport_smoke", "original_prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            })]
        else:
            samples = [s for s in task.dataset if str(s.id) in set(ids)]
        task.dataset = MemoryDataset(samples, name="transport_smoke" if smoke else "olmo_stratified_pilot")
        task.config = GenerateConfig(temperature=.6, top_p=.95, top_k=20, max_tokens=max_tokens,
                                     seed=42, max_retries=2, timeout=180)
        task.metadata.update(pilot=True, smoke=smoke, selection_seed=42,
                             sampling_note="Qwen thinking mode: temperature=.6, top_p=.95, top_k=20")
        tasks.append(task)
    return tasks


def make_plan(args):
    conditions = args.conditions.split(",") if args.conditions else list(PRIMARY_CONDITIONS)
    if args.smoke:
        conditions = ["baseline", "decommissioned"]
    if len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError("Conditions must be unique names from metagaming.conditions.CONDITIONS")
    ids = [] if args.smoke else select_ids(args.per_benchmark)
    epochs = 1 if args.smoke else args.epochs
    tasks = make_tasks(conditions, ids, epochs, args.max_tokens, args.judge_connections,
                       args.judge_max_tokens, smoke=args.smoke)
    n = sum(len(t.dataset) for t in tasks) * epochs
    plan = {
        "target": TARGET, "target_model_args": model_args("alibaba"),
        "judge": JUDGE, "judge_model_args": model_args("azure"),
        "conditions": conditions, "system_messages": {c: CONDITIONS[c].system for c in conditions},
        "prompt_ids": [s.id for s in tasks[0].dataset],
        "prompt_sha256": {str(s.id): s.metadata["prompt_sha256"] for s in tasks[0].dataset},
        "benchmarks": dict(Counter(s.metadata["benchmark"] for s in tasks[0].dataset)),
        "epochs": epochs, "target_calls": n, "maximum_judge_calls": 2*n,
        "target_connections": args.connections, "judge_connections": args.judge_connections,
        "generation_config": tasks[0].config.model_dump(exclude_none=True),
        "judge_config": tasks[0].metadata["judge_config"],
        "cost_estimate": estimate_cost(n, args.max_tokens, args.judge_max_tokens),
        "smoke": args.smoke,
    }
    return tasks, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Make paid target and judge calls")
    parser.add_argument("--smoke", action="store_true", help="Only 2 harmless target calls plus up to 4 judge calls")
    parser.add_argument("--per-benchmark", type=int, default=5, help="Questions per benchmark; 0 selects all 444")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--conditions", help="Comma-separated condition names; default: seven conditions, no persona")
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--judge-connections", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--judge-max-tokens", type=int, default=4000)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    for field in ("epochs", "connections", "judge_connections", "max_tokens", "judge_max_tokens"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    tasks, plan = make_plan(args)
    print(json.dumps(plan, indent=2))
    if not args.run:
        print("PLAN ONLY: no credentials loaded and no network/model calls. Add --run to execute.")
        return
    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY in the repo .env or environment; the key is never printed.")
    directory = args.log_dir or ROOT / "logs" / (
        ("smoke-" if args.smoke else "pilot-") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if directory.exists() and any(directory.iterdir()):
        parser.error(f"Use a new, empty log directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    # Conditions run concurrently; sample order differs by arm to avoid a fixed
    # prompt-order/time confound. Global per-model connection limits bound requests.
    for i, task in enumerate(tasks):
        samples = list(task.dataset)
        random.Random(42 + i).shuffle(samples)
        task.dataset = MemoryDataset(samples, name=task.dataset.name)
    logs = eval(tasks, model=TARGET, model_args=model_args("alibaba"),
                max_connections=args.connections, max_samples=args.connections,
                max_tasks=len(tasks), log_dir=str(directory), display="plain",
                fail_on_error=.1, retry_on_error=0, log_model_api=False)
    if any(log.status != "success" for log in logs):
        raise SystemExit(f"Run incomplete. Inspect errors in {directory}; do not pool it as a complete experiment.")
    samples = [s for log in logs for s in (log.samples or [])]
    available = sum(bool(s.metadata.get("reasoning_available")) for s in samples)
    final_available = sum(bool(s.metadata.get("final_available")) for s in samples)
    print(f"Exposed reasoning: {available}/{len(samples)}; final answers: {final_available}/{len(samples)}")
    if args.smoke and (available != len(samples) or final_available != len(samples)
                       or any(s.error or len(s.scores or {}) != 2 for s in samples)):
        raise SystemExit("Smoke failed: missing reasoning, answer, or judge scores. Inspect logs before scaling.")
    print(f"Logs: {directory}")
    if not args.smoke:
        from .report import write_report
        write_report([str(p) for p in directory.glob("*.eval")], directory / "analysis")
        print(f"Report: {directory / 'analysis'}")


if __name__ == "__main__":
    main()
