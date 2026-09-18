"""offline paired analysis of already-scored Inspect logs"""
import argparse
import json
import random
from collections import defaultdict
from statistics import mean

from inspect_ai.log import read_eval_log


def paired_contrast(rows: list[dict], weights: dict[str, float], metric: str,
                    bootstrap: int = 2000, seed: int = 42) -> dict:
    """Average within-prompt contrasts; stratified bootstrap of prompt clusters.

    A prompt is eligible only if every requested condition has the same epoch IDs
    and nonmissing measurements. Missing rows cannot silently become zeros.
    """
    grouped = defaultdict(lambda: defaultdict(dict))
    prompt_hashes = {}
    for row in rows:
        if row["condition"] not in weights:
            continue
        key = (row["benchmark"], row["prompt_id"])
        prior = prompt_hashes.setdefault(key, row["prompt_sha256"])
        if prior != row["prompt_sha256"]:
            raise ValueError(f"Different user prompts share an ID: {key}")
        bucket = grouped[key][row["condition"]]
        if row["epoch"] in bucket:
            raise ValueError(f"Duplicate sample/epoch for {key}: {row['condition']}")
        bucket[row["epoch"]] = row.get(metric)
    strata = defaultdict(list)
    for (benchmark, _), conditions in grouped.items():
        if set(conditions) != set(weights):
            continue
        epoch_sets = [set(conditions[c]) for c in weights]
        if any(s != epoch_sets[0] for s in epoch_sets):
            continue
        if any(v is None for c in weights for v in conditions[c].values()):
            continue
        strata[benchmark].append(sum(
            weight * mean(conditions[c].values()) for c, weight in weights.items()
        ))
    effects = [v for group in strata.values() for v in group]
    result = {"metric": metric, "weights": weights, "eligible_prompts": len(effects),
              "excluded_prompts": len(grouped) - len(effects),
              "delta": mean(effects) if effects else None,
              "by_benchmark": {b: {"n": len(v), "delta": mean(v)} for b, v in strata.items()}}
    if effects and bootstrap:
        rng = random.Random(seed)
        draws = sorted(mean([v for group in strata.values()
                             for v in rng.choices(group, k=len(group))])
                       for _ in range(bootstrap))
        result["ci95"] = [draws[int(.025 * (bootstrap - 1))], draws[int(.975 * (bootstrap - 1))]]
    else:
        result["ci95"] = None
    return result


def load_logs(paths: list[str]) -> list[dict]:
    rows, signatures = [], set()

    for path in paths:
        log = read_eval_log(path)
        if log.status != "success":
            raise ValueError(f"Incomplete/failed log: {path}")
        
        metadata = log.eval.metadata or {}
        signatures.add(json.dumps({
            "model": log.eval.model, "model_args": log.eval.model_args,
            "model_base_url": log.eval.model_base_url,
            "generate_config": log.eval.model_generate_config.model_dump(mode="json"),
            "plan_config": log.plan.config.model_dump(mode="json") if log.plan else None,
            "judge": metadata.get("judge_model"), "source": metadata.get("source_commit"),
            "response_mode": metadata.get("response_mode"),
            "refusal_rubric": metadata.get("refusal_rubric"),
            "vea_rubric": metadata.get("vea_rubric"),
        }, sort_keys=True, default=str))

        for sample in log.samples or []:
            sm = sample.metadata or {}
            scores = sample.scores or {}
            record = {k: sm[k] for k in ("condition", "benchmark", "prompt_sha256")}
            record.update(prompt_id=str(sample.id), epoch=sample.epoch)
            for metric in ("vea", "refusal"):
                score = scores.get(metric)
                value = score.value if score is not None and sample.error is None else None
                if value not in (None, 0, 1):
                    raise ValueError(f"Unexpected {metric} score: {value}")
                record[metric] = value
            rows.append(record)

    if len(signatures) != 1:
        raise ValueError("Logs differ in model, generation config, judge, or scoring setup; analyze separately")
    if not rows:
        raise ValueError("No samples in supplied logs")
    return rows


def summarize(rows: list[dict], reference: str = "baseline") -> dict:
    conditions = sorted({r["condition"] for r in rows})
    coverage = {}
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        coverage[condition] = {"total_rollouts": len(subset)}
        for metric in ("vea", "refusal"):
            values = [r[metric] for r in subset if r[metric] is not None]
            coverage[condition][metric] = {
                "observed": len(values), "missing": len(subset) - len(values),
                "rate_observed": mean(values) if values else None,
            }
            
    contrasts = []
    if reference in conditions:
        for condition in conditions:
            if condition != reference:
                for metric in ("vea", "refusal"):
                    contrasts.append(paired_contrast(rows, {condition: 1, reference: -1}, metric))
    factorial = {"test_active": 1, "test_inactive": -1,
                 "ordinary_active": -1, "ordinary_inactive": 1}
    if set(factorial).issubset(conditions):
        for metric in ("vea", "refusal"):
            contrasts.append(paired_contrast(rows, factorial, metric))
            for cue in ("test", "ordinary"):
                contrasts.append(paired_contrast(
                    rows, {f"{cue}_active": 1, f"{cue}_inactive": -1}, metric))
    return {"coverage": coverage, "paired_contrasts": contrasts,
            "interpretation": "Behavioral context effects; no motive classification. Deltas are proportions."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--reference", default="baseline")
    args = parser.parse_args()
    print(json.dumps(summarize(load_logs(args.logs), args.reference), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
