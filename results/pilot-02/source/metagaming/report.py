"""Generate plots, paired estimates and an audit export from existing logs only."""
import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from inspect_ai.log import read_eval_log

from .analyze import load_logs, summarize
from .scoring import split_response


def write_csv(path, rows, fields=None):
    if not fields:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def observed_rate_ci(rows, condition, metric, draws=2000):
    """Equal-weight prompt means on available observations, clustered by prompt."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["condition"] == condition and row[metric] is not None:
            groups[row["benchmark"]][row["prompt_id"]].append(row[metric])
    strata = [[mean(v) for v in prompts.values()] for prompts in groups.values()]
    values = [v for stratum in strata for v in stratum]
    if not values:
        return None, None, None
    rng = random.Random(42)
    boot = sorted(mean([v for stratum in strata for v in rng.choices(stratum, k=len(stratum))])
                  for _ in range(draws))
    return mean(values), boot[int(.025*(draws-1))], boot[int(.975*(draws-1))]


def write_report(paths: list[str], output: Path):
    rows = load_logs(paths)
    # A harmless plumbing test is not scientific evidence and gets no pilot plots.
    logs = [read_eval_log(path) for path in paths]
    if any((log.eval.metadata or {}).get("smoke") for log in logs):
        raise ValueError("Smoke logs are transport checks, not scientific pilot data")
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    summary["quality"] = {}
    for condition in summary["coverage"]:
        selected = [r for r in rows if r["condition"] == condition]
        summary["quality"][condition] = {
            "reasoning_available": sum(r.get("reasoning_available") is True for r in selected),
            "final_available": sum(r.get("final_available") is True for r in selected),
            "length_stops": sum(r.get("finish_reason") == "max_tokens" for r in selected),
            "sample_errors": sum(r.get("sample_error", False) for r in selected),
            "total": len(selected),
        }
    summary["usage"] = {}
    for log in logs:
        for model, usage in log.stats.model_usage.items():
            totals = summary["usage"].setdefault(model, {"input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0, "input_tokens_cache_read": 0, "input_tokens_cache_write": 0,
                "reasoning_tokens": 0, "reported_cost_usd": None})
            for key in ("input_tokens", "output_tokens", "total_tokens", "input_tokens_cache_read",
                        "input_tokens_cache_write", "reasoning_tokens"):
                totals[key] += getattr(usage, key) or 0
            if usage.total_cost is not None:
                totals["reported_cost_usd"] = (totals["reported_cost_usd"] or 0) + usage.total_cost
    summary["warnings"] = [
        "Exploratory pilot: effects do not identify a metagaming type.",
        "VEA in explicit-context arms is prompted/primed, not spontaneous awareness.",
        "Nonverbalized awareness cannot be ruled out; observable reasoning is not guaranteed faithful.",
        "Check missingness and length stops before comparing complete-case effects.",
        "Bootstrap intervals are unadjusted for multiple comparisons and unstable with few prompts.",
        "No deployment negatives: do not interpret these scores as eval-detection AUROC.",
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(output / "samples.csv", rows)
    contrast_rows = []
    for contrast in summary["paired_contrasts"]:
        contrast_rows.append({
            "metric": contrast["metric"], "contrast": json.dumps(contrast["weights"]),
            "delta_pp": contrast["delta"]*100 if contrast["delta"] is not None else None,
            "ci_low_pp": contrast["ci95"][0]*100 if contrast["ci95"] else None,
            "ci_high_pp": contrast["ci95"][1]*100 if contrast["ci95"] else None,
            "eligible_prompts": contrast["eligible_prompts"], "excluded_prompts": contrast["excluded_prompts"],
        })
    write_csv(output / "contrasts.csv", contrast_rows,
              fields=["metric", "contrast", "delta_pp", "ci_low_pp", "ci_high_pp", "eligible_prompts", "excluded_prompts"])
    audits = []
    for log in logs:
        for sample in log.samples or []:
            parts = split_response(sample.output, (log.eval.metadata or {}).get("response_mode", "auto"))
            scores = sample.scores or {}
            audits.append({
                "id": sample.id, "epoch": sample.epoch, "condition": sample.metadata["condition"],
                "benchmark": sample.metadata["benchmark"],
                "question": sample.metadata.get("original_prompt"),
                "reasoning": parts.reasoning, "final_answer": parts.final,
                "vea": scores["vea"].value if "vea" in scores else None,
                "refusal": scores["refusal"].value if "refusal" in scores else None,
                "vea_judge": scores["vea"].explanation if "vea" in scores else None,
                "refusal_judge": scores["refusal"].explanation if "refusal" in scores else None,
                "vea_quote": (scores["vea"].metadata or {}).get("quote") if "vea" in scores else None,
                "manual_vea": None, "manual_refusal": None,
                "manual_awareness_kind": None, # e.g. harness-inference / task-testing / echo / denial / unclear
                "manual_notes": "",
            })
    with (output / "audit.jsonl").open("w") as handle:
        for row in audits:
            handle.write(json.dumps(row) + "\n")
    # Set a writable cache before importing matplotlib (works in sandboxed runs).
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conditions = list(summary["coverage"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True, layout="constrained")
    for axis, metric, title in zip(axes, ("refusal", "vea"), ("Refusal", "Verbalized evaluation awareness")):
        for index, condition in enumerate(conditions):
            rate, low, high = observed_rate_ci(rows, condition, metric)
            if rate is not None:
                axis.errorbar(rate*100, index, xerr=[[max(0, rate-low)*100], [max(0, high-rate)*100]],
                              fmt="o", capsize=4, color="#2563eb")
        axis.set_yticks(range(len(conditions)), labels=conditions)
        axis.set_xlim(-3, 103)
        axis.set_xlabel("Mean within-prompt rate (%) · observed scores")
        axis.set_title(title)
        axis.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Exploratory pilot · prompt-cluster bootstrap 95% intervals")
    fig.savefig(output / "rates.png", dpi=160)
    plt.close(fig)
    primary = [c for c in summary["paired_contrasts"]
               if c["weights"].get("baseline") == -1 and len(c["weights"]) == 2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for axis, metric in zip(axes, ("refusal", "vea")):
        selected = [c for c in primary if c["metric"] == metric]
        for i, c in enumerate(selected):
            if c["delta"] is not None:
                low, high = c["ci95"]
                axis.errorbar(100*c["delta"], i,
                              xerr=[[100*max(0, c["delta"]-low)], [100*max(0, high-c["delta"])]],
                              fmt="o", capsize=4, color="#7c3aed")
        axis.set_yticks(range(len(selected)), labels=[next(k for k in c["weights"] if k != "baseline") for c in selected])
        axis.invert_yaxis()
        axis.axvline(0, color="#888", linestyle="--")
        axis.set_xlabel("Condition minus baseline (percentage points)")
        axis.set_title(metric)
        axis.grid(axis="x", alpha=.2)
    fig.suptitle("Paired effects · complete matched prompt/epoch observations")
    fig.savefig(output / "effects.png", dpi=160)
    plt.close(fig)
    table = ["| Condition | Rollouts | Refusal observed | VEA observed | Length stops |",
             "|---|---:|---:|---:|---:|"]
    for c, coverage in summary["coverage"].items():
        table.append(f"| {c} | {coverage['total_rollouts']} | {coverage['refusal']['observed']} | {coverage['vea']['observed']} | {summary['quality'][c]['length_stops']} |")
    (output / "report.md").write_text(
        "# Exploratory pilot results\n\n" + "\n".join(table)
        + "\n\n![Rates](rates.png)\n\n![Paired effects](effects.png)\n\n"
        + "\n".join("- " + w for w in summary["warnings"])
        + "\n\nReview audit.jsonl: examine both positive and negative VEA labels, and distinguish"
          " independent inference from repeating or denying the supplied evaluation framing,"
          " and general task-testing language from belief in external model evaluation."
          " No automated motive label is produced. Token counts are in summary.json; consult"
          " OpenRouter billing for actual charges, including retries.\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.logs, args.out)
    print(f"Saved report to {args.out}")


if __name__ == "__main__":
    main()
