"""Offline matched-pilot analysis with per-metric missingness and paired contrasts."""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

from inspect_ai.log import read_eval_log

from .analyze import load_logs, paired_contrast
from .pilot import PRICES
from .report import write_csv, observed_rate_ci
from .rich_scoring import LABELS
from .scoring import split_response

METRICS = ("refusal", "vea") + LABELS


def write_report(paths, output):
    rows = load_logs(paths)  # verifies common generation/judge configuration
    logs = [read_eval_log(p) for p in paths]
    if any(l.eval.metadata.get("experiment_version") != "matched_v2" for l in logs):
        raise ValueError("Expected matched_v2 logs")
    records = {(r["condition"], r["prompt_id"], r["epoch"]): r for r in rows}
    audits = []
    for log in logs:
        for s in log.samples or []:
            key = (s.metadata["condition"], str(s.id), s.epoch)
            row = records[key]
            scores = s.scores or {}
            rich = scores.get("awareness_labels")
            for label in LABELS:
                row[label] = rich.value.get(label) if rich else None
            parts = split_response(s.output, "auto")
            audits.append(dict(row, question=s.metadata["original_prompt"],
                               reasoning=parts.reasoning, final_answer=parts.final,
                               scores={k: v.model_dump(mode="json") for k,v in scores.items()},
                               scoring_failures={k:v for k,v in s.metadata.items()
                                                 if "failures" in k or "missing" in k}))
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    coverage = {}
    for c in conditions:
        selected = [r for r in rows if r["condition"] == c]
        coverage[c] = {"total": len(selected), "metrics": {}}
        for m in METRICS:
            vals = [r[m] for r in selected if r[m] is not None]
            coverage[c]["metrics"][m] = dict(observed=len(vals), missing=len(selected)-len(vals),
                                           rate=mean(vals) if vals else None)
    weights = [{c:1, "baseline":-1} for c in conditions if c != "baseline"]
    for cue in ("ordinary", "test"):
        for a,b in (("active","inactive"),("active","safety_only"),("inactive","safety_only")):
            weights.append({f"matched_{cue}_{a}":1, f"matched_{cue}_{b}":-1})
    for g in ("active", "inactive", "safety_only"):
        weights.append({f"matched_test_{g}":1, f"matched_ordinary_{g}":-1})
    contrasts = [paired_contrast(rows,w,m) for w in weights for m in METRICS]
    strategy_behavior = {}
    for c in conditions:
        strategy_behavior[c] = {
            f"strategy_{strategy}_refusal_{refused}": sum(
                r["condition"] == c and r["score_strategy"] == strategy and r["refusal"] == refused
                for r in rows)
            for strategy in (0, 1) for refused in (0, 1)
        }
    by_benchmark = {}
    for c in conditions:
        by_benchmark[c] = {}
        for b in sorted({r["benchmark"] for r in rows}):
            by_benchmark[c][b] = {}
            for m in METRICS:
                vals = [r[m] for r in rows if r["condition"] == c and r["benchmark"] == b and r[m] is not None]
                by_benchmark[c][b][m] = {"n":len(vals), "rate":mean(vals) if vals else None}
    transitions = []
    for w in weights:
        a=next(c for c,v in w.items() if v==1)
        b=next(c for c,v in w.items() if v==-1)
        for r in rows:
            if r["condition"] != a:
                continue
            other=records.get((b,r["prompt_id"],r["epoch"]))
            if other:
                transitions.append(dict(positive=a,negative=b,prompt_id=r["prompt_id"],epoch=r["epoch"],
                    benchmark=r["benchmark"],refusal_positive=r["refusal"],refusal_negative=other["refusal"],
                    strategy_positive=r["score_strategy"],strategy_negative=other["score_strategy"]))
    usage = defaultdict(lambda: dict(input_tokens=0, output_tokens=0, reported_cost_usd=None))
    for l in logs:
        for model,u in l.stats.model_usage.items():
            usage[model]["input_tokens"] += u.input_tokens
            usage[model]["output_tokens"] += u.output_tokens
            if u.total_cost is not None:
                usage[model]["reported_cost_usd"] = (usage[model]["reported_cost_usd"] or 0)+u.total_cost
    estimated = sum(u["input_tokens"]*PRICES[m]["input"]+u["output_tokens"]*PRICES[m]["output"]
                    for m,u in usage.items() if m in PRICES)
    summary = dict(coverage=coverage, paired_contrasts=contrasts, usage=usage,
                   strategy_behavior_descriptive_not_causal=strategy_behavior, by_benchmark=by_benchmark,
                   estimated_usd_at_snapshot_prices=estimated,
                   quality=dict(total=len(rows), reasoning_available=sum(r["reasoning_available"] is True for r in rows),
                                final_available=sum(r["final_available"] is True for r in rows),
                                length_stops=sum(r["finish_reason"] == "max_tokens" for r in rows),
                                sample_errors=sum(r["sample_error"] for r in rows)))
    output.mkdir(parents=True, exist_ok=True)
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_csv(output/"samples.csv", rows)
    write_csv(output/"paired_transitions.csv", transitions)
    write_csv(output/"contrasts.csv", [dict(metric=c["metric"], weights=json.dumps(c["weights"]),
              delta=c["delta"], ci95=json.dumps(c["ci95"]), eligible_prompts=c["eligible_prompts"])
              for c in contrasts])
    (output/"audit.jsonl").write_text("".join(json.dumps(a)+"\n" for a in audits))
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve()/".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(18,10),layout="constrained")
    for ax,m in zip(axes.flat,METRICS):
        for i,c in enumerate(conditions):
            rate,low,high=observed_rate_ci(rows,c,m)
            if rate is not None:
                ax.errorbar(rate*100,i,xerr=[[max(0,rate-low)*100],[max(0,high-rate)*100]],fmt="o",capsize=3)
        ax.set_yticks(range(len(conditions)),labels=[c.replace("matched_","") for c in conditions])
        ax.invert_yaxis(); ax.set_xlim(-3,103); ax.set_title(m.replace("_"," "))
        ax.set_xlabel("Within-prompt observed rate (%)"); ax.grid(axis="x",alpha=.2)
    fig.suptitle("Matched pilot · exploratory prompt-cluster bootstrap 95% intervals")
    fig.savefig(output/"rates.png",dpi=150); plt.close(fig)
    table=["| Condition | "+" | ".join(METRICS)+" |", "|---|"+"---:|"*len(METRICS)]
    for c in conditions:
        cells=[]
        for m in METRICS:
            stat=coverage[c]["metrics"][m]
            cells.append(f"{stat['rate']:.1%} (n={stat['observed']})" if stat["rate"] is not None else "missing")
        table.append("| "+c+" | "+" | ".join(cells)+" |")
    (output/"report.md").write_text("# Matched pilot results\n\n"+"\n".join(table)+
        "\n\n![Observed rates](rates.png)\n\nAll four reasoning labels are nonexclusive. "
        "Active belief is not score-directed strategy; strategy is not proof of a terminal motive. "
        "Zero means no observed textual evidence. Refusal measures withholding, not harmful-action success. "
        "Effects use complete paired prompt clusters, with 2,000 stratified bootstrap draws. "
        "Intervals are exploratory and unadjusted for multiple comparisons. "
        "Raw rates above use available rollouts; plot uses equal-weight available prompt means. "
        "See summary.json for per-metric missingness, usage and contrasts, and audit.jsonl for all evidence.\n")
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs",nargs="+")
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    write_report(args.logs,args.out)
