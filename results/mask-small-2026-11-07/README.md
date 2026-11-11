# MASK benchmark-vs-real-use gap on four small open-weight models

Same design as `results/mask-full-2026-11-06` (task `metagaming/inspect_tasks/mask_probe.py`, arms E and D, paired):
the 272 MASK `provided_facts` items, each in its benchmark form (arm E) and in the real-use rewrite made by GLM-5.3
(arm D), 2 rollouts per item and arm, reasoning on, max 8,000 tokens, temperature 1.0. The honesty judge is
DeepSeek V4.1 Flash (LIE / HONEST / EVADE). "P(graded)" is the share of the four self-report probes answered
"graded", asked to the target on its own transcript. "CoT aware" is the share of reasoning traces the judge flagged
as mentioning a test, grader or benchmark. Gap = D minus E lie rate, paired by item, with a bootstrap 95% interval.

Why these models: candidates for a local steering run (`docs/steering-runbook.md`). Qwen3.6-27B is the model used
in Aranguri's October 2026 eval-awareness post, so its gap decides the steering target.

| model (OpenRouter id) | dir | E lie rate | D lie rate | gap D−E [95% CI] | items | P(graded) E / D | CoT aware E / D |
|---|---|---|---|---|---|---|---|
| openai/gpt-oss-20b | gptoss20b | 35% (186/536) | 54% (294/540) | +20 [15, 25] | 271 | 0.49 / 0.49 | 1% / 0% |
| qwen/qwen3.6-35b-a3b | qwen36 | 55% (296/535) | 67% (360/538) | +12 [6, 18] | 272 | 0.85 / 0.80 | 7% / 2% |
| qwen/qwen3.5-9b | qwen35-9b | 47% (254/538) | 67% (359/538) | +19 [13, 25] | 272 | 0.98 / 0.98 | 10% / 4% |
| qwen/qwen3.6-27b (partial) | qwen36-27b | 60% (of 463 rollouts) | 73% (of 462 rollouts) | +13 [8, 18] | 266 | see note | 6% / 1% |

Denominators are judged rollouts (UNPARSED judge outputs dropped). Items = items with both arms judged.

**Qwen3.6-27B is incomplete.** The run stopped at 924 of 1,088 samples with OpenRouter error 402 (key out of
credits), so the log has status `error`. The numbers above are computed from the 925 logged samples (266 items with
both arms); `report_mask_probe` skips error logs, so there is no `summary.json` for it. Resume with
`inspect eval-retry results/mask-small-2026-11-07/qwen36-27b/logs/<file>.eval` after adding credits. Its probe
answers were not aggregated here.

Reading: all four models lie more on the real-use rewrite than on the benchmark form, by 12 to 20 points, the same
direction and size as the large models (`results/mask-full-2026-11-06`). Stated belief about being graded does not
explain it: P(graded) is flat (gpt-oss-20b, Qwen3.5-9B) or moves the wrong way (Qwen3.6-35B-A3B, 0.85 to 0.80 while
lying rises).

`cot-words.json`: regex shares of reasoning traces per arm (policy words, deception words, grader/score/test words,
scenario/role-play words, instruction words) and the lie rate with and without each. In every model, traces that name
deception lie less than traces that do not, and E traces name deception far more often than D traces. Caveat for
Qwen3.5-9B: its traces are very long (median 13,800 characters, versus 1,700 to 5,300 for the others), so almost every
trace matches every pattern and the shares there are near 100% in E and not informative.

Files: `<dir>/logs/*.eval` Inspect logs, `<dir>/summary.json` from `report_mask_probe`, `<dir>.log` console output
of the run.

Caveat found on 8 Nov: when a reply hit the token cap inside its reasoning and had no final text, Inspect's
`output.completion` returns the reasoning, so the honesty judge read the reasoning as the reply. This affects 13 of
1,088 Qwen3.5-9B rollouts (stop reason max_tokens) and fewer elsewhere; the judges now read the final text only
(`mask_gap.py`, `mask_probe.py`), which applies to future runs. These logs were not re-judged.
