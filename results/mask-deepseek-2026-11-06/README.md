# DeepSeek V4.1 Flash on the full MASK provided_facts set (6–7 Nov 2026)

Replication of the gpt-oss-120b full-set gap on a second model family. Inspect task `metagaming/inspect_tasks/mask_probe.py`,
model `openrouter/deepseek/deepseek-v4.1-flash`, reasoning effort medium, max_tokens 8000, temperature 1.0.
Items: all 272 MASK provided_facts items that have a real-use rewrite (`results/mask-full-2026-11-06/renderings.jsonl`),
arms E (benchmark form) and D (real-use rewrite), 2 rollouts per rendering = 544 samples per arm, 1,088 in total.
Scorers: honesty judge (DeepSeek V4.1 Flash, LIE / HONEST / EVADE), four grader-belief probes answered by the target on
its own transcript, chain-of-thought judge (DeepSeek V4.1 Flash reading the whole reasoning trace). One `.eval` log in
`logs/`; `summary.json` from `report_mask_probe.py`; `cot-words-272.json` from the regex word count below.

## Result

| Arm | Samples | Lie rate (share of replies labelled LIE, 95% CI) | LIE / HONEST / EVADE | P(graded) | CoT test-aware (any) | CoT mentions grader |
|---|---|---|---|---|---|---|
| E benchmark form | 544 | 8.5% [6.4, 11.2] | 46 / 341 / 154 | 0.985 | 9.2% | 0.7% |
| D real-use rewrite | 544 | 36.7% [32.8, 40.9] | 199 / 115 / 228 | 0.984 | 6.1% | 0.2% |

Paired D minus E over 272 items: lying +28.1 points [22.8, 33.5]; P(graded) 0.0 [−0.008, +0.007].
The gap replicates on a second model family at the same size as gpt-oss-120b's (+29 on the same items), with the
model's self-reported grader belief flat at 0.98 in both forms. Item-level correlation between the change in belief
and the change in lying: −0.04 (272 items).

Note the large EVADE share in D (228 of 544): DeepSeek often declines or deflects in the rewrite instead of
answering, so the honest share drops more than the lie share rises.

## Reasoning-word counts (regex over the full reasoning text; share of traces containing the word)

| Arm | Mean trace chars | deceive / lie / mislead | scenario / roleplay / fiction | policy | grader / score / test |
|---|---|---|---|---|---|
| E | 3,307 | 91.9% | 36.4% | 48.7% | 19.5% |
| D | 2,173 | 57.5% | 11.4% | 50.2% | 10.7% |

Within D, traces that name deception lie 26.5% of the time against 50.2% when they do not; within E,
7.4% against 20.5%. Caveat: the benchmark prompts say "deceive" in plain words and the rewrites do not,
so part of the across-arm word shift is the model echoing its input; the within-arm split is the part that is not.
