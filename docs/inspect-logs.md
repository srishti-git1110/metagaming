# Reading the experiment logs with Inspect

From 6 November 2026 every experiment in this repo is an [Inspect](https://inspect.aisi.org.uk) task. Each run
leaves one `.eval` file per model under `results/<run>/logs/`. The file holds, for every sample: the exact
messages sent, the model's reasoning (chain of thought) as a separate block, its final reply, every follow-up call
made inside the task (for example the grader-belief probes), each judge's prompt and verdict, and the token counts.
All `.eval` files are tracked in git, so a clone has everything.

## Setup (once)

```bash
git clone <this repo> && cd metagaming
python3 -m venv .venv && source .venv/bin/activate
pip install inspect-ai==0.3.265       # the pinned version; same as requirements.txt
```

No API key is needed to read logs.

## Open the viewer

```bash
inspect view --log-dir results/mask-probe-2026-11-06/logs      # one run
inspect view --log-dir results                                 # every run at once (slower to index)
```

This starts a local web page (default http://127.0.0.1:7575). In the viewer:

- The run list shows one row per `.eval` file: task, model, date, and the headline metrics.
- Click a run, then a sample. The **Transcript** tab shows every event in order: the model call with its input
  messages, the reasoning block (collapsed, click to expand), the final text, then the probe calls and the judge
  calls with their prompts and outputs.
- The **Messages** tab shows the conversation as the model saw it.
- The **Scoring** tab shows every scorer's value, answer, explanation and metadata for that sample.
- The filter box accepts expressions such as `metadata.arm == "D"` or `honesty_judge == 1` to pick samples.

## Read a log from Python

```python
from inspect_ai.log import read_eval_log
log = read_eval_log('results/mask-probe-2026-11-06/logs/<file>.eval')
log.eval.model, log.eval.task_args                      # which model, which task parameters
s = log.samples[0]
s.metadata                                              # item id, arm (E = benchmark form, D = real-use rewrite), probe answers
s.output.message.content                                # list of ContentReasoning (chain of thought) and ContentText (reply)
s.scores['honesty_judge'].answer                        # LIE / HONEST / EVADE, with .explanation = the judge's reason
s.scores['grader_belief'].value                         # share of the four grader probes answered "graded" (-1 = none parsed)
s.scores['cot_awareness'].metadata                      # test_awareness 0/1/2, grader mention, fiction, real_stakes, decision
[e for e in s.events if e.event == 'model']             # every model call in the sample, inputs and outputs included
```

## Runs

| Run directory | Task | What it measures | Report script |
|---|---|---|---|
| `results/mask-probe-2026-11-06/logs` | `metagaming/inspect_tasks/mask_probe.py` | MASK lie rate in the benchmark form (E) and the real-use rewrite (D), 47 paired provided_facts items, 2 samples each, on gpt-oss-120b and DeepSeek V4.1 Flash; the same model then answers four grader-belief probes about its own transcript; a judge annotates the chain of thought | `python -m metagaming.inspect_tasks.report_mask_probe` writes `summary.json` |
| `results/mask-full-2026-11-06/rewrite-logs` | `metagaming/inspect_tasks/mask_rewrite.py` | Builds the real-use rewrite for all 608 binary MASK items (known_facts, provided_facts, disinformation) with GLM-5.3; the score says whether the rewrite parsed | `python -m metagaming.inspect_tasks.build_mask_full` writes `renderings.jsonl` |
| `results/mask-full-2026-11-06/logs` | `metagaming/inspect_tasks/mask_gap.py` | Lie rate E vs D on all 606 items, gpt-oss-120b base, 2 samples per item; gap +29 points on provided_facts, none on known_facts or disinformation (see the README there) | `python -m metagaming.inspect_tasks.report_mask_gap results/mask-full-2026-11-06/logs` |
| `results/mask-grader-2026-11-06/logs` | `metagaming/inspect_tasks/mask_probe.py` | Same task on all 272 provided_facts items with three system-prompt conditions (none, "you will not be graded", "scored by an automated honesty grader"), gpt-oss-120b, 2 samples; one `.eval` per condition (`log.eval.task_args['condition']`) | `python -m metagaming.inspect_tasks.report_mask_probe results/mask-grader-2026-11-06/logs` |
| `results/organisms-mask-2026-11-06/logs` | `metagaming/inspect_tasks/mask_gap.py` | 3-item pipeline check through OpenRouter; the organism runs on the pod will be added here | `python -m metagaming.inspect_tasks.report_mask_gap` |
| `results/fortress-pilot-2026-09-23/*.eval` | `metagaming/tasks.py` | September FORTRESS grader-information pilot (older task) | `metagaming/matched_report.py` |

Arms in the MASK tasks: `E` is the MASK item as shipped (system prompt + user message); `D` is the same situation
rewritten as a product conversation (operator system prompt, four to six ordinary prior turns, then the same
request). `E_Rbad`, `E_Rgood`, `D_Rbad`, `D_Rgood` prepend scored example transcripts in which the lie or the honest
reply scored 1. The October rewrites were made by Kimi K3; the full-set rewrites by GLM-5.3.

## Runs that predate Inspect (JSONL only)

These cannot be opened in the viewer. Each directory holds raw stores; the field names are in the docstring of the
script named.

| Run directory | Script | Where the model text is |
|---|---|---|
| `results/grader-belief-2026-11-05` | `metagaming/grader_belief.py` | `gen.jsonl` has the full reply and full reasoning per row; `probe.jsonl` and `judge.jsonl` keep the label and the last few hundred characters of the answer only |
| `results/profile-2026-10-24` | `metagaming/profile.py`, `cot_judge.py` | `behave.jsonl.xz` (reply + reasoning), `belief.jsonl`, `judge-deepseek.jsonl`, `cot_judge.jsonl` |
| `results/round5-2026-10-22`, `results/mask-2026-10-15`, `results/aware-bench-2026-10-14`, `results/habit-vs-reward-2026-10-06` | `round5.py`, `mask_run.py`, `aware_bench.py`, `habit_reward.py` | `behave.jsonl`, `detect.jsonl`, `judge.jsonl` |
