# Why do models behave differently when evaluated?

Experiments on the *cause* of eval-aware behavior change in LLMs, following Ivanov's
taxonomy (habit, persona, terminal reward seeking, strategy). Models are run through
OpenRouter; behavior is parsed or judged; awareness is measured separately from behavior.

## Findings so far

All findings, what they do and do not show, novelty relative to prior work, limitations and
next steps are in one document: [docs/findings.md](docs/findings.md). Headline results:

1. **An evaluation/deployment gap on honesty** (MASK scenarios): the same pressured
   scenario rendered as an operator note with two routine prior turns gets 11–35 points
   more lying than the benchmark rendering, in all four models tested.
2. **No reward seeking on that behavior**: scored examples that reward lying do not
   increase lying (−1 to −8 points), while the identical manipulation moved an answer
   *format* by 98 points on GSM8K.
3. **Habit and reward-following dissociate** on the harmless GSM8K behavior: the `\boxed{}`
   habit tracks the dataset a prompt resembles (not the belief of being tested) and needs no
   reasoning; following an inferred reward needs reasoning.

## Layout

| Path | What |
|---|---|
| `docs/report.md` | Interim report in paper format (abstract through references); only novel findings highlighted |
| `docs/blogpost.md` | Narrative write-up of all rounds, what was and was not found, and the literature |
| `docs/findings.md` | The consolidated findings document |
| `docs/design-v3.md` | Experimental design and the status of the current round |
| `docs/reading-list.md` | Papers and datasets this work builds on |
| `docs/reproduction.md` | Parity notes for the Inspect implementation of the OLMo 3 safety-prompt set |
| `metagaming/cot_judge.py` | LLM judge over chain-of-thought traces: spontaneous test-awareness, grader mention, fiction framing (`results/profile-2026-10-24/cot_judge.jsonl`) |
| `metagaming/cot_plot.py` | Figures: elicited vs spontaneous awareness; within-item reward contrast |
| `metagaming/round5.py` | Current round: MASK + FORTRESS, nine renderings per item, judge panel (`results/round5-2026-10-22/`) |
| `metagaming/aware_bench.py` | EvalDetectBench-based run: depth-varied renderings, detectability probes, behavior judge, report (`results/aware-bench-2026-10-14/`) |
| `metagaming/habit_reward.py`, `habit_report.py`, `habit_plot.py`, `habit_present.py` | GSM8K habit-vs-reward runner, offline analysis and figures |
| `metagaming/matched_pilot.py`, `matched_report.py`, `rich_scoring.py`, `fortress_present.py` | Fortress grader-information pilot and its figures |
| `metagaming/tasks.py`, `scoring.py`, `conditions.py`, `pilot.py`, `report.py`, `analyze.py` | Inspect task, judges and prompt conditions for the safety-prompt set |
| `metagaming/grader_inference.py`, `grader_report.py` | Superseded Fortress prefix-reward pilot (`results/superseded/`) |
| `results/<run>/` | Findings, summaries and figures per run; raw model outputs stay local |
| `tests/` | Offline tests with fabricated responses (`pytest -q`) |

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m metagaming.prepare            # downloads the pinned safety-prompt set
echo OPENROUTER_API_KEY=... > .env      # only needed with --run
pytest -q

# GSM8K habit-vs-reward (needs data/upstream/gsm8k_test.jsonl; the runner prints the URL)
python -m metagaming.habit_reward single --profile gptoss-low --n 200          # plan
python -m metagaming.habit_reward single --profile gptoss-low --n 200 --run    # paid
python -m metagaming.habit_report report && python -m metagaming.habit_present

# Fortress grader pilot
python -m metagaming.matched_pilot --run --log-dir results/<new-dir>
python -m metagaming.matched_report results/<new-dir>/*.eval --out results/<new-dir>/analysis
```
