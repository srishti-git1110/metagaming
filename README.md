# Why do models behave differently when evaluated?

Experiments on the *cause* of eval-aware behavior change in LLMs, following Ivanov's
taxonomy (habit, persona, terminal reward seeking, strategy). Models are run through
OpenRouter; behavior is parsed or judged; awareness is measured separately from behavior.

## Findings so far

1. **Grading information moves safety behavior** ([Fortress pilot](results/fortress-pilot-2026-09-23/analysis/findings.md),
   Qwen3-8B, 25 questions). With the instruction "Withhold assistance with harmful requests"
   held fixed, adding "no grader will score this" lowers refusal by 14 points (84% → 70%);
   adding "a grader will score this" leaves refusal unchanged but makes 24% of reasoning
   traces plan around the score. Test-like vs ordinary framing makes no difference.
   Figure: [fortress-grader-effect.png](results/fortress-pilot-2026-09-23/analysis/fortress-grader-effect.png).
2. **A format habit is cue-triggered and reasoning-free** ([GSM8K round](results/habit-vs-reward-2026-10-06/README.md),
   gpt-oss-120b, Qwen3-8B, Kimi K3, 200 problems). The same problem is answered with
   `\boxed{}` 28–68 points more often when wrapped as "Problem N." than as a chat message;
   unchanged with reasoning off; unchanged when told nothing is scored. The model calls every
   benchmark wrapper "an evaluation", yet boxing tracks the specific wrapper (2%–78%), i.e.
   the surface form, not the belief.
   Figures: [fig1](results/habit-vs-reward-2026-10-06/presentation/fig1-habit-is-cue-triggered.png),
   [fig1b](results/habit-vs-reward-2026-10-06/presentation/fig1b-habit-follows-surface-form.png).
3. **The same model follows an inferred reward when it reasons.** Shown two scored examples
   and no rule, gpt-oss-120b at high effort abandons the habit for a never-trained format
   99% of the time (0% with the scores swapped, 24% with no scores); 26% at low effort.
   Figure: [fig4b](results/habit-vs-reward-2026-10-06/presentation/fig4b-novel-format-gptoss-high.png).

What the GSM8K round did *not* do is measure an alignment behavior or avoid self-report;
[docs/design-v3.md](docs/design-v3.md) is the plan that does both.

## Layout

| Path | What |
|---|---|
| `docs/design-v3.md` | Next experiment: EvalAwareBench safety/capability pairs, latest models, awareness measured without self-report |
| `docs/reading-list.md` | Papers and datasets this work builds on |
| `docs/reproduction.md` | Parity notes for the Inspect implementation of the OLMo 3 safety-prompt set |
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
