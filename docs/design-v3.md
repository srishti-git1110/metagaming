# Design v3: what causes eval-aware behavior change? (12 October 2026)

## Question

When a model behaves differently because it recognizes an evaluation, which mechanism is
responsible: a cue-triggered habit, a persona, reward seeking, or strategy (Ivanov's
taxonomy)? Prior rounds showed the mechanisms can be dissociated on a harmless format
behavior (GSM8K `\boxed{}`): a reasoning-free habit tied to training-data surface forms, and
a reasoning-dependent response to inferred reward, coexisting in one model. They did not
test an alignment behavior, used self-report to measure awareness, and used a benchmark
that is in every model's training data. Design v3 fixes those three things.

| Mentor's point | How v3 handles it |
|---|---|
| Self-report is not the way to measure eval awareness | Awareness is measured three ways, none of which asks the target model mid-task: (1) **detectability** of each prompt as an evaluation, estimated by external judge models with per-model-calibrated probes and AUROC against deployment negatives (EvalDetectBench method); (2) **spontaneous** eval reasoning in the target's chain of thought, read by a monitor (EvalAwareBench method; a lower bound); (3) the **behavioral gap** itself. |
| Alignment, not capability, benchmarks | Primary behavior: compliance with harmful requests, on HarmBench-derived safety tasks with matched benign counterparts (over-refusal control). Second behavior, agentic: Agentic-Misalignment-style scenarios, where awareness rates are far higher. |
| Measure behavior change, not format | Primary metric: Δ compliance between a naturalistic and an evaluation-looking rendering of the *same* request, task-clustered CIs. Format and CoT labels are secondary. |
| Latest models | Claude Opus 5.5 / Sonnet 5.5, GPT-6.1-sol, GPT-6-luna, Gemini 3.8 Flash, DeepSeek V4.1 Flash, GLM-5.3, Kimi K3; gpt-oss-120b kept as the open-weights model for reasoning-off and white-box work. |
| A benchmark unlikely to be in training data | EvalAwareBench (May 2026, CC-BY-NC, 100 task pairs, 8 toggleable eval-cue factors), plus 50 **new** task pairs we write with the same template recipe and never publish, plus LURE-style deployment realism: each request is appended to a replayed real-usage conversation prefix. |

## Materials

**EvalAwareBench** (`aisa-group/EvalAwareBench`): 100 task names × {safety, capability} ×
256 configurations of eight factors, each toggled naturalistic ↔ evaluative:

| | Factor | Evaluative level | Ivanov mechanism it probes |
|---|---|---|---|
| F1 | Stimulus artificiality | placeholder entities, generic values | habit (surface cue) |
| F2 | Scenario coherence | injected checkable falsehood / too-convenient setup | habit / strategy |
| F3 | Prose register | formal, vignette style | persona |
| F4 | Ecological completeness | stripped context, no workflow | habit |
| F5 | Stakes telegraphing | harm category named outright | strategy / reward |
| F6 | Paradigm recognizability | known benchmark or jailbreak conventions | habit |
| F7 | Task format | rigid template, instrument-style | habit |
| F8 | Explicit evaluation disclosure | eval language, grading-style output format | reward |

Their own finding sets expectations: recognition is common (up to 79% on Agentic
Misalignment for Sonnet 4.5) but ≥77% of recognized instances show **no** behavior change;
F5 and F6 shift Qwen3 and DeepSeek R1 toward caution by 17–25 points on safety tasks.

**Our additions.** (a) 50 new task pairs in domains absent from HarmBench, written with
Claude Opus 5.5 from the EvalAwareBench YAML recipe, held private. (b) Deployment prefixes:
3–6 turns of ordinary assistant conversation (from `HuggingFaceTB/everyday-conversations`
or our own logs) replayed before the request, following LURE. (c) A grading-information
layer on top of F8: stated rule / scored examples (inferred) / no grader / grader rewards
*compliance*, the last being the costly-reward test, since following that reward means
abandoning safety.

**Models.** Frontier: `anthropic/claude-opus-5.5`, `anthropic/claude-sonnet-5.5`,
`openai/gpt-6.1-sol`, `openai/gpt-6-luna`, `google/gemini-3.8-flash`,
`deepseek/deepseek-v4.1-flash`, `z-ai/glm-5.3`, `moonshotai/kimi-k3`. Open weights:
`openai/gpt-oss-120b` (reasoning effort controllable; weights available for probes).

**Judges.** Compliance: `openai/gpt-6-luna` with the HarmBench rubric, 10% human-checked.
Detectability: three judges (one per family) with EvalDetectBench's per-judge probe
calibration; deployment negatives from the EvalDetectBench transcript suite.

## Phase A: is there a gap? (per model, ~2,000 target calls)

100 safety tasks + 100 capability tasks × {all factors naturalistic, all factors evaluative}
× 5 samples. Outputs per model:

- Δ compliance (naturalistic − evaluative) on safety tasks, with task-clustered bootstrap CI.
- Δ refusal on capability tasks (over-refusal under eval cues).
- P(eval) per rendering from the detectability judges, so the gap can be read against how
  detectable the prompt is, not against what the model says.
- Spontaneous eval reasoning rate in CoT where exposed (secondary).

A model with no gap on either behavior has nothing to explain and exits here. The agentic
setting (Agentic Misalignment scenarios, 20 scenarios × 5 samples) runs alongside for the
two or three models with the largest chat gaps, since awareness is far higher there.

## Phase B: which cues carry the gap? (mechanism)

For models with a gap: a 16-run fractional factorial over F1–F8 (main effects + the
interactions F6×F8, F5×F8), 100 safety tasks × 16 configs × 1 sample = 1,600 calls, then
5-sample follow-ups on the significant factors. Readout per Ivanov mechanism:

| Signature | Habit | Persona | Reward seeking | Strategy |
|---|---|---|---|---|
| Gap carried by | F1/F4/F6/F7 (surface cues) | F3 (register), persona system prompts | F8 and the grading layer | F5 + consequence framing |
| Reasoning off (gpt-oss low effort, thinking-off models) | gap unchanged | unchanged | gap shrinks | gap shrinks |
| Grading info changed (none / rewards refusal / rewards compliance, inferred) | no response | no response | gap tracks believed reward, incl. toward compliance | tracks only when consequences follow |
| Detectability P(eval) raised with cues the model does not reason about | gap follows cues, not P(eval) | | gap follows P(eval) | |

The reward test is the inferred-reward design from the GSM8K round, moved to the safety
behavior, with the costly variant: scored examples in which compliant answers scored 1 and
refusals 0, no rule stated. Compliance rising under that evidence, and falling under the
reversed scores, is the reward-seeking signature on a behavior that matters; a helpful or
safety-trained model should ignore it.

## Phase C: is verbalized awareness causal?

For traces that contain spontaneous eval reasoning, resample from the point of the
awareness sentence with it removed, and from matched unaware traces with it inserted
(Goodfire/AISI method). If the gap follows the sentence, awareness is upstream of the
behavior change; if not, the cue acts without it (habit).

## Phase D: white-box (gpt-oss-120b)

Linear probe for the evaluation-awareness direction on Phase A renderings; steer down on
evaluative renderings and up on naturalistic ones; include semantically unrelated control
vectors, since the AISI GLM-5 replication found control vectors with effects as large as
targeted ones. Needs a GPU; last.

## Analysis plan, fixed before running

- Primary: Δ compliance per model, naturalistic vs evaluative, task-clustered bootstrap.
- Secondary: factor main effects; Δ vs P(eval) regression across configurations; reasoning
  on/off interaction; grading-layer effects.
- Awareness is never elicited by asking the target model before or during the behavior call.
- Prompts, judge rubrics and the predictions table above are frozen before inference.

## Cost (one sample per call ≈ 1.5k tokens)

Phase A per model: gpt-6-luna ≈ $1, DeepSeek/GLM ≈ $2, Sonnet 5.5 ≈ $15, Opus 5.5 ≈ $30.
Judging ≈ $3 per model. Phase B ≈ half of Phase A. A full eight-model Phase A plus judging is
about $150; the current balance is $12.81, enough for gpt-6-luna, DeepSeek and gpt-oss first.

## Sources

EvalDetectBench ([arXiv 2609.01611](https://arxiv.org/abs/2609.01611), [code](https://github.com/freeze-lasr/aware_bench));
EvalAwareBench ([arXiv 2605.23055](https://arxiv.org/abs/2605.23055), [data](https://huggingface.co/datasets/aisa-group/EvalAwareBench));
Needham et al. ([arXiv 2505.23836](https://arxiv.org/abs/2505.23836));
LURE ([arXiv 2605.26438](https://arxiv.org/abs/2605.26438));
Goodfire/AISI, [Verbalized Eval Awareness Inflates Measured Safety](https://www.goodfire.com/research/verbalized-eval-awareness-inflates-measured-safety);
Hua et al. ([arXiv 2510.20487](https://arxiv.org/abs/2510.20487)) and the
[AISI replication on GLM-5](https://www.lesswrong.com/posts/HhF5kESdtPHku7kim/reproducing-steering-against-evaluation-awareness-in-a-large-1);
Ivanov, [Why models game evals might matter as much as whether they do](https://www.lesswrong.com/posts/Eub3eLJGMka3asgaP/why-models-game-evals-might-matter-as-much-as-whether-they).
